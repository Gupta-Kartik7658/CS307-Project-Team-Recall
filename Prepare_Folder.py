import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class NoiseDatasetGenerator:
    def __init__(self, clean_images_dir, output_dir, num_noise_samples=100):
        """
        Args:
            clean_images_dir: Path to folder containing pure/clean images
            output_dir: Base output directory
            num_noise_samples: Number of noisy variants per image per noise type
        """
        self.clean_images_dir = clean_images_dir
        self.output_dir = output_dir
        self.num_noise_samples = num_noise_samples
        
        # Create directory structure
        self.noisy_images_dir = os.path.join(output_dir, "noisy_images")
        self.dataset_csv = os.path.join(output_dir, "dataset.csv")
        
        self.noise_types = ["gaussian", "salt_pepper", "poisson"]
        self._create_directories()
        
    def _create_directories(self):
        """Create output directory structure"""
        os.makedirs(self.noisy_images_dir, exist_ok=True)
        for noise_type in self.noise_types:
            os.makedirs(os.path.join(self.noisy_images_dir, noise_type), exist_ok=True)
        print(f"✓ Directory structure created at {self.output_dir}")
    
    def add_gaussian_noise(self, image, intensity):
        """Add Gaussian noise with varying intensity"""
        noise = np.random.normal(0, intensity, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def add_salt_pepper_noise(self, image, intensity):
        """Add Salt & Pepper noise"""
        noisy = image.copy().astype(np.float32)
        # intensity represents probability of noise (0-1 range mapped to percentage)
        prob = intensity / 255.0  # Map 0-255 to 0-1
        
        salt_pepper = np.random.random(image.shape[:2])
        noisy[salt_pepper < prob/2] = 255  # Salt
        noisy[salt_pepper < prob] = 0      # Pepper
        
        return noisy.astype(np.uint8)
    
    def add_poisson_noise(self, image, intensity):
        """Add Poisson noise with varying intensity"""
        # Scale image for Poisson distribution
        image_float = image.astype(np.float32) / 255.0
        scaled = image_float * intensity
        
        # Generate Poisson noise
        noisy = np.random.poisson(scaled) / intensity * 255.0
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    def extract_features(self, image):
        """Extract statistical features from image"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Flatten for statistics
        pixels = gray.flatten()
        
        # Extract features
        features = {
            'mean': np.mean(pixels),
            'std': np.std(pixels),
            'variance': np.var(pixels),
            'skewness': (np.mean((pixels - np.mean(pixels))**3)) / (np.std(pixels)**3 + 1e-8),
            'kurtosis': (np.mean((pixels - np.mean(pixels))**4)) / (np.std(pixels)**4 + 1e-8) - 3,
            'median': np.median(pixels),
            'min': np.min(pixels),
            'max': np.max(pixels),
            'q25': np.percentile(pixels, 25),
            'q75': np.percentile(pixels, 75),
            'entropy': self._calculate_entropy(gray),
        }
        
        # Laplacian variance (edge detection) - high for Gaussian, low for smooth
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)
        
        # Gradient magnitude
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        features['gradient_mean'] = np.mean(gradient_mag)
        features['gradient_std'] = np.std(gradient_mag)
        
        return features
    
    def _calculate_entropy(self, image):
        """Calculate image entropy"""
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy
    
    def generate_dataset(self):
        """Generate all noisy images and create dataset CSV"""
        print(f"\n📊 Starting dataset generation...")
        print(f"   - Clean images directory: {self.clean_images_dir}")
        print(f"   - Noise samples per type: {self.num_noise_samples}")
        print(f"   - Total expected samples: {len(self.noise_types) * self.num_noise_samples}")
        
        dataset_rows = []
        image_counter = 0
        
        # Get all clean images
        clean_images = [f for f in os.listdir(self.clean_images_dir) 
                       if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        
        print(f"\n📁 Found {len(clean_images)} clean images\n")
        
        for img_idx, clean_img_name in enumerate(clean_images, 1):
            clean_img_path = os.path.join(self.clean_images_dir, clean_img_name)
            
            try:
                # Read clean image
                clean_image = cv2.imread(clean_img_path)
                if clean_image is None:
                    print(f"⚠️  Skipping {clean_img_name} - couldn't read")
                    continue
                
                # Resize to consistent size (optional, comment out if not needed)
                clean_image = cv2.resize(clean_image, (256, 256))
                
                print(f"[{img_idx}/{len(clean_images)}] Processing: {clean_img_name}")
                
                # Generate noisy variants for each noise type
                for noise_type in self.noise_types:
                    for sample_idx in range(self.num_noise_samples):
                        # Vary intensity parameter
                        if noise_type == "gaussian":
                            intensity = np.random.uniform(5, 50)  # std dev
                            noisy_image = self.add_gaussian_noise(clean_image, intensity)
                        
                        elif noise_type == "salt_pepper":
                            intensity = np.random.uniform(5, 50)  # mapped to probability
                            noisy_image = self.add_salt_pepper_noise(clean_image, intensity)
                        
                        elif noise_type == "poisson":
                            intensity = np.random.uniform(10, 100)  # Poisson parameter
                            noisy_image = self.add_poisson_noise(clean_image, intensity)
                        
                        # Save noisy image
                        output_filename = f"{clean_img_name.split('.')[0]}_noise_{sample_idx:03d}.png"
                        output_path = os.path.join(self.noisy_images_dir, noise_type, output_filename)
                        cv2.imwrite(output_path, noisy_image)
                        
                        # Extract features
                        features = self.extract_features(noisy_image)
                        
                        # Create row for CSV
                        row = {
                            'image_name': output_filename,
                            'noise_type': noise_type,
                            'clean_image': clean_img_name,
                            'intensity': intensity,
                        }
                        row.update(features)
                        dataset_rows.append(row)
                        
                        image_counter += 1
                        
                        # Progress indicator
                        if (sample_idx + 1) % 25 == 0:
                            print(f"  ✓ {noise_type}: {sample_idx + 1}/{self.num_noise_samples}")
                
            except Exception as e:
                print(f"❌ Error processing {clean_img_name}: {str(e)}")
        
        # Create DataFrame and save CSV
        print(f"\n💾 Creating dataset CSV...")
        df = pd.DataFrame(dataset_rows)
        df.to_csv(self.dataset_csv, index=False)
        
        print(f"✅ Dataset generation complete!")
        print(f"\n📈 Dataset Statistics:")
        print(f"   - Total images generated: {image_counter}")
        print(f"   - Dataset shape: {df.shape}")
        print(f"   - Class distribution:\n{df['noise_type'].value_counts()}")
        print(f"\n📁 Outputs saved:")
        print(f"   - Noisy images: {self.noisy_images_dir}")
        print(f"   - Dataset CSV: {self.dataset_csv}")
        
        return df


# ============ USAGE ============

if __name__ == "__main__":
    # Configure these paths
    CLEAN_IMAGES_DIR = "./Dataset"  # Folder with your 15 clean images
    OUTPUT_DIR = "./noise_dataset"  # Output folder
    NUM_NOISE_SAMPLES = 100  # Samples per noise type per image
    
    # Create generator
    generator = NoiseDatasetGenerator(
        clean_images_dir=CLEAN_IMAGES_DIR,
        output_dir=OUTPUT_DIR,
        num_noise_samples=NUM_NOISE_SAMPLES
    )
    
    # Generate dataset
    df = generator.generate_dataset()
    
    print(f"\n🎉 Dataset ready for XGBoost training!")
    print(f"CSV location: {generator.dataset_csv}")