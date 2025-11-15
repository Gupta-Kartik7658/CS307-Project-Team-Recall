import os
import cv2
import numpy as np
import pickle
import xgboost as xgb
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class FeatureExtractor:
    """Extract statistical features from images"""
    
    @staticmethod
    def extract_features(image):
        """Extract 15 statistical features from image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        pixels = gray.flatten()
        
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
            'entropy': FeatureExtractor._calculate_entropy(gray),
        }
        
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features['laplacian_var'] = np.var(laplacian)
        
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        features['gradient_mean'] = np.mean(gradient_mag)
        features['gradient_std'] = np.std(gradient_mag)
        
        return features
    
    @staticmethod
    def _calculate_entropy(image):
        hist, _ = np.histogram(image, bins=256, range=(0, 256))
        hist = hist / hist.sum()
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        return entropy


class NoiseAdder:
    """Add different types of noise to images"""
    
    @staticmethod
    def gaussian(image, intensity):
        noise = np.random.normal(0, intensity, image.shape)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy
    
    @staticmethod
    def salt_pepper(image, intensity):
        noisy = image.copy().astype(np.float32)
        prob = intensity / 255.0
        
        salt_pepper = np.random.random(image.shape[:2])
        noisy[salt_pepper < prob/2] = 255
        noisy[salt_pepper < prob] = 0
        
        return noisy.astype(np.uint8)
    
    @staticmethod
    def poisson(image, intensity):
        image_float = image.astype(np.float32) / 255.0
        scaled = image_float * intensity
        
        noisy = np.random.poisson(scaled) / intensity * 255.0
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        return noisy


class NoiseClassifier:
    """Load model and classify noise types"""
    
    def __init__(self, model_dir="./noise_classifier_model"):
        self.model_dir = model_dir
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.feature_names = None
        
        self._load_components()
    
    def _load_components(self):
        print("\n" + "="*70)
        print("🔄 LOADING MODEL COMPONENTS")
        print("="*70)
        
        try:
            # Load model
            model_path = os.path.join(self.model_dir, 'xgboost_noise_classifier.pkl')
            self.model = xgb.XGBClassifier()
            self.model.load_model(model_path)
            print(f"✅ Model loaded")
            
            # Load label encoder
            le_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            with open(le_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✅ Label encoder loaded")
            
            # Load scaler
            scaler_path = os.path.join(self.model_dir, 'scaler.pkl')
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✅ Scaler loaded")
            
            # Load feature names
            features_path = os.path.join(self.model_dir, 'feature_names.pkl')
            with open(features_path, 'rb') as f:
                self.feature_names = pickle.load(f)
            print(f"✅ Feature names loaded ({len(self.feature_names)} features)")
            
            print("\n✅ ALL COMPONENTS LOADED SUCCESSFULLY!\n")
            
        except Exception as e:
            print(f"\n❌ ERROR LOADING MODEL: {e}")
            raise
    
    def classify(self, image):
        """Classify noise type in image"""
        # Extract features
        features = FeatureExtractor.extract_features(image)
        
        # Create feature vector
        feature_vector = np.array([features[fname] for fname in self.feature_names]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Predict
        pred_encoded = self.model.predict(feature_vector_scaled)[0]
        pred_label = self.label_encoder.inverse_transform([int(pred_encoded)])[0]
        
        # Get probabilities
        pred_proba = self.model.predict_proba(feature_vector_scaled)[0]
        
        # Results
        results = {
            'predicted_noise': pred_label,
            'probabilities': {},
            'confidence': float(pred_proba[int(pred_encoded)]),
            'features': features
        }
        
        for cls, prob in zip(self.label_encoder.classes_, pred_proba):
            results['probabilities'][cls] = float(prob)
        
        return results
    
    def print_results(self, results):
        """Pretty print classification results"""
        print("\n" + "="*70)
        print("🎯 CLASSIFICATION RESULTS")
        print("="*70)
        
        print(f"\n🔍 PREDICTED NOISE TYPE: {results['predicted_noise'].upper()}")
        print(f"📊 CONFIDENCE: {results['confidence']:.2%}")
        
        print(f"\n📈 PROBABILITY DISTRIBUTION:")
        for noise_type in sorted(results['probabilities'].keys()):
            prob = results['probabilities'][noise_type]
            bar_length = int(prob * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"   {noise_type:15s} | {bar} | {prob:.4f}")
        
        print(f"\n📊 EXTRACTED FEATURES:")
        for feat_name in sorted(results['features'].keys()):
            feat_value = results['features'][feat_name]
            print(f"   {feat_name:20s}: {feat_value:12.6f}")
        
        print("\n" + "="*70 + "\n")


def main():
    print("\n" + "="*70)
    print("🤖 NOISE TYPE CLASSIFIER - INFERENCE")
    print("="*70)
    
    # Load classifier
    try:
        classifier = NoiseClassifier(model_dir="./noise_classifier_model")
    except Exception as e:
        print(f"Failed to load classifier: {e}")
        return
    
    noise_methods = {
        '1': ('gaussian', NoiseAdder.gaussian),
        '2': ('salt_pepper', NoiseAdder.salt_pepper),
        '3': ('poisson', NoiseAdder.poisson)
    }
    
    while True:
        print("\n" + "-"*70)
        print("MENU:")
        print("-"*70)
        print("  [1] Add GAUSSIAN NOISE and classify")
        print("  [2] Add SALT & PEPPER NOISE and classify")
        print("  [3] Add POISSON NOISE and classify")
        print("  [4] Classify existing noisy image")
        print("  [5] EXIT")
        print("-"*70)
        
        choice = input("\n👉 Enter choice (1-5): ").strip()
        
        if choice == '5':
            print("\n👋 Exiting...\n")
            break
        
        if choice in ['1', '2', '3']:
            # Get clean image path
            image_path = input("\n📁 Enter path to CLEAN image: ").strip()
            
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue
            
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Cannot read image: {image_path}")
                continue
            
            # Resize for consistency
            image = cv2.resize(image, (256, 256))
            
            # Get intensity
            try:
                intensity_input = input("\n🎚️  Enter noise intensity (recommended: 10-50): ").strip()
                intensity = float(intensity_input)
            except ValueError:
                print("❌ Invalid intensity value")
                continue
            
            # Add noise
            noise_name, noise_func = noise_methods[choice]
            print(f"\n⏳ Adding {noise_name.upper()} noise (intensity={intensity})...")
            
            noisy_image = noise_func(image, intensity)
            
            # Save
            output_path = f"./noisy_{noise_name}_{np.random.randint(1000, 9999)}.png"
            cv2.imwrite(output_path, noisy_image)
            print(f"✅ Noisy image saved: {output_path}")
            
            # Classify
            print(f"\n🔍 Classifying image...")
            results = classifier.classify(noisy_image)
            classifier.print_results(results)
        
        elif choice == '4':
            # Classify existing image
            image_path = input("\n📁 Enter path to noisy image: ").strip()
            
            if not os.path.exists(image_path):
                print(f"❌ File not found: {image_path}")
                continue
            
            image = cv2.imread(image_path)
            if image is None:
                print(f"❌ Cannot read image: {image_path}")
                continue
            
            image = cv2.resize(image, (256, 256))
            
            print(f"\n🔍 Classifying image...")
            results = classifier.classify(image)
            classifier.print_results(results)
        
        else:
            print("❌ Invalid choice!")


if __name__ == "__main__":
    main()