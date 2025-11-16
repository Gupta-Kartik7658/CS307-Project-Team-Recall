
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)
plt.rcParams['font.size'] = 10


class DataVisualizer:
    """Visualize noise dataset and feature distributions"""
    
    def __init__(self, csv_path="./noise_dataset/dataset.csv"):
        """
        Args:
            csv_path: Path to dataset CSV
        """
        print("📊 Loading dataset...")
        self.df = pd.read_csv(csv_path)
        print(f"✅ Dataset loaded: {self.df.shape[0]} samples, {self.df.shape[1]} features")
        
        self.noise_types = self.df['noise_type'].unique()
        self.feature_cols = [col for col in self.df.columns 
                            if col not in ['image_name', 'noise_type', 'clean_image', 'intensity']]
        
        print(f"✅ Found {len(self.feature_cols)} features")
        print(f"✅ Noise types: {', '.join(self.noise_types)}\n")
    
    def plot_class_distribution(self):
        """Plot distribution of noise types"""
        print("📈 Creating class distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Count plot
        counts = self.df['noise_type'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        axes[0].bar(counts.index, counts.values, color=colors, edgecolor='black', linewidth=2)
        axes[0].set_title('Class Distribution (Count)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Noise Type', fontsize=12)
        axes[0].set_ylabel('Number of Images', fontsize=12)
        axes[0].grid(axis='y', alpha=0.3)
        
        for i, v in enumerate(counts.values):
            axes[0].text(i, v + 20, str(v), ha='center', fontweight='bold')
        
        # Pie chart
        axes[1].pie(counts.values, labels=counts.index, autopct='%1.1f%%',
                   colors=colors, explode=(0.05, 0.05, 0.05), startangle=90,
                   textprops={'fontsize': 11, 'fontweight': 'bold'})
        axes[1].set_title('Class Distribution (Percentage)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('./visualizations/01_class_distribution.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 01_class_distribution.png\n")
        plt.close()
    
    def plot_feature_distributions(self):
        """Plot distribution of each feature by noise type"""
        print("📈 Creating feature distributions plot...")
        
        n_features = len(self.feature_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        colors = {'gaussian': '#FF6B6B', 'salt_pepper': '#4ECDC4', 'poisson': '#45B7D1'}
        
        for idx, feature in enumerate(self.feature_cols):
            ax = axes[idx]
            
            for noise_type in self.noise_types:
                data = self.df[self.df['noise_type'] == noise_type][feature]
                ax.hist(data, bins=30, alpha=0.6, label=noise_type, color=colors[noise_type])
            
            ax.set_title(f'Distribution of {feature}', fontsize=11, fontweight='bold')
            ax.set_xlabel(feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.legend()
            ax.grid(alpha=0.3)
        
        # Remove empty subplots
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('./visualizations/02_feature_distributions.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 02_feature_distributions.png\n")
        plt.close()
    
    def plot_boxplots(self):
        """Create boxplots for each feature by noise type"""
        print("📈 Creating boxplots...")
        
        n_features = len(self.feature_cols)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
        axes = axes.flatten()
        
        for idx, feature in enumerate(self.feature_cols):
            ax = axes[idx]
            
            data_to_plot = [self.df[self.df['noise_type'] == nt][feature].values 
                           for nt in self.noise_types]
            
            bp = ax.boxplot(data_to_plot, labels=self.noise_types, patch_artist=True)
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
            ax.set_ylabel(feature, fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        
        for idx in range(n_features, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plt.savefig('./visualizations/03_boxplots.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 03_boxplots.png\n")
        plt.close()
    
    def plot_violin_plots(self):
        """Create violin plots for key distinguishing features"""
        print("📈 Creating violin plots...")
        
        # Select top 8 most important features
        important_features = ['std', 'entropy', 'kurtosis', 'laplacian_var',
                             'gradient_mean', 'skewness', 'variance', 'gradient_std']
        
        fig, axes = plt.subplots(2, 4, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(important_features):
            ax = axes[idx]
            
            sns.violinplot(data=self.df, x='noise_type', y=feature, ax=ax,
                          palette={'gaussian': '#FF6B6B', 'salt_pepper': '#4ECDC4', 'poisson': '#45B7D1'})
            
            ax.set_title(f'{feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Noise Type', fontsize=10)
            ax.set_ylabel(feature, fontsize=10)
            ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./visualizations/04_violin_plots.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 04_violin_plots.png\n")
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of features by noise type"""
        print("📈 Creating correlation heatmaps...")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        for idx, noise_type in enumerate(self.noise_types):
            data = self.df[self.df['noise_type'] == noise_type][self.feature_cols]
            corr_matrix = data.corr()
            
            sns.heatmap(corr_matrix, ax=axes[idx], cmap='coolwarm', center=0,
                       square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
            axes[idx].set_title(f'Feature Correlation - {noise_type.upper()}', 
                               fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('./visualizations/05_correlation_heatmap.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 05_correlation_heatmap.png\n")
        plt.close()
    
    def plot_feature_statistics(self):
        """Create detailed statistics visualization"""
        print("📈 Creating feature statistics...")
        
        stats_data = []
        
        for noise_type in self.noise_types:
            subset = self.df[self.df['noise_type'] == noise_type][self.feature_cols]
            stats_data.append({
                'Noise Type': noise_type,
                'Mean (avg)': subset.mean().mean(),
                'Std (avg)': subset.std().mean(),
                'Variance (avg)': subset.var().mean(),
            })
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Mean values comparison
        ax = axes[0, 0]
        for noise_type in self.noise_types:
            means = self.df[self.df['noise_type'] == noise_type][self.feature_cols].mean()
            ax.bar(range(len(means)), means.values, alpha=0.7, label=noise_type)
        ax.set_title('Mean Feature Values by Noise Type', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index', fontsize=10)
        ax.set_ylabel('Mean Value', fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Std values comparison
        ax = axes[0, 1]
        for noise_type in self.noise_types:
            stds = self.df[self.df['noise_type'] == noise_type][self.feature_cols].std()
            ax.bar(range(len(stds)), stds.values, alpha=0.7, label=noise_type)
        ax.set_title('Std Feature Values by Noise Type', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature Index', fontsize=10)
        ax.set_ylabel('Std Value', fontsize=10)
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Summary statistics table
        ax = axes[1, 0]
        ax.axis('tight')
        ax.axis('off')
        
        summary_df = pd.DataFrame(stats_data)
        table = ax.table(cellText=summary_df.round(4).values,
                        colLabels=summary_df.columns,
                        cellLoc='center', loc='center',
                        colColours=['#E8E8E8']*len(summary_df.columns))
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Feature count
        ax = axes[1, 1]
        ax.text(0.5, 0.8, f'Total Features: {len(self.feature_cols)}', 
               ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.65, f'Total Samples: {len(self.df)}', 
               ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.5, f'Noise Types: {len(self.noise_types)}', 
               ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.text(0.5, 0.35, f'Samples/Type: {len(self.df) // len(self.noise_types)}', 
               ha='center', fontsize=14, fontweight='bold', transform=ax.transAxes)
        ax.axis('off')
        
        plt.tight_layout()
        plt.savefig('./visualizations/06_feature_statistics.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 06_feature_statistics.png\n")
        plt.close()
    
    def plot_pairplot(self):
        """Create pairplot of top features"""
        print("📈 Creating pairplot (this may take a moment)...")
        
        # Select top 5 discriminative features
        top_features = ['std', 'entropy', 'kurtosis', 'laplacian_var', 'gradient_mean']
        
        plot_df = self.df[top_features + ['noise_type']].copy()
        
        colors = {'gaussian': '#FF6B6B', 'salt_pepper': '#4ECDC4', 'poisson': '#45B7D1'}
        
        pp = sns.pairplot(plot_df, hue='noise_type', diag_kind='kde',
                         palette=colors, plot_kws={'alpha': 0.6, 's': 30},
                         diag_kws={'alpha': 0.7})
        
        pp.fig.suptitle('Pairplot of Top 5 Discriminative Features', 
                       fontsize=14, fontweight='bold', y=1.001)
        
        plt.tight_layout()
        plt.savefig('./visualizations/07_pairplot.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 07_pairplot.png\n")
        plt.close()
    
    def plot_intensity_analysis(self):
        """Analyze how features vary with noise intensity"""
        print("📈 Creating intensity analysis...")
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        
        key_features = ['std', 'entropy', 'kurtosis', 'laplacian_var', 'variance', 'gradient_mean']
        colors = {'gaussian': '#FF6B6B', 'salt_pepper': '#4ECDC4', 'poisson': '#45B7D1'}
        
        for idx, feature in enumerate(key_features):
            ax = axes[idx]
            
            for noise_type in self.noise_types:
                subset = self.df[self.df['noise_type'] == noise_type].sort_values('intensity')
                ax.scatter(subset['intensity'], subset[feature], 
                          alpha=0.5, s=30, label=noise_type, color=colors[noise_type])
                
                # Add trend line
                z = np.polyfit(subset['intensity'], subset[feature], 2)
                p = np.poly1d(z)
                x_trend = np.linspace(subset['intensity'].min(), subset['intensity'].max(), 100)
                ax.plot(x_trend, p(x_trend), color=colors[noise_type], linewidth=2, alpha=0.8)
            
            ax.set_title(f'{feature} vs Noise Intensity', fontsize=11, fontweight='bold')
            ax.set_xlabel('Noise Intensity', fontsize=10)
            ax.set_ylabel(feature, fontsize=10)
            ax.legend()
            ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./visualizations/08_intensity_analysis.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 08_intensity_analysis.png\n")
        plt.close()
    
    def plot_3d_feature_space(self):
        """Create 3D visualization of feature space"""
        print("📈 Creating 3D feature space visualization...")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = {'gaussian': '#FF6B6B', 'salt_pepper': '#4ECDC4', 'poisson': '#45B7D1'}
        
        for noise_type in self.noise_types:
            subset = self.df[self.df['noise_type'] == noise_type]
            ax.scatter(subset['std'], subset['entropy'], subset['kurtosis'],
                      c=colors[noise_type], label=noise_type, s=50, alpha=0.6)
        
        ax.set_xlabel('Std', fontsize=11, fontweight='bold')
        ax.set_ylabel('Entropy', fontsize=11, fontweight='bold')
        ax.set_zlabel('Kurtosis', fontsize=11, fontweight='bold')
        ax.set_title('3D Feature Space: std vs entropy vs kurtosis', 
                    fontsize=12, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('./visualizations/09_3d_feature_space.png', dpi=300, bbox_inches='tight')
        print("✅ Saved: 09_3d_feature_space.png\n")
        plt.close()
    
    def generate_summary_report(self):
        """Generate text summary of dataset"""
        print("📄 Generating summary report...")
        
        report = f"""
{'='*70}
NOISE DATASET - SUMMARY REPORT
{'='*70}

DATASET OVERVIEW
{'-'*70}
Total Samples:          {len(self.df)}
Total Features:         {len(self.feature_cols)}
Noise Types:            {len(self.noise_types)}
Clean Images Used:      15
Variants per Image:     100
Noise per Type:         {len(self.df) // len(self.noise_types)}

NOISE TYPE DISTRIBUTION
{'-'*70}
"""
        
        for noise_type in self.noise_types:
            count = len(self.df[self.df['noise_type'] == noise_type])
            pct = (count / len(self.df)) * 100
            report += f"{noise_type:15s}: {count:4d} samples ({pct:5.1f}%)\n"
        
        report += f"\nFEATURES EXTRACTED\n{'-'*70}\n"
        for i, feat in enumerate(self.feature_cols, 1):
            report += f"{i:2d}. {feat}\n"
        
        report += f"\n\nFEATURE STATISTICS BY NOISE TYPE\n{'-'*70}\n"
        
        for noise_type in self.noise_types:
            subset = self.df[self.df['noise_type'] == noise_type][self.feature_cols]
            report += f"\n{noise_type.upper()}:\n"
            report += f"  Mean:     {subset.mean().mean():.4f}\n"
            report += f"  Std:      {subset.std().mean():.4f}\n"
            report += f"  Min:      {subset.min().min():.4f}\n"
            report += f"  Max:      {subset.max().max():.4f}\n"
        
        report += f"\n{'='*70}\n"
        
        with open('./visualizations/dataset_summary_report.txt', 'w') as f:
            f.write(report)
        
        print(report)
        print("✅ Saved: dataset_summary_report.txt\n")
    
    def run_all_visualizations(self):
        """Run all visualizations"""
        print("\n" + "="*70)
        print("🎨 GENERATING ALL VISUALIZATIONS")
        print("="*70 + "\n")
        
        # Create visualizations directory
        import os
        os.makedirs('./visualizations', exist_ok=True)
        
        self.plot_class_distribution()
        self.plot_feature_distributions()
        self.plot_boxplots()
        self.plot_violin_plots()
        self.plot_correlation_heatmap()
        self.plot_feature_statistics()
        self.plot_pairplot()
        self.plot_intensity_analysis()
        self.plot_3d_feature_space()
        self.generate_summary_report()
        
        print("="*70)
        print("✅ ALL VISUALIZATIONS COMPLETE!")
        print("="*70)
        print("\n📁 Saved to ./visualizations/")
        print("   - 01_class_distribution.png")
        print("   - 02_feature_distributions.png")
        print("   - 03_boxplots.png")
        print("   - 04_violin_plots.png")
        print("   - 05_correlation_heatmap.png")
        print("   - 06_feature_statistics.png")
        print("   - 07_pairplot.png")
        print("   - 08_intensity_analysis.png")
        print("   - 09_3d_feature_space.png")
        print("   - dataset_summary_report.txt\n")


if __name__ == "__main__":
    # Initialize visualizer
    viz = DataVisualizer(csv_path="./noise_dataset/dataset.csv")
    
    # Run all visualizations
    viz.run_all_visualizations()