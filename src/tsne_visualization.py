import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap

def create_purple_pink_colormap():
    """
    Create a custom purple to pink colormap for visualization
    """
    # Define colors from dark purple to light pink
    colors = [
        '#2E0854',  # Dark purple
        '#4B0082',  # Indigo
        '#8A2BE2',  # BlueViolet
        '#9370DB',  # MediumPurple
        '#DA70D6',  # Orchid
        '#FF69B4',  # HotPink
        '#FFB6C1'   # LightPink
    ]
    
    # Create and return the colormap
    return LinearSegmentedColormap.from_list('purple_pink', colors)


class FeatureExtractor(nn.Module):
    """
    Module to extract features from the CNN encoder for tSNE visualization
    """
    def __init__(self, model, use_deep_decoder=True):
        super().__init__()
        self.model = model
        self.use_deep_decoder = use_deep_decoder
        
        # Determine where to extract features from
        if use_deep_decoder:
            if hasattr(model, 'encoder'):
                self.encoder = model.encoder
            elif hasattr(model, 'base_model') and hasattr(model.base_model, 'encoder'):
                self.encoder = model.base_model.encoder
            else:
                raise ValueError("Could not find encoder in model")
        else:
            if hasattr(model, 'seg_model'):
                if hasattr(model.seg_model, 'encoder'):
                    self.encoder = model.seg_model.encoder
                elif hasattr(model.seg_model, 'module') and hasattr(model.seg_model.module, 'encoder'):
                    self.encoder = model.seg_model.module.encoder
                else:
                    raise ValueError("Could not find encoder in segmentation model")
            else:
                raise ValueError("Could not find segmentation model")

        # Add average pooling to reduce feature dimensions
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
    def forward(self, x):
        """
        Extract features from the deep layers of the CNN
        Returns both pooled features (for tSNE) and segmentation output
        """
        # Get features from encoder
        with torch.no_grad():
            features = self.encoder(x)
        
        # Get deepest features
        deep_features = features[-1]
        
        # Spatial average pooling to get a feature vector
        pooled_features = self.avg_pool(deep_features)
        pooled_features = pooled_features.flatten(1)
        
        # Get segmentation output from full model
        with torch.no_grad():
            outputs = self.model(x)
            
        if isinstance(outputs, tuple):
            # Handle different model output formats
            if len(outputs) >= 3:
                seg_outputs, reg_outputs, cls_outputs = outputs[:3]
            elif len(outputs) == 2:
                seg_outputs, reg_outputs = outputs
                cls_outputs = None
            else:
                seg_outputs = outputs[0]
                reg_outputs = cls_outputs = None
        else:
            seg_outputs = outputs
            reg_outputs = cls_outputs = None
            
        # Handle tuple segmentation outputs
        if isinstance(seg_outputs, tuple):
            seg_outputs = seg_outputs[0]
            
        return {
            'features': pooled_features,
            'seg_outputs': seg_outputs,
            'reg_outputs': reg_outputs,
            'cls_outputs': cls_outputs
        }


def extract_features_and_metadata(model, dataloader, device, max_samples=500):
    """
    Extract features and metadata from the dataset for tSNE visualization
    
    Args:
        model: Trained model
        dataloader: DataLoader with samples
        device: Device to run model on
        max_samples: Maximum number of samples to process
        
    Returns:
        dict: Dictionary with features and metadata
    """
    # Create feature extractor
    feature_extractor = FeatureExtractor(model, use_deep_decoder=True)
    feature_extractor.eval()
    feature_extractor.to(device)
    
    # Containers for data
    all_features = []
    all_density_values = []
    all_birads_categories = []
    all_patient_ids = []  # If available in dataset
    
    # Track samples processed
    processed_samples = 0
    
    # Process batches
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            # Get inputs and targets
            if len(batch) == 4:  # (images, masks, densities, birads)
                images, _, densities, birads = batch
            elif len(batch) == 5:  # (patient_ids, images, masks, densities, birads)
                patient_ids, images, _, densities, birads = batch
                all_patient_ids.extend(patient_ids)
                
            # Move to device
            images = images.to(device)
            
            # Calculate how many samples to take from this batch
            batch_size = images.size(0)
            samples_needed = min(max_samples - processed_samples, batch_size)
            
            if samples_needed <= 0:
                break
                
            # Extract features for the needed number of samples
            outputs = feature_extractor(images[:samples_needed])
            features = outputs['features']
            
            # Store features and metadata
            all_features.append(features.cpu().numpy())
            all_density_values.append(densities[:samples_needed].cpu().numpy() * 100)  # Convert to percentage
            all_birads_categories.append(birads[:samples_needed].cpu().numpy() + 1)  # Convert 0-index to 1-index
            
            # Update counter
            processed_samples += samples_needed
            
            if processed_samples >= max_samples:
                break
                
    # Concatenate data
    features = np.vstack(all_features) if all_features else np.array([])
    density_values = np.concatenate(all_density_values) if all_density_values else np.array([])
    birads_categories = np.concatenate(all_birads_categories) if all_birads_categories else np.array([])
    
    # Return as dictionary
    result = {
        'features': features,
        'density_values': density_values,
        'birads_categories': birads_categories,
    }
    
    if all_patient_ids:
        result['patient_ids'] = all_patient_ids[:processed_samples]
        
    return result


def create_tsne_visualization(features_data, output_dir, timestamp, perplexity=30, n_iter=1000):
    """
    Create tSNE visualization from extracted features
    
    Args:
        features_data: Dictionary with features and metadata
        output_dir: Directory to save visualizations
        timestamp: Timestamp for filenames
        perplexity: tSNE perplexity parameter
        n_iter: Number of iterations for tSNE
        
    Returns:
        dict: Dictionary with paths to saved visualizations
    """
    # Create output directory
    plots_dir = Path(output_dir) / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    features = features_data['features']
    density_values = features_data['density_values']
    birads_categories = features_data['birads_categories']
    
    # Check if we have data
    if len(features) == 0:
        print("No features data available for tSNE visualization")
        return {}
        
    print(f"Creating tSNE visualization with {len(features)} samples...")
    
    # Optional: Use PCA first to reduce dimensions if feature space is very high-dimensional
    if features.shape[1] > 50:
        print(f"Reducing feature dimensionality from {features.shape[1]} to 50 with PCA...")
        pca = PCA(n_components=50)
        features = pca.fit_transform(features)
    
    # Apply tSNE
    tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
    tsne_result = tsne.fit_transform(features)
    
    # Create output paths
    density_path = plots_dir / f"tsne_density_{timestamp}.png"
    birads_path = plots_dir / f"tsne_birads_{timestamp}.png"
    
    # Create custom colormap
    purple_pink = create_purple_pink_colormap()
    
    # Plot by density
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        tsne_result[:, 0], 
        tsne_result[:, 1], 
        c=density_values, 
        cmap=purple_pink,
        alpha=0.8, 
        s=50
    )
    
    plt.colorbar(scatter, label='Breast Density Percentage')
    plt.title('tSNE Visualization of Breast Density Data')
    plt.xlabel('tSNE Dimension 1')
    plt.ylabel('tSNE Dimension 2')
    plt.grid(alpha=0.3)
    
    # Save density plot
    plt.savefig(density_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot by BI-RADS category with discrete colormap
    plt.figure(figsize=(10, 8))
    
    # Create categorical colormap
    category_cmap = plt.cm.get_cmap('PuRd', 4)  # Purple-Red colormap with 4 discrete colors
    
    scatter = plt.scatter(
        tsne_result[:, 0], 
        tsne_result[:, 1], 
        c=birads_categories, 
        cmap=category_cmap,
        alpha=0.8, 
        s=50,
        vmin=1,
        vmax=4
    )
    
    # Add custom colorbar with BI-RADS labels
    cbar = plt.colorbar(scatter, ticks=[1.375, 2.125, 2.875, 3.625])
    cbar.set_ticklabels(['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4'])
    
    plt.title('tSNE Visualization by BI-RADS Category')
    plt.xlabel('tSNE Dimension 1')
    plt.ylabel('tSNE Dimension 2')
    plt.grid(alpha=0.3)
    
    # Save BI-RADS plot
    plt.savefig(birads_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create a more informative visualization: BI-RADS with distribution density
    plt.figure(figsize=(12, 10))
    
    # Create plots for each BI-RADS category
    for category in range(1, 5):
        # Get indices for this category
        indices = birads_categories == category
        
        if np.sum(indices) > 0:
            # Get tSNE points for this category
            points = tsne_result[indices]
            
            # Plot with kernel density estimate
            ax = sns.kdeplot(
                x=points[:, 0], 
                y=points[:, 1],
                levels=5,
                fill=True,
                alpha=0.4,
                cmap=[f"PuRd_{category}"], 
                label=f"BI-RADS {category}"
            )
    
    # Also add scatter plot with small points
    plt.scatter(
        tsne_result[:, 0], 
        tsne_result[:, 1], 
        c=birads_categories, 
        cmap=category_cmap,
        alpha=0.5, 
        s=20,
        vmin=1,
        vmax=4
    )
    
    plt.title('tSNE Distribution by BI-RADS Category')
    plt.xlabel('tSNE Dimension 1')
    plt.ylabel('tSNE Dimension 2')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Save distribution plot
    distribution_path = plots_dir / f"tsne_distribution_{timestamp}.png"
    plt.savefig(distribution_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'density_plot': str(density_path),
        'birads_plot': str(birads_path),
        'distribution_plot': str(distribution_path)
    }


# Function to add to your training script
def generate_tsne_visualizations(model, valid_dataloader, device, output_dir, timestamp, num_samples=500):
    """
    Generate tSNE visualizations from a trained model
    
    Args:
        model: Trained model
        valid_dataloader: Validation dataloader
        device: Device to run on
        output_dir: Directory to save visualizations
        timestamp: Timestamp for filenames
        num_samples: Maximum number of samples to process
        
    Returns:
        dict: Dictionary with paths to saved visualizations
    """
    print("\nGenerating tSNE visualizations...")
    
    # Extract features and metadata
    features_data = extract_features_and_metadata(
        model, valid_dataloader, device, max_samples=num_samples
    )
    
    # Create visualizations
    vis_paths = create_tsne_visualization(
        features_data, output_dir, timestamp
    )
    
    return vis_paths
