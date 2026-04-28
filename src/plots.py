"""
plots.py - Visualization utilities for breast density training

This script handles plotting and visualization of training metrics.
It creates and saves plots for loss curves, dice coefficients, and regression errors.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.manifold import TSNE


def create_tsne_plot(
    embeddings: np.ndarray,
    birads_labels: np.ndarray,
    output_path: str,
    filename: str = "tsne_plot.png"
) -> str:
    """
    Perform t-SNE on the given embeddings and create a 2D scatter plot.

    Args:
        embeddings (np.ndarray): Shape (N, feature_dim). Embeddings from the model.
        birads_labels (np.ndarray): Shape (N,). Discrete BI-RADS categories (1..4).
        output_path (str): Directory to save the plot.
        filename (str): Name of the output image file (PNG).

    Returns:
        str: The path where the t-SNE plot is saved.
    """
    # 1) Run t-SNE (2D)
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(embeddings)  # shape => (N,2)

    # 2) Plot
    plt.figure(figsize=(8, 6))
    # Typically you’d see multiple BI-RADS categories, so we can color by each category
    unique_categories = np.unique(birads_labels)
    for category in unique_categories:
        idx = np.where(birads_labels == category)
        plt.scatter(
            tsne_result[idx, 0],
            tsne_result[idx, 1],
            label=f'BI-RADS {category}',
            alpha=0.7,
            s=30
        )
    plt.title("t-SNE of Mammogram Embeddings by BI-RADS")
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')

    # 3) Save
    plot_dir = Path(output_path)
    plot_dir.mkdir(parents=True, exist_ok=True)
    save_path = plot_dir / filename
    plt.savefig(str(save_path), dpi=300, bbox_inches='tight')
    plt.close()
    return str(save_path)


def create_training_plots(
    train_loss,
    valid_loss,
    train_dice,
    valid_dice,
    train_reg_mae,
    valid_reg_mae,
    output_path,
    timestamp
):
    """
    Create and save training curves.

    Args:
        train_loss (list): Training loss values per epoch
        valid_loss (list): Validation loss values per epoch
        train_dice (list): Training dice coefficient values per epoch
        valid_dice (list): Validation dice coefficient values per epoch
        train_reg_mae (list): Training regression MAE values per epoch
        valid_reg_mae (list): Validation regression MAE values per epoch
        output_path (str): Directory where plots will be saved
        timestamp (str): Timestamp for unique filename

    Returns:
        str: Path to saved plot file
    """
    epochs_range = range(1, len(train_loss) + 1)

    plt.figure(figsize=(15, 5))
    plt.style.use('default')  # Reset style

    # Apply custom styling
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['grid.linestyle'] = '--'

    # Loss curve
    plt.subplot(1, 3, 1)
    plt.plot(epochs_range, train_loss, color='#8A2BE2', label='Train Loss')  # Purple
    plt.plot(epochs_range, valid_loss, color='#FF69B4', label='Valid Loss')  # Pink
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')

    # Dice coefficient curve
    plt.subplot(1, 3, 2)
    plt.plot(epochs_range, train_dice, color='#8A2BE2', label='Train Dice')  # Purple
    plt.plot(epochs_range, valid_dice, color='#FF69B4', label='Valid Dice')  # Pink
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.title('Segmentation Performance')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')

    # Regression MAE curve
    plt.subplot(1, 3, 3)
    plt.plot(epochs_range, train_reg_mae, color='#8A2BE2', label='Train MAE')  # Purple
    plt.plot(epochs_range, valid_reg_mae, color='#FF69B4', label='Valid MAE')  # Pink
    plt.xlabel('Epochs')
    plt.ylabel('MAE (%)')
    plt.title('Density Prediction Error')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Ensure output directory exists
    plot_dir = Path(output_path).parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Save plot with timestamp
    plot_filename = f"training_curves_{timestamp}.png"
    plot_path = plot_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plot_path)


def create_distribution_plot(train_densities, valid_densities, output_path, timestamp):
    """
    Create histograms showing the distribution of breast density percentages in the dataset.

    Args:
        train_densities (list): List of breast density percentages in training set
        valid_densities (list): List of breast density percentages in validation set
        output_path (str): Directory where plots will be saved
        timestamp (str): Timestamp for unique filename

    Returns:
        str: Path to saved plot file
    """
    plt.figure(figsize=(10, 6))

    # Create bins corresponding to BI-RADS categories
    bins = [0, 25, 50, 75, 100]
    labels = ['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4']

    # Histogram for training set
    plt.hist(
        train_densities,
        bins=bins,
        alpha=0.5,
        color='#8A2BE2',  # Purple
        label='Training Set'
    )

    # Histogram for validation set
    plt.hist(
        valid_densities,
        bins=bins,
        alpha=0.5,
        color='#FF69B4',  # Pink
        label='Validation Set'
    )

    # Add vertical lines for BI-RADS boundaries
    for boundary in [25, 50, 75]:
        plt.axvline(x=boundary, color='gray', linestyle='--', alpha=0.7)

    # Add category labels
    for i, label in enumerate(labels):
        plt.text(
            (bins[i] + bins[i+1])/2,
            plt.ylim()[1] * 0.9,
            label,
            horizontalalignment='center',
            fontsize=10
        )

    plt.xlabel('Breast Density Percentage (%)')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Breast Density in Dataset')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend()

    # Save plot
    plot_dir = Path(output_path).parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = f"density_distribution_{timestamp}.png"
    plot_path = plot_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plot_path)


def create_confusion_matrix(predictions, ground_truth, output_path, timestamp):
    """
    Create a confusion matrix for BI-RADS classifications.

    Args:
        predictions (list): Predicted BI-RADS categories (1-4)
        ground_truth (list): Ground truth BI-RADS categories (1-4)
        output_path (str): Directory where plots will be saved
        timestamp (str): Timestamp for unique filename

    Returns:
        str: Path to saved plot file
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Calculate confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=[1, 2, 3, 4])

    # Normalize by row (true labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.round(cm_normalized * 100, 1)  # Convert to percentages

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.1f',
        cmap='PuRd',  # Purple-pink colormap
        xticklabels=['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4'],
        yticklabels=['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4']
    )

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('BI-RADS Classification Confusion Matrix (%)')

    # Save plot
    plot_dir = Path(output_path).parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = f"birads_confusion_matrix_{timestamp}.png"
    plot_path = plot_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plot_path)

def create_epoch_confusion_matrices(val_predictions, val_targets, best_epoch, final_epoch, output_path, timestamp):
    """
    Create confusion matrices for both best and final epochs.

    Args:
        val_predictions (list of lists): Predictions for each epoch
        val_targets (list of lists): Targets for each epoch
        best_epoch (int): Index of best epoch
        final_epoch (int): Index of final epoch
        output_path (str): Directory where plots will be saved
        timestamp (str): Timestamp for unique filename

    Returns:
        tuple: Paths to saved best and final confusion matrices
    """
    # Create best epoch confusion matrix
    best_preds = val_predictions[best_epoch]
    best_targets = val_targets[best_epoch]
    best_cm_path = create_confusion_matrix(
        best_preds,
        best_targets,
        output_path,
        f"{timestamp}_best_epoch{best_epoch+1}"
    )

    # Create final epoch confusion matrix
    final_preds = val_predictions[final_epoch]
    final_targets = val_targets[final_epoch]
    final_cm_path = create_confusion_matrix(
        final_preds,
        final_targets,
        output_path,
        f"{timestamp}_final_epoch{final_epoch+1}"
    )

    return best_cm_path, final_cm_path


def create_learning_rate_plot(lr_history, output_path, timestamp):
    """
    Create a plot showing learning rate changes over epochs.

    Args:
        lr_history (list): Learning rate values per epoch
        output_path (str): Directory where plot will be saved
        timestamp (str): Timestamp for unique filename

    Returns:
        str: Path to saved plot file
    """
    plt.figure(figsize=(10, 6))

    # Learning rate curve
    plt.plot(range(1, len(lr_history) + 1), lr_history, color='#4B0082')  # Indigo

    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.grid(True, alpha=0.3, linestyle='--')

    # Save plot
    plot_dir = Path(output_path).parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = f"learning_rate_{timestamp}.png"
    plot_path = plot_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plot_path)


def visualize_segmentations(model, dataloader, device, output_path, timestamp, num_samples=4):
    """
    Create visualizations of segmentation predictions vs. ground truth.

    Args:
        model (nn.Module): Trained model
        dataloader (DataLoader): Validation dataloader
        device (torch.device): Device to run inference on
        output_path (str): Path to save visualization
        timestamp (str): Timestamp for unique filename
        num_samples (int): Number of samples to visualize

    Returns:
        str: Path to saved visualization
    """
    import torch
    import numpy as np

    model.eval()

    # Get samples
    sample_images = []
    sample_masks = []
    sample_preds = []

    with torch.no_grad():
        # Get one batch
        for images, masks, _, _ in dataloader:
            images = images.to(device)

            # Get predictions
            outputs = model(images)
            if isinstance(outputs, tuple):
                seg_outputs = outputs[0]
            else:
                seg_outputs = outputs

            # Process samples
            for i in range(min(num_samples, len(images))):
                # Process image for display
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-8)

                # Process mask
                if masks.shape[1] == 3:  # Multi-channel mask
                    msk = masks[i, 0].cpu().numpy()  # First channel
                else:
                    msk = masks[i, 0].cpu().numpy()

                # Normalize mask if needed
                if msk.max() > 1.0:
                    msk = msk / 255.0

                # Process prediction
                pred = seg_outputs[i, 0].cpu().numpy()
                pred_binary = (pred > 0.5).astype(np.float32)

                sample_images.append(img)
                sample_masks.append(msk)
                sample_preds.append(pred_binary)

            # Only process one batch
            break

    # Create visualization
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 4 * num_samples))

    for i in range(len(sample_images)):
        # Original image
        axes[i, 0].imshow(sample_images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')

        # Ground truth mask
        axes[i, 1].imshow(sample_masks[i], cmap='gray')
        axes[i, 1].set_title('Ground Truth Mask')
        axes[i, 1].axis('off')

        # Prediction mask
        axes[i, 2].imshow(sample_preds[i], cmap='gray')
        axes[i, 2].set_title('Predicted Mask')
        axes[i, 2].axis('off')

    plt.tight_layout()

    # Save visualization
    plot_dir = Path(output_path).parent
    plot_dir.mkdir(parents=True, exist_ok=True)

    plot_filename = f"segmentation_viz_{timestamp}.png"
    plot_path = plot_dir / plot_filename
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plot_path)
