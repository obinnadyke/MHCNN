"""
losses_metrics.py - metrics and loss functions for breast density analysis |(c) itrustal.com
Provisions: 
1. Evaluation metrics (Dice, IoU, accuracy, etc.)
2. Custom loss functions (FocalTverskyLoss, FocalRegLoss, MultiTaskLoss)
3. BI-RADS classification utilities (percentage to category conversion) 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path


def dice_coef(pred, target, smooth=1e-6, threshold=0.5):
    """
    Calculate Dice coefficient for segmentation, handling various input shapes and formats
    """
    # 1) If pred or target is a tuple/list, pick the first element
    if isinstance(pred, (list, tuple)):
        if len(pred) >= 1:
            pred = pred[0]
        else:
            raise ValueError("dice_coef received an empty tuple/list for pred.")
    if isinstance(target, (list, tuple)):
        if len(target) >= 1:
            target = target[0]
        else:
            raise ValueError("dice_coef received an empty tuple/list for target.")

    # 2) Convert 5D -> 4D for pred
    # e.g. [B,1,H,W,3], keep only channel 0 => [B,1,H,W]
    if pred.dim() == 5:
        if pred.shape[-1] == 3:  # e.g. [B,1,H,W,3]
            pred = pred[..., 0]  # => now [B,1,H,W]
        else:
            raise ValueError(
                f"Unsupported 5D shape for pred={pred.shape} in dice_coef. "
                "If you have more channels, adapt slicing logic accordingly."
            )

    # 3) Convert 5D -> 4D for target
    if target.dim() == 5:
        if target.shape[-1] == 3:  # e.g. [B,1,H,W,3]
            target = target[..., 0]  # => now [B,1,H,W]
        else:
            raise ValueError(
                f"Unsupported 5D shape for target={target.shape} in dice_coef. "
                "If you have more channels, adapt slicing logic accordingly."
            )

    # 4) If [B,3,H,W], pick only channel 0 => [B,1,H,W]
    if pred.dim() == 4 and pred.shape[1] > 1:
        pred = pred[:, 0:1, :, :]
    if target.dim() == 4 and target.shape[1] > 1:
        target = target[:, 0:1, :, :]

    # 5) Final shape check
    if pred.shape != target.shape:
        raise ValueError(
            f"Shape mismatch in dice_coef: pred={pred.shape}, target={target.shape}"
        )

    # 6) Normalize if target has values outside 0-1 range (e.g., 0-255)
    if target.max() > 1.0:
        # Normalize to 0-1 range
        # Using 255.0 specifically if values appear to be in 0-255 range
        if target.max() > 200:
            target = target / 255.0
        else:
            target = target / target.max()

    # 7) Convert types to float for calculation
    pred = pred.float()
    target = target.float()

    # 8) Threshold -> binary (both pred and target for consistency)
    pred = (pred > threshold).float()
    target = (target > threshold).float()  # Also threshold target to ensure binary values

    # 9) Compute dice with safeguards
    intersection = (pred * target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()

    # 10) Handle empty masks case
    if pred_sum < smooth and target_sum < smooth:
        return 1.0  # Both masks empty, consider perfect match

    dice = (2.0 * intersection + smooth) / (pred_sum + target_sum + smooth)

    # 11) Ensure result is in [0,1] range
    dice = torch.clamp(dice, 0.0, 1.0)

    # 12) Return a Python float
    return dice.item()


def iou_score(pred, target, smooth=1e-6, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) for segmentation
    """
    # Handle tuple inputs
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if isinstance(target, (list, tuple)):
        target = target[0]

    pred = pred.float()
    target = target.float()

    # Normalize target if needed
    if target.max() > 1.0:
        if target.max() > 200:
            target = target / 255.0
        else:
            target = target / target.max()

    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


def accuracy(pred, target, threshold=0.5):
    """
    Calculate pixel-wise accuracy for segmentation
    Args:
        pred (torch.Tensor): Predicted mask
        target (torch.Tensor): Ground truth mask
        threshold (float): Threshold for binary segmentation
    Returns:
        float: Accuracy score
    """
    # Handle tuple inputs
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if isinstance(target, (list, tuple)):
        target = target[0]

    pred = pred.float()
    target = target.float()

    # Normalize target if needed
    if target.max() > 1.0:
        if target.max() > 200:
            target = target / 255.0
        else:
            target = target / target.max()

    pred = (pred > threshold).float()
    target = (target > threshold).float()

    correct = (pred == target).float().sum()
    total = target.numel()

    return (correct / total).item()


def sensitivity(pred, target, smooth=1e-6, threshold=0.5):
    """
    Calculate sensitivity (recall) for segmentation
    """
    # Handle tuple inputs
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if isinstance(target, (list, tuple)):
        target = target[0]

    pred = pred.float()
    target = target.float()

    # Normalize target if needed
    if target.max() > 1.0:
        if target.max() > 200:
            target = target / 255.0
        else:
            target = target / target.max()

    pred = (pred > threshold).float()
    target = (target > threshold).float()

    intersection = (pred * target).sum()

    # Handle empty target mask
    if target.sum() < smooth:
        return 1.0 if pred.sum() < smooth else 0.0

    return (intersection + smooth) / (target.sum() + smooth)


def specificity(pred, target, smooth=1e-6, threshold=0.5):
    """
    Calculate specificity for segmentation 
    """
    # Handle tuple inputs
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    if isinstance(target, (list, tuple)):
        target = target[0]

    pred = pred.float()
    target = target.float()

    # Normalize target if needed
    if target.max() > 1.0:
        if target.max() > 200:
            target = target / 255.0
        else:
            target = target / target.max()

    pred = (pred > threshold).float()
    target = (target > threshold).float()

    true_neg = ((1 - pred) * (1 - target)).sum()

    # Handle all-positive target mask
    if (1 - target).sum() < smooth:
        return 1.0 if (1 - pred).sum() < smooth else 0.0

    return (true_neg + smooth) / ((1 - target).sum() + smooth)


def percentage_to_birads_category(percentage):
    """
    Convert density percentage to BI-RADS category number
    Args - percentage (float): Breast density percentage (0-100)
    Returns - int: BI-RADS category (1-4)
    """
    if percentage < 25:
        return 1            # "BI-RADS A (Almost entirely fatty)"
    elif percentage < 50:
        return 2            # "BI-RADS B (Scattered fibroglandular)"
    elif percentage < 75:
        return 3            # "BI-RADS C (Heterogeneously dense)"
    else:
        return 4            # "BI-RADS D (Extremely dense)"


def get_birads_from_percentage(percentage):
    """
    Convert density percentage to BI-RADS category and description
    Args - percentage (float): Breast density percentage (0-100)
    Returns - tuple: (category, description)
    """
    category = percentage_to_birads_category(percentage)

    if category == 1:
        return category, "Almost entirely fatty"
    elif category == 2:
        return category, "Scattered fibroglandular"
    elif category == 3:
        return category, "Heterogeneously dense"
    else:
        return category, "Extremely dense"


def compute_metrics_batch(pred_masks, true_masks, pred_densities=None, true_densities=None,
                         pred_birads=None, true_birads=None):
    """
    Compute all relevant metrics for a batch of predictions
    Args:
        pred_masks (torch.Tensor): Predicted segmentation masks
        true_masks (torch.Tensor): Ground truth segmentation masks
        pred_densities (torch.Tensor, optional): Predicted density percentages
        true_densities (torch.Tensor, optional): Ground truth density percentages
        pred_birads (torch.Tensor, optional): Predicted BI-RADS classes
        true_birads (torch.Tensor, optional): Ground truth BI-RADS classes
    Returns:
        dict: Dictionary of computed metrics
    """
    metrics = {}

    # Segmentation metrics
    metrics['dice'] = dice_coef(pred_masks, true_masks)
    metrics['iou'] = iou_score(pred_masks, true_masks)
    metrics['accuracy'] = accuracy(pred_masks, true_masks)
    metrics['sensitivity'] = sensitivity(pred_masks, true_masks)
    metrics['specificity'] = specificity(pred_masks, true_masks)

    # Regression metrics (if provided)
    if pred_densities is not None and true_densities is not None:
        # Convert to percentages (0-100)
        pred_percentages = pred_densities * 100
        true_percentages = true_densities * 100

        # Mean Absolute Error
        metrics['reg_mae'] = torch.abs(pred_percentages - true_percentages).mean().item()

        # Mean Squared Error
        metrics['reg_mse'] = ((pred_percentages - true_percentages) ** 2).mean().item()

        # RMSE
        metrics['reg_rmse'] = torch.sqrt(((pred_percentages - true_percentages) ** 2).mean()).item()

        # BI-RADS classification accuracy from regression
        if pred_birads is None and true_birads is None:
            # Vectorized operation instead of Python loops
            pred_percentages_np = pred_percentages.detach().cpu().numpy()
            true_percentages_np = true_percentages.detach().cpu().numpy()

            # Vectorized category assignment
            def percentage_to_category_vec(percentages):
                categories = np.ones_like(percentages, dtype=int)
                categories[percentages >= 25] = 2
                categories[percentages >= 50] = 3
                categories[percentages >= 75] = 4
                return categories

            pred_categories = percentage_to_category_vec(pred_percentages_np)
            true_categories = percentage_to_category_vec(true_percentages_np)

            # Calculate accuracy
            correct = np.sum(pred_categories == true_categories)
            metrics['birads_accuracy'] = (correct / len(pred_categories)) * 100

            # Add confusion matrix
            try:
                metrics['birads_cm'] = confusion_matrix(
                    true_categories.flatten(),
                    pred_categories.flatten(),
                    labels=[1, 2, 3, 4]
                )
            except Exception as e:
                print(f"Error creating confusion matrix: {e}")
                metrics['birads_cm'] = None

    # Direct BI-RADS classification metrics (if provided)
    if pred_birads is not None and true_birads is not None:
        # Use argmax for predicted class if in logits form
        if len(pred_birads.shape) > 1 and pred_birads.shape[1] > 1:
            pred_birads_classes = torch.argmax(pred_birads, dim=1)
        else:
            pred_birads_classes = pred_birads

        correct = (pred_birads_classes == true_birads).sum().item()
        total = true_birads.size(0)
        metrics['birads_accuracy'] = (correct / total) * 100

        # Add confusion matrix
        try:
            # Convert to 1-indexed BI-RADS categories for clarity (model uses 0-indexed)
            pred_np = (pred_birads_classes + 1).cpu().numpy()
            true_np = (true_birads + 1).cpu().numpy()

            metrics['birads_cm'] = confusion_matrix(
                true_np,
                pred_np,
                labels=[1, 2, 3, 4]
            )
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            metrics['birads_cm'] = None

    return metrics


def create_confusion_matrix_plot(cm, output_path, title='BI-RADS Confusion Matrix'):
    """
    Create and save a confusion matrix plot 
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4'],
                yticklabels=['BI-RADS 1', 'BI-RADS 2', 'BI-RADS 3', 'BI-RADS 4'])

    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    return output_path


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for handling imbalanced segmentation 
    """
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        # Ensure inputs are float tensors
        y_pred = y_pred.float()
        y_true = y_true.float()

        # Handling multi-channel masks by taking first channel
        if y_true.dim() == 4 and y_true.shape[1] > 1:
            y_true = y_true[:, 0:1, :, :]

        # Normalize values if needed (for 0-255 range masks)
        if y_true.max() > 1.0:
            if y_true.max() > 200:  # Likely 0-255 range
                y_true = y_true / 255.0
            else:
                y_true = y_true / y_true.max()

        # Flatten tensors for calculation
        batch_size = y_pred.size(0)
        y_pred = y_pred.reshape(batch_size, -1)
        y_true = y_true.reshape(batch_size, -1)

        # Calculate true positives, false positives, false negatives
        tp = torch.sum(y_pred * y_true, dim=1)
        fp = torch.sum(y_pred * (1 - y_true), dim=1)
        fn = torch.sum((1 - y_pred) * y_true, dim=1)

        # Tversky index - batch-wise calculation
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Apply focal component with numerical stability
        focal_tversky = torch.pow(1.0 - tversky, self.gamma)

        # Return mean across batch
        return focal_tversky.mean()


class FocalRegLoss(nn.Module):
    """
    Focal Regression Loss with numerical stability improvements
    """
    def __init__(self, gamma=2.0, reduction='mean', eps=1e-6):
        super(FocalRegLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps  # Add small epsilon to prevent division by zero
        self.mse = nn.MSELoss(reduction='none')

    def forward(self, pred, target):
        """
        Calculate focal regression loss with numerical stability improvements
        """
        # Ensure inputs are float tensors (prevent int division)
        pred = pred.float()
        target = target.float()

        # Clip prediction to small positive range to avoid NaN in backward
        pred = torch.clamp(pred, min=self.eps, max=1.0-self.eps)

        # Base MSE loss
        mse_loss = self.mse(pred, target)

        # Apply focal weighting with safeguards against extremely large values
        # Use torch.clamp to prevent overflow/underflow
        focal_weight = torch.exp(torch.clamp(self.gamma * mse_loss, min=-20, max=20))

        # Normalize the focal weights to prevent explosion
        mean_focal_weight = torch.mean(focal_weight) + self.eps
        focal_weight = focal_weight / mean_focal_weight

        # Weighted loss
        focal_loss = focal_weight * mse_loss

        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss


class MultiTaskLoss(nn.Module):
    """
    Combined loss for segmentation, regression, and classification with safeguards 
    """
    def __init__(self, seg_loss, reg_loss, lambda_reg=0.2, lambda_cls=0.3, device=None):
        super(MultiTaskLoss, self).__init__()

        # Set device (default to CUDA if available)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Store loss functions
        self.seg_loss = seg_loss
        self.reg_loss = reg_loss

        # Balanced weight for less frequent classifications 
        class_weights = torch.tensor([1.5, 1.2, 1.0, 1.3], device=self.device)
        self.cls_loss = nn.CrossEntropyLoss(weight=class_weights)  # For BI-RADS classification

        # Loss weights
        self.lambda_reg = lambda_reg
        self.lambda_cls = lambda_cls

        # Move loss functions to the correct device if they have parameters
        if hasattr(self.cls_loss, 'to') and callable(getattr(self.cls_loss, 'to')):
            self.cls_loss = self.cls_loss.to(self.device)

    def forward(self, pred_seg, target_seg, pred_reg, target_reg, pred_cls=None, target_cls=None):
        """
        Calculate multi-task loss with safeguards against negative values 
        """
        # Handle tuple outputs (common in some segmentation models)
        if isinstance(pred_seg, tuple):
            # Use the first element of the tuple (usually the main prediction)
            pred_seg = pred_seg[0]
        if isinstance(target_seg, tuple):
            target_seg = target_seg[0]

        # Ensure all inputs are on the correct device
        pred_seg = pred_seg.to(self.device)
        target_seg = target_seg.to(self.device)
        pred_reg = pred_reg.to(self.device)
        target_reg = target_reg.to(self.device)

        # Calculate segmentation loss
        loss_seg = self.seg_loss(pred_seg, target_seg)

        # IMPORTANT: Ensure segmentation loss is positive
        loss_seg = torch.abs(loss_seg)

        # Calculate regression loss
        loss_reg = self.reg_loss(pred_reg, target_reg)

        # Ensure regression loss is positive too
        loss_reg = torch.abs(loss_reg)

        # Calculate classification loss if provided
        if pred_cls is not None and target_cls is not None:
            pred_cls = pred_cls.to(self.device)
            target_cls = target_cls.to(self.device)
            loss_cls = self.cls_loss(pred_cls, target_cls)

            # Combined loss with all three components (all guaranteed to be positive)
            total_loss = loss_seg + self.lambda_reg * loss_reg + self.lambda_cls * loss_cls

            return total_loss, loss_seg, loss_reg, loss_cls
        else:
            # Legacy mode - only segmentation and regression
            total_loss = loss_seg + self.lambda_reg * loss_reg

            # Return a zero tensor for cls_loss to maintain consistent return signature
            zero_loss = torch.tensor(0.0, device=self.device)

            return total_loss, loss_seg, loss_reg, zero_loss


def create_epoch_confusion_matrices(val_predictions_history, val_targets_history,
                                   best_epoch_idx, final_epoch_idx, logs_file_path, timestamp):
    """
    Generate confusion matrix plots for best and final epochs
    """
    plot_dir = Path(logs_file_path).parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Create best epoch confusion matrix
    if 0 <= best_epoch_idx < len(val_predictions_history):
        best_preds = val_predictions_history[best_epoch_idx]
        best_targets = val_targets_history[best_epoch_idx]

        if len(best_preds) > 0 and len(best_targets) > 0:
            try:
                cm = confusion_matrix(best_targets, best_preds, labels=[1, 2, 3, 4])
                best_cm_path = plot_dir / f"confusion_matrix_best_epoch{best_epoch_idx+1}_{timestamp}.png"
                create_confusion_matrix_plot(
                    cm,
                    str(best_cm_path),
                    f"BI-RADS Confusion Matrix (Best Epoch {best_epoch_idx+1})"
                )
            except Exception as e:
                print(f"Error creating best epoch confusion matrix: {e}")
                best_cm_path = None
        else:
            best_cm_path = None
    else:
        best_cm_path = None

    # Create final epoch confusion matrix
    if 0 <= final_epoch_idx < len(val_predictions_history):
        final_preds = val_predictions_history[final_epoch_idx]
        final_targets = val_targets_history[final_epoch_idx]

        if len(final_preds) > 0 and len(final_targets) > 0:
            try:
                cm = confusion_matrix(final_targets, final_preds, labels=[1, 2, 3, 4])
                final_cm_path = plot_dir / f"confusion_matrix_final_epoch{final_epoch_idx+1}_{timestamp}.png"
                create_confusion_matrix_plot(
                    cm,
                    str(final_cm_path),
                    f"BI-RADS Confusion Matrix (Final Epoch {final_epoch_idx+1})"
                )
            except Exception as e:
                print(f"Error creating final epoch confusion matrix: {e}")
                final_cm_path = None
        else:
            final_cm_path = None
    else:
        final_cm_path = None

    return best_cm_path, final_cm_path


def create_learning_rate_plot(lr_history, logs_file_path, timestamp):
    """
    Create learning rate plot to visualize learning rate changes during training 
    """
    plot_dir = Path(logs_file_path).parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(lr_history) + 1), lr_history)
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)

    plot_path = plot_dir / f"learning_rate_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plot_path)


def visualize_segmentations(model, dataloader, device, logs_file_path, timestamp, num_samples=3):
    """
    Create segmentation visualization plots
    """
    plot_dir = Path(logs_file_path).parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    model.eval()

    # Get samples from dataloader
    samples = []
    with torch.no_grad():
        for images, masks, densities, birads in dataloader:
            # Only take a few samples
            batch_size = min(num_samples, images.size(0))
            if batch_size < images.size(0):
                images = images[:batch_size]
                masks = masks[:batch_size]
                densities = densities[:batch_size]
                birads = birads[:batch_size]

            images = images.to(device)
            masks = masks.to(device)

            # Forward pass through model
            seg_outputs, reg_outputs, cls_outputs = model(images)

            if isinstance(seg_outputs, tuple):
                seg_outputs = seg_outputs[0]

            # Get predicted BI-RADS from classifier
            _, predicted_classes = torch.max(cls_outputs, 1)
            predicted_classes = predicted_classes + 1  # Convert from 0-indexed to 1-indexed

            # Store data for visualization
            for i in range(batch_size):
                # Move tensors to CPU and convert to numpy
                img = images[i].cpu().numpy().transpose(1, 2, 0)
                true_mask = masks[i, 0].cpu().numpy()
                pred_mask = seg_outputs[i, 0].cpu().numpy()
                true_density = densities[i].item() * 100
                pred_density = reg_outputs[i, 0].item() * 100
                true_birads = birads[i].item() + 1  # Convert from 0-indexed to 1-indexed
                pred_birads = predicted_classes[i].item()

                # Denormalize image for display
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = img * std + mean
                img = np.clip(img, 0, 1)

                samples.append({
                    'image': img,
                    'true_mask': true_mask,
                    'pred_mask': pred_mask,
                    'true_density': true_density,
                    'pred_density': pred_density,
                    'true_birads': true_birads,
                    'pred_birads': pred_birads
                })

            if len(samples) >= num_samples:
                break

    # Create plots
    if len(samples) == 0:
        print("No samples to visualize")
        return None

    fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5 * len(samples)))

    # If only one sample, make axes 2D
    if len(samples) == 1:
        axes = axes.reshape(1, -1)

    # Create colormaps
    mask_cmap = plt.cm.get_cmap('jet')

    for i, sample in enumerate(samples):
        # Original image
        axes[i, 0].imshow(sample['image'])
        axes[i, 0].set_title(f"Original Image\nTrue Density: {sample['true_density']:.1f}%\nTrue BI-RADS: {sample['true_birads']}")
        axes[i, 0].axis('off')

        # True mask overlay
        axes[i, 1].imshow(sample['image'])
        mask_overlay = axes[i, 1].imshow(sample['true_mask'], alpha=0.5, cmap=mask_cmap)
        axes[i, 1].set_title("True Density Mask")
        axes[i, 1].axis('off')

        # Predicted mask overlay
        axes[i, 2].imshow(sample['image'])
        axes[i, 2].imshow(sample['pred_mask'], alpha=0.5, cmap=mask_cmap)
        axes[i, 2].set_title(f"Predicted Density Mask\nPred Density: {sample['pred_density']:.1f}%\nPred BI-RADS: {sample['pred_birads']}")
        axes[i, 2].axis('off')

    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(mask_overlay, cax=cbar_ax)

    # Adjust layout and save
    plt.tight_layout()

    output_path = plot_dir / f"segmentation_viz_{timestamp}.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(output_path)
