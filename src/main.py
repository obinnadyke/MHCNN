""" 
main.py - Training script for Breast Density Analysis
Can be trained on multiple GPUs using nn.DataParallel 
CMD: 
python main.py --data_path /path/to/data \
               --train_batch_size 4 \
               --valid_batch_size 4 \
               --accumulation_steps 4 \
               --optimizer AdamW \
               --lr 5e-5 \
               --lr_schedule onecycle \
               --use_deep_decoder \
               --confusion_matrix_epochs 20 \
               --weight_decay 1e-4
""" 

import os
import sys
import argparse
import warnings
import time
from tqdm import tqdm
from pathlib import Path
import numpy as np
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR 
import matplotlib.pyplot as plt
import seaborn as sns  # For visualization
import json

# Import from modules
from plots import create_tsne_plot
from device_checker import DeviceManager, verify_device_placement, move_to_device
from dataset_br import create_data_loaders
from density_model import setup_model
from param_stats import print_model_summary, calculate_model_statistics
from losses_metrics import (
    dice_coef,
    FocalRegLoss,
    MultiTaskLoss,
    compute_metrics_batch,
    FocalTverskyLoss,
    percentage_to_birads_category,
    # New imports from updated losses_metrics.py
    create_confusion_matrix_plot,
    create_learning_rate_plot,
    create_epoch_confusion_matrices,
    visualize_segmentations
)

# Suppress warnings
warnings.filterwarnings("ignore")

# Default hyperparameters - UPDATED defaults
DEFAULT_CONFIG = {
    'optimizer': 'AdamW',  # CHANGED: Default to AdamW for better training
    'loss_function': 'FocalTverskyLoss',  # Already using FocalTverskyLoss
    'lr': 5e-5,  # Slightly increased for AdamW
    'lr_schedule': 'onecycle',  # CHANGED: Default to OneCycleLR for faster convergence
    'segmentation_model': 'Unet',
    'encoder': 'resnet101',
    'pretrained_weights': 'imagenet',
    'activation_function': 'sigmoid',
    'lambda_reg': 0.3,  # Weight for regression loss
    'lambda_cls': 0.2,  # Weight for BI-RADS classification loss
    'early_stopping': 20,   # Early stopping patience (epochs)
    'output_size': 512,     # Output image size
    'use_deep_decoder': True,  # ADDED: Use deeper decoder option
}


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Breast Density Analysis Training')

    # Data paths
    parser.add_argument('--data_path', default='data', type=str,
                        help='Root directory with subfolders input_image/ and dense_mask/')

    # Training parameters
    parser.add_argument('--train_batch_size', default=2, type=int, help='Batch size for training.')
    parser.add_argument('--valid_batch_size', default=2, type=int, help='Batch size for validation.')
    parser.add_argument('--num_workers', default=0, type=int, help='Number of workers for DataLoader.')
    parser.add_argument('--num_epochs', default=250, type=int, help='Number of training epochs.')
    # Gradient accumulation steps
    parser.add_argument('--accumulation_steps', default=4, type=int,
                        help='Number of steps to accumulate gradients. Effectively increases batch size.')

    # Model architecture
    parser.add_argument('--segmentation_model', default=DEFAULT_CONFIG['segmentation_model'], type=str,
                        help='Model: Unet, FPN, etc.')
    parser.add_argument('--encoder', default=DEFAULT_CONFIG['encoder'], type=str,
                        help='Encoder name (e.g., resnet101, resnet18, vgg16, etc.)')
    parser.add_argument('--pretrained_weights', default=DEFAULT_CONFIG['pretrained_weights'], type=str,
                        help='Pretrained weights for encoder (e.g., "imagenet")')
    parser.add_argument('--activation_function', default=DEFAULT_CONFIG['activation_function'], type=str,
                        help='Activation for final layer (sigmoid/softmax/None)')
    # Option for deeper UNet decoder
    parser.add_argument('--use_deep_decoder', action='store_true',
                        help='Use deeper custom UNet decoder')

    # Use DEFAULT_CONFIG values
    parser.add_argument('--loss_function', default=DEFAULT_CONFIG['loss_function'], type=str,
                        help='Loss function: DiceLoss, TverskyLoss, FocalTverskyLoss, etc.')
    parser.add_argument('--optimizer', default=DEFAULT_CONFIG['optimizer'], type=str,
                        help='Optimizer (Adam, AdamW, SGD, etc.)')
    parser.add_argument('--lr', default=DEFAULT_CONFIG['lr'], type=float,
                        help='Initial learning rate.')
    parser.add_argument('--lambda_reg', default=DEFAULT_CONFIG['lambda_reg'], type=float,
                        help='Weight for regression loss component.')
    parser.add_argument('--lambda_cls', default=DEFAULT_CONFIG['lambda_cls'], type=float,
                        help='Weight for classification loss component.')
    parser.add_argument('--lr_schedule', default=DEFAULT_CONFIG['lr_schedule'], type=str,
                        help='Learning rate scheduler (onecycle, cosine, reducelr, steplr)')

    # Output settings
    parser.add_argument('--logs_file_path', default='output/logs/training.txt', type=str,
                        help='Path to save logs.')
    parser.add_argument('--model_save_path', default='output/models/density_model.pth', type=str,
                        help='Path to save model checkpoint.')

    # PLOTS: Confusion matrix generation frequency and t-SNE
    parser.add_argument('--confusion_matrix_epochs', default=50, type=int,
                        help='Generate confusion matrix every N epochs')
    parser.add_argument('--tsne_plot', action='store_true',
                        help='If set, generate a t-SNE plot from encoder embeddings after training.')

    # GPU settings
    parser.add_argument('--device_ids', default=None, type=str,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2").')

    # Weight decay parameter
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay parameter for optimizer')

    config = parser.parse_args()
    return config


def setup_directories(config):
    """Set up output directories for models and logs."""
    os.makedirs(os.path.dirname(config.logs_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(config.model_save_path), exist_ok=True)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    model_dir = Path(config.model_save_path).parent
    model_stem = Path(config.model_save_path).stem

    paths = {
        'timestamp': timestamp,
        'checkpoint': model_dir / f"{model_stem}_checkpoint_{timestamp}.pth",
        'final': model_dir / f"{model_stem}_final_{timestamp}.pth",
        'best': model_dir / f"{model_stem}_best_{timestamp}.pth",
        'summary': Path(os.path.dirname(config.logs_file_path)) / f"training_summary_{timestamp}.txt"
    }
    return paths


def extract_encoder_embeddings(model, data_loader, device, max_batches=None):
    """
    Forward pass on data_loader to extract features from the encoder.
    Returns arrays: (embeddings, birads_labels)
    """
    model.eval()

    # If wrapped in DataParallel, the encoder is under model.module.encoder
    #encoder = model.module.encoder if hasattr(model, 'module') else model.encoder

    if hasattr(model, 'module'):
        if hasattr(model.module, 'encoder'):
            encoder = model.module.encoder
        else:
            print("[extract_encoder_embeddings] --> (use_deep_decoder=False?). Skipping t-SNE.")
            return None, None
    else:
        # Single-GPU model
        if hasattr(model, 'encoder'):
            encoder = model.encoder
        else:
            print("[extract_encoder_embeddings] --> (use_deep_decoder=False?). Skipping t-SNE.")
            return None, None

    all_embeddings = []
    all_birads = []

    with torch.no_grad():
        loader_iter = tqdm(data_loader, desc="Extracting Embeddings", unit="batch")
        batch_count = 0
        for images, masks, density, birads in loader_iter:
            images = images.to(device)
            # Pass images through encoder only
            feats = encoder(images)  # shape depends on your backbone
            # Typically feats might be a 4D tensor (B, C, H, W). Doing average pooling:
            pooled = torch.nn.functional.adaptive_avg_pool2d(feats, (1,1)).squeeze(-1).squeeze(-1)
            # => shape (B, C)

            # Convert to numpy for t-SNE
            all_embeddings.append(pooled.cpu().numpy())
            # Add +1 because dataset uses 0..3 for BI-RADS
            all_birads.append(birads.cpu().numpy() + 1)
            # ^ recall your code stores birads_category = percentage_to_birads() - 1, so +1 to get real BI-RADS

            batch_count += 1
            if max_batches and batch_count >= max_batches:
                break

    # Concatenate all arrays
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_birads = np.concatenate(all_birads, axis=0)

    return all_embeddings, all_birads


def setup_training(config, model, device):
    """
    Set up loss functions, optimizer, and scheduler with proper device placement
    """
    import segmentation_models_multi_tasking as smp

    model = move_to_device(model, device)

    # Create segmentation loss - UPDATED: Always use FocalTverskyLoss
    seg_loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=0.75)
    seg_loss_fn.to(device)

    # Create focal regression loss - UPDATED: Higher gamma for better focus on hard examples
    reg_loss_fn = FocalRegLoss(gamma=2.0).to(device)

    # Multi-task loss
    multi_task_loss = MultiTaskLoss(
        seg_loss=seg_loss_fn,
        reg_loss=reg_loss_fn,
        lambda_reg=config.lambda_reg,
        lambda_cls=config.lambda_cls,
        device=device
    ).to(device)

    # Verify each submodule is on correct device
    for name, module in multi_task_loss.named_children():
        if hasattr(module, 'parameters'):
            for param in module.parameters():
                if param.device != device:
                    print(f"Warning: param in {name} on {param.device}, moving to {device}")
                    param.data = param.data.to(device)

    # Optimizer configuration
    if config.optimizer == 'AdamW':
        # AdamW with weight decay
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),  # Default AdamW betas
            eps=1e-8
        )
        print(f"Using AdamW optimizer with weight_decay={config.weight_decay}")
    elif config.optimizer == 'Adam':
        # Standard Adam
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        print(f"Using Adam optimizer with weight_decay={config.weight_decay}")
    elif config.optimizer == 'SGD':
        # SGD with momentum and nesterov
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.lr,
            momentum=0.9,
            weight_decay=config.weight_decay,
            nesterov=True
        )
        print(f"Using SGD optimizer with momentum=0.9, weight_decay={config.weight_decay}")
    else:
        # Fallback to any other optimizer specified by name
        optimizer = getattr(torch.optim, config.optimizer)(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        print(f"Using {config.optimizer} optimizer with weight_decay={config.weight_decay}")

    # Learning rate schedulers
    if config.lr_schedule == 'steplr':
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        print("Using StepLR scheduler: step_size=20, gamma=0.5")

    elif config.lr_schedule == 'reducelr':
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=5, verbose=True, min_lr=1e-7
        )
        print("Using ReduceLROnPlateau scheduler: factor=0.2, patience=5")

    elif config.lr_schedule == 'cosine':
        # Cosine annealing with warm-up
        T_max = config.num_epochs  # Total number of epochs
        warmup_epochs = int(T_max * 0.1)  # 10% of total epochs for warm-up

        # Create a cosine scheduler
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=T_max - warmup_epochs,  # Remaining epochs after warm-up
            eta_min=config.lr * 0.01  # Minimum learning rate at the end
        )

        # Create a wrapper class for warm-up + cosine annealing
        class WarmupCosineScheduler:
            def __init__(self, optimizer, warmup_epochs, cosine_scheduler, init_lr, max_lr):
                self.optimizer = optimizer
                self.warmup_epochs = warmup_epochs
                self.cosine_scheduler = cosine_scheduler
                self.init_lr = init_lr
                self.max_lr = max_lr
                self.current_epoch = 0

            def step(self, epoch=None, metrics=None):
                if epoch is not None:
                    self.current_epoch = epoch

                if self.current_epoch < self.warmup_epochs:
                    # Linear warm-up
                    lr_scale = self.current_epoch / self.warmup_epochs
                    lr = self.init_lr + (self.max_lr - self.init_lr) * lr_scale
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = lr
                else:
                    # Cosine annealing
                    self.cosine_scheduler.step(self.current_epoch - self.warmup_epochs)

                self.current_epoch += 1

            def get_last_lr(self):
                if self.current_epoch <= self.warmup_epochs:
                    # During warm-up
                    lr_scale = self.current_epoch / self.warmup_epochs
                    return [self.init_lr + (self.max_lr - self.init_lr) * lr_scale]
                else:
                    # During cosine annealing
                    return self.cosine_scheduler.get_last_lr()

        # Create the combined scheduler
        lr_scheduler = WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_epochs=warmup_epochs,
            cosine_scheduler=cosine_scheduler,
            init_lr=config.lr * 0.1,  # Start at 10% of max lr
            max_lr=config.lr  # Target lr after warm-up
        )
        print(f"Using Cosine Annealing with {warmup_epochs} warm-up epochs")

    # OneCycleLR 
    elif config.lr_schedule == 'onecycle':
        lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=config.lr,
            total_steps=config.num_epochs,
            pct_start=0.3,  # Peak lr at 30% of training
            div_factor=25,  # initial_lr = max_lr/div_factor
            final_div_factor=10000,  # min_lr = initial_lr/final_div_factor
            anneal_strategy='cos',
            three_phase=False
        )
        print(f"Using OneCycleLR with max_lr={config.lr}, pct_start=0.3")

    else:
        # Default fallback
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.num_epochs)
        print(f"Using basic CosineAnnealingLR scheduler with T_max={config.num_epochs}")

    return multi_task_loss, optimizer, lr_scheduler


def collect_density_percentages(dataloader, device, max_batches=None):
    """
    Collect breast density percentages from the dataset for distribution analysis.
    Args:
        dataloader: DataLoader with dataset
        device: Device to move tensors to
        max_batches: Maximum number of batches to process (None=all)
    Returns:
        list: List of density percentages
    """
    densities = []

    progress_bar = tqdm(total=len(dataloader) if max_batches is None else min(max_batches, len(dataloader)),
                       desc="Collecting density data", unit="batch")

    batch_count = 0
    for _, _, density_targets, _ in dataloader:
        density_targets = density_targets.to(device)
        densities.extend((density_targets * 100).cpu().numpy().flatten().tolist())
        progress_bar.update(1)

        batch_count += 1
        if max_batches is not None and batch_count >= max_batches:
            break

    progress_bar.close()
    return densities


def train_epoch(model, train_dataloader, optimizer, multi_task_loss, device, max_grad_norm=1.0, accumulation_steps=4):
    """
    Run a single training epoch and return metrics.
    Gradient accumulation implementation, proper error handling, more debugging
    """
    model.train()
    train_seg_loss = train_reg_loss = train_cls_loss = train_total_loss = 0.0
    train_dice_sum = train_reg_mae_sum = 0.0
    train_cls_correct = 0
    train_samples = 0

    # First batch debugging flag
    first_batch = True

    # Set up progress bar for better tracking
    progress_bar = tqdm(train_dataloader, desc="Training", unit="batch")

    optimizer.zero_grad()  # Zero gradients at the start

    for batch_idx, (images, masks, density_targets, birads_targets) in enumerate(progress_bar):
        try:
            # Move tensors to device
            images = images.to(device)
            masks = masks.to(device)
            density_targets = density_targets.to(device)
            birads_targets = birads_targets.to(device)

            # Forward pass
            outputs = model(images)

            if isinstance(outputs, tuple) and len(outputs) == 3:
                seg_outputs, reg_outputs, cls_outputs = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                seg_outputs, reg_outputs = outputs
                cls_outputs = None  # Handle cases where classification output is missing
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")

            # Handle tuple outputs from segmentation model
            if isinstance(seg_outputs, tuple):
                seg_outputs = seg_outputs[0]  # Use the main segmentation output

            # Debug info for first batch only
            if first_batch:
                print("\nDiagnostic Information (First Batch):")
                print(f"- Images: {images.shape}, range [{images.min().item():.4f}, {images.max().item():.4f}]")
                print(f"- Masks: {masks.shape}, values: min={masks.min().item()}, max={masks.max().item()}")

                # Add these additional debug lines:
                print(f"- Density targets: {density_targets.cpu().numpy().flatten()}")
                print(f"- BIRADS targets: {birads_targets.cpu().numpy()}")

                # Calculate density histogram
                density_hist, _ = np.histogram(
                    density_targets.cpu().numpy(),
                    bins=[0, 0.25, 0.5, 0.75, 1.0]
                )
                print(f"- Density histogram: {density_hist} (bins=[0, 25%, 50%, 75%, 100%])")

                # Display mask statistics
                mask_pixels = masks.shape[2] * masks.shape[3]
                for i in range(min(3, len(masks))):
                    nonzero = torch.count_nonzero(masks[i]).item()
                    nonzero_percent = (nonzero / mask_pixels) * 100
                    print(f"  Mask {i}: {nonzero_percent:.2f}% non-zero pixels")

                # Tuple handling for seg_outputs
                if isinstance(seg_outputs, tuple):
                    print(f"- Seg outputs is a tuple with {len(seg_outputs)} elements")
                    seg_main = seg_outputs[0]  # First element is usually the main prediction
                    print(f"- Seg outputs[0]: {seg_main.shape}, range [{seg_main.min().item():.4f}, {seg_main.max().item():.4f}]")
                else:
                    print(f"- Seg outputs: {seg_outputs.shape}, range [{seg_outputs.min().item():.4f}, {seg_outputs.max().item():.4f}]")

                print(f"- Density targets: {density_targets[:5].cpu().numpy().flatten()}")
                print(f"- Regression outputs: {reg_outputs[:5].cpu().detach().numpy().flatten()}")
                print(f"- BIRADS targets: {birads_targets[:5].cpu().numpy()}")
                first_batch = False

            # Calculate loss
            total_loss, loss_seg, loss_reg, loss_cls = multi_task_loss(
                seg_outputs, masks, reg_outputs, density_targets, cls_outputs, birads_targets
            )

            # Normalize loss for gradient accumulation
            total_loss = total_loss / accumulation_steps

            # Backward pass
            total_loss.backward()

            # Update metrics (use the original loss values, not the normalized)
            batch_size = images.size(0)
            train_seg_loss += loss_seg.item() * batch_size
            train_reg_loss += loss_reg.item() * batch_size
            train_cls_loss += loss_cls.item() * batch_size
            train_total_loss += total_loss.item() * accumulation_steps * batch_size  # Adjust for accumulation

            # Calculate dice coefficient
            dice_val = dice_coef(seg_outputs, masks)
            train_dice_sum += dice_val * batch_size

            # Regression MAE
            train_reg_mae_sum += torch.abs(reg_outputs * 100 - density_targets * 100).sum().item()

            # Classification accuracy
            _, predicted = torch.max(cls_outputs, 1)
            train_cls_correct += (predicted == birads_targets).sum().item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{total_loss.item() * accumulation_steps:.4f}",
                'dice': f"{dice_val:.4f}"
            })

            # Track per-class predictions (for debugging class imbalance)
            if batch_idx == 0:  # First batch only
                pred_classes = torch.argmax(cls_outputs, dim=1).cpu().numpy()
                class_counts = np.bincount(pred_classes, minlength=4)
                print(f"  Class distribution in predictions: {class_counts} (BI-RADS 1-4)")

            train_samples += batch_size

            # Step optimizer after accumulation
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_dataloader):
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Take optimizer step
                optimizer.step()
                optimizer.zero_grad()

                # Log loss breakdown every X batches
                if (batch_idx + 1) % (accumulation_steps * 5) == 0:
                    print(f"[Batch {batch_idx+1}/{len(train_dataloader)}] Loss Breakdown: "
                          f"Seg={loss_seg.item():.4f}, Reg={loss_reg.item():.4f}, "
                          f"Cls={loss_cls.item():.4f}, Total={loss_seg.item() + loss_reg.item() + loss_cls.item():.4f}")

        except Exception as e:
            print(f"Error in batch {batch_idx}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Continue training with the next batch
            continue

    # Close progress bar
    progress_bar.close()

    # Calculate epoch metrics
    if train_samples > 0:
        metrics = {
            'seg_loss': train_seg_loss / train_samples,
            'reg_loss': train_reg_loss / train_samples,
            'cls_loss': train_cls_loss / train_samples,
            'total_loss': train_total_loss / train_samples,
            'dice_score': train_dice_sum / train_samples,
            'reg_mae': train_reg_mae_sum / train_samples,
            'birads_accuracy': (train_cls_correct / train_samples) * 100
        }
    else:
        print("WARNING: No valid samples processed in epoch!")
        metrics = {
            'seg_loss': 0.0, 'reg_loss': 0.0, 'cls_loss': 0.0, 'total_loss': 0.0,
            'dice_score': 0.0, 'reg_mae': 0.0, 'birads_accuracy': 0.0
        }

    return metrics


def validate_epoch(model, valid_dataloader, multi_task_loss, device):
    """
    Run validation and return metrics with comprehensive error handling

    Validation metrics collection and progress tracking
    """
    model.eval()
    val_seg_loss = val_reg_loss = val_cls_loss = val_total_loss = 0.0
    val_dice_sum = val_reg_mae_sum = 0.0
    val_cls_correct = 0
    val_samples = 0

    all_preds, all_targets = [], []

    # Set up progress bar for better tracking
    progress_bar = tqdm(valid_dataloader, desc="Validation", unit="batch")

    with torch.no_grad():
        for images, masks, density_targets, birads_targets in progress_bar:
            try:
                # Move data to device
                images = images.to(device)
                masks = masks.to(device)
                density_targets = density_targets.to(device)
                birads_targets = birads_targets.to(device)

                # Forward pass through model
                outputs = model(images)

                if isinstance(outputs, tuple) and len(outputs) == 3:
                    seg_outputs, reg_outputs, cls_outputs = outputs
                elif isinstance(outputs, tuple) and len(outputs) == 2:
                    seg_outputs, reg_outputs = outputs
                    cls_outputs = None  # Handle cases where classification output is missing
                    raise ValueError("Missing classification outputs from model")
                else:
                    raise ValueError(f"Unexpected model output format: {type(outputs)}")

                # Handle tuple segmentation outputs
                if isinstance(seg_outputs, tuple):
                    seg_outputs = seg_outputs[0]  # Use the main prediction

                # Calculate loss
                total_loss, loss_seg, loss_reg, loss_cls = multi_task_loss(
                    seg_outputs, masks, reg_outputs, density_targets, cls_outputs, birads_targets
                )

                batch_size = images.size(0)
                val_seg_loss += loss_seg.item() * batch_size
                val_reg_loss += loss_reg.item() * batch_size
                val_cls_loss += loss_cls.item() * batch_size
                val_total_loss += total_loss.item() * batch_size

                # Calculate dice coefficient
                dice_val = dice_coef(seg_outputs, masks)
                val_dice_sum += dice_val * batch_size

                # Calculate regression MAE
                val_reg_mae_sum += torch.abs(reg_outputs * 100 - density_targets * 100).sum().item()

                # Calculate classification accuracy
                _, predicted = torch.max(cls_outputs, 1)
                val_cls_correct += (predicted == birads_targets).sum().item()

                # For confusion matrix - convert to 1-indexed BIRADS (model uses 0-indexed)
                all_preds.extend((predicted + 1).cpu().numpy())
                all_targets.extend((birads_targets + 1).cpu().numpy())

                val_samples += batch_size

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{total_loss.item():.4f}",
                    'dice': f"{dice_val:.4f}"
                })

            except Exception as e:
                print(f"Error in validation: {e}")
                import traceback
                traceback.print_exc()
                continue  # Skip this batch

        # Close progress bar
        progress_bar.close()

    if val_samples > 0:
        metrics = {
            'seg_loss': val_seg_loss / val_samples,
            'reg_loss': val_reg_loss / val_samples,
            'cls_loss': val_cls_loss / val_samples,
            'total_loss': val_total_loss / val_samples,
            'dice_score': val_dice_sum / val_samples,
            'reg_mae': val_reg_mae_sum / val_samples,
            'birads_accuracy': (val_cls_correct / val_samples) * 100,
            'predictions': all_preds,
            'targets': all_targets
        }

        # Add confusion matrix
        try:
            from sklearn.metrics import confusion_matrix
            metrics['confusion_matrix'] = confusion_matrix(
                all_targets, all_preds, labels=[1, 2, 3, 4]
            )
        except Exception as e:
            print(f"Error creating confusion matrix: {e}")
            metrics['confusion_matrix'] = None
    else:
        print("WARNING: No valid samples processed in validation!")
        metrics = {
            'seg_loss': 0.0, 'reg_loss': 0.0, 'cls_loss': 0.0, 'total_loss': 0.0,
            'dice_score': 0.0, 'reg_mae': 0.0, 'birads_accuracy': 0.0,
            'predictions': [], 'targets': [], 'confusion_matrix': None
        }

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, train_history, config, path):
    """Save model checkpoint."""
    if not isinstance(config, dict):
        config = vars(config)

    save_dict = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history
    }

    # Add metrics (if provided)
    if metrics is not None:
        # Remove large numpy arrays and other non-serializable data
        save_metrics = {}
        for k, v in metrics.items():
            if k not in ['predictions', 'targets', 'confusion_matrix']:
                save_metrics[k] = v
        save_dict['metrics'] = save_metrics

    # Add config settings
    config_copy = {}
    for k, v in config.items():
        # Skip device IDs and other non-serializable data
        if k not in ['device_ids', 'device']:
            config_copy[k] = v
    save_dict['config'] = config_copy

    # Save the checkpoint
    torch.save(save_dict, path)
    print(f"Checkpoint saved to {path}")


def create_training_plots(train_loss, valid_loss, train_dice, valid_dice,
                      train_reg_mae, valid_reg_mae, output_path, timestamp):
    """
    Create training history plots
    Args:
        train_loss (list): Training loss history
        valid_loss (list): Validation loss history
        train_dice (list): Training Dice score history
        valid_dice (list): Validation Dice score history
        train_reg_mae (list): Training regression MAE history
        valid_reg_mae (list): Validation regression MAE history
        output_path (str): Base path for logs
        timestamp (str): Timestamp for unique filenames
    Returns:
        str: Path to saved plot
    """
    import matplotlib.pyplot as plt
    from pathlib import Path

    plot_dir = Path(output_path).parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(15, 10))

    # Plot 1: Losses
    plt.subplot(2, 2, 1)
    plt.plot(train_loss, label='Train Loss')
    plt.plot(valid_loss, label='Valid Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Dice Score
    plt.subplot(2, 2, 2)
    plt.plot(train_dice, label='Train Dice')
    plt.plot(valid_dice, label='Valid Dice')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.title('Segmentation Performance (Dice)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Regression MAE
    plt.subplot(2, 2, 3)
    plt.plot(train_reg_mae, label='Train MAE')
    plt.plot(valid_reg_mae, label='Valid MAE')
    plt.xlabel('Epochs')
    plt.ylabel('MAE (%)')
    plt.title('Regression Performance (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plots_path = plot_dir / f"training_history_{timestamp}.png"
    plt.savefig(plots_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plots_path)


def create_distribution_plot(train_densities, valid_densities, logs_file_path, timestamp):
    """
    Create density distribution plot
    Args:
        train_densities (list): List of density percentages in training set
        valid_densities (list): List of density percentages in validation set
        logs_file_path (str): Path to logs file
        timestamp (str): Timestamp for unique filenames
    Returns:
        str: Path to saved plot
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    plot_dir = Path(logs_file_path).parent / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 6))

    plt.hist(train_densities, bins=20, alpha=0.7, label='Training')
    plt.hist(valid_densities, bins=20, alpha=0.7, label='Validation')

    plt.xlabel('Breast Density Percentage')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Breast Density Percentages')
    plt.legend()
    plt.grid(alpha=0.3)

    plot_path = plot_dir / f"density_distribution_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Also create a distribution plot by BI-RADS category
    plt.figure(figsize=(10, 6))

    # Convert to BI-RADS categories
    from losses_metrics import percentage_to_birads_category

    train_birads = [percentage_to_birads_category(p) for p in train_densities]
    valid_birads = [percentage_to_birads_category(p) for p in valid_densities]

    # Count occurrences
    train_counts = np.bincount(train_birads, minlength=5)[1:]  # Skip index 0
    valid_counts = np.bincount(valid_birads, minlength=5)[1:]  # Skip index 0

    # Convert to percentages
    train_pct = train_counts / len(train_birads) * 100
    valid_pct = valid_counts / len(valid_birads) * 100

    width = 0.35
    x = np.arange(1, 5)

    plt.bar(x - width/2, train_pct, width, label='Training')
    plt.bar(x + width/2, valid_pct, width, label='Validation')

    plt.xlabel('BI-RADS Category')
    plt.ylabel('Percentage')
    plt.title('Distribution of BI-RADS Categories')
    plt.xticks(x)
    plt.legend()
    plt.grid(alpha=0.3)

    birads_plot_path = plot_dir / f"birads_distribution_{timestamp}.png"
    plt.savefig(birads_plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return str(plot_path)


def check_data_quality(train_dataloader, device, threshold=0.95, max_batches=5):
    """
    Check data quality before training
    Args:
        train_dataloader: DataLoader with training data
        device: Device to move tensors to
        threshold: Threshold for average density warning
        max_batches: Maximum number of batches to check
    Returns:
        bool: True if data quality is acceptable, False otherwise
    """
    import numpy as np

    print("Running data quality checks...")

    density_values = []
    mask_nonzero_percentages = []

    # Check a subset of batches
    batch_count = 0

    with torch.no_grad():
        for images, masks, density_targets, birads_targets in train_dataloader:
            # Check density values
            density_values.extend(density_targets.cpu().numpy().flatten().tolist())

            # Check mask properties
            mask_pixels = masks.shape[2] * masks.shape[3]
            for mask in masks:
                nonzero = torch.count_nonzero(mask).item()
                nonzero_percent = (nonzero / mask_pixels) * 100
                mask_nonzero_percentages.append(nonzero_percent)

            batch_count += 1
            if batch_count >= max_batches:
                break

    # Print statistics
    print("\n=== Data Quality Report ===")
    print(f"Density stats: Min={min(density_values):.4f}, Max={max(density_values):.4f}, Mean={np.mean(density_values):.4f}")
    print(f"Mask non-zero stats: Min={min(mask_nonzero_percentages):.2f}%, Max={max(mask_nonzero_percentages):.2f}%, Mean={np.mean(mask_nonzero_percentages):.2f}%")

    # Check density distribution
    hist, bins = np.histogram(density_values, bins=[0, 0.25, 0.5, 0.75, 1.0, 1.1])
    print(f"Density distribution: {hist} (bins= 0-25%, 25-50%, 50-75%, 75-100%, >100%)")

    # Issue warnings
    if np.mean(density_values) > threshold:
        print("\nWARNING: Average density is very high. This likely indicates a data processing issue!")
        print("   - Check how mask values are being interpreted in dataset_br.py")
        print("   - Ensure dense tissue is properly distinguished from normal breast tissue")
        print("   - Typical mammographic breast density should average 40-50%")
        user_input = input("\nContinue with training anyway? (y/n): ")
        return user_input.lower() == 'y'

    if hist[-1] > 0:
        print("\nWARNING: Some density values exceed 100%, which is invalid!")
        user_input = input("\nContinue with training anyway? (y/n): ")
        return user_input.lower() == 'y'

    # Check for empty masks
    if min(mask_nonzero_percentages) < 0.1:
        print("\nWARNING: Some masks appear to be empty or nearly empty (< 0.1% non-zero pixels)!")
        print("   - Check mask loading and preprocessing in dataset_br.py")
        user_input = input("\nContinue with training anyway? (y/n): ")
        return user_input.lower() == 'y'

    print("\nData quality checks completed - no major issues found!")
    return True


def write_summary(model_stats, training_time, best_score, final_metrics, paths, config):
    """
    Write training summary to file.

    Args:
        model_stats (dict): Model statistics
        training_time (float): Total training time in seconds
        best_score (float): Best combined score
        final_metrics (dict): Final metrics from validation
        paths (dict): Dictionary of paths
        config: Configuration object
    """
    if not isinstance(config, dict):
        config = vars(config)

    with open(paths['summary'], 'w') as f:
        f.write("=== Model Architecture Statistics ===\n")
        f.write(f"Model: {config['segmentation_model']}\n")
        f.write(f"Encoder: {config['encoder']}\n")
        f.write(f"Input shape: {model_stats.get('input_shape', '(?, ?, ?)')}\n")
        f.write(f"Trainable parameters: {model_stats['trainable_parameters']}\n")
        f.write(f"Total parameters: {model_stats['total_parameters']}\n")
        if 'gflops_ptflops' in model_stats:
            f.write(f"GFLOPs: {model_stats['gflops_ptflops']:.2f}\n")
        elif 'gflops_estimated' in model_stats:
            f.write(f"GFLOPs (estimated): {model_stats['gflops_estimated']:.2f}\n")

        f.write("\n=== Training Summary ===\n")
        f.write(f"Training completed at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total training time: {training_time/3600:.2f} hours\n")
        f.write(f"Best Combined Score: {best_score:.4f}\n")
        f.write(f"Final Dice Score: {final_metrics['dice_score']:.4f}\n")
        f.write(f"Final Regression MAE: {final_metrics['reg_mae']:.2f}%\n")
        f.write(f"Final BI-RADS Accuracy: {final_metrics['birads_accuracy']:.2f}%\n")

        # Add confusion matrix (if available)
        if 'confusion_matrix' in final_metrics and final_metrics['confusion_matrix'] is not None:
            f.write("\n=== Final BI-RADS Confusion Matrix ===\n")
            cm = final_metrics['confusion_matrix']
            f.write(f"    B1   B2   B3   B4\n")
            for i, row in enumerate(cm):
                f.write(f"B{i+1} {' '.join(f'{cell:4d}' for cell in row)}\n")

        f.write("\n=== Training Configuration ===\n")
        for key, value in config.items():
            # Skip device IDs and lengthy parameters
            if key not in ['device_ids', 'model_state_dict', 'optimizer_state_dict']:
                f.write(f"{key}: {value}\n")

        f.write("\n=== Output Files ===\n")
        f.write(f"Best model: {paths['best']}\n")
        f.write(f"Final model: {paths['final']}\n")

        # Add plot paths if available
        if 'plots' in paths and paths['plots']:
            f.write(f"Training plots: {paths['plots']}\n")
        if 'distribution_plot' in paths and paths['distribution_plot']:
            f.write(f"Density distribution plot: {paths['distribution_plot']}\n")
        if 'birads_plot' in paths and paths['birads_plot']:
            f.write(f"BI-RADS distribution plot: {paths['birads_plot']}\n")
        if 'lr_plot' in paths and paths['lr_plot']:
            f.write(f"Learning rate plot: {paths['lr_plot']}\n")
        if 'seg_viz' in paths and paths['seg_viz']:
            f.write(f"Segmentation visualization: {paths['seg_viz']}\n")
        if 'best_cm' in paths and paths['best_cm']:
            f.write(f"Best epoch confusion matrix: {paths['best_cm']}\n")
        if 'final_cm' in paths and paths['final_cm']:
            f.write(f"Final epoch confusion matrix: {paths['final_cm']}\n")

    print(f"Training summary saved to: {paths['summary']}")


def main():
    """
    Main training function with improved error handling and debugging
    """
    try:
        # Parse command-line arguments
        config = parse_args()
        print("\n=== Training Configuration ===")
        for key, value in vars(config).items():
            print(f"{key}: {value}")

        # Set up device for training
        device_manager = DeviceManager(config.device_ids)
        device_manager.print_device_info()
        device = device_manager.device
        print(f"Using device for training: {device}")

        # Set up directories for output
        paths = setup_directories(config)

        # Create data loaders
        train_dataloader, valid_dataloader, num_train, num_valid = create_data_loaders(
            vars(config), device_manager
        )
        print(f"Training samples: {num_train}, Validation samples: {num_valid}")

        # Check data quality before training
        if not check_data_quality(train_dataloader, device):
            print("Stopping training due to data quality concerns.")
            return

        print("\n=== Setting up model ===")
        # Add 'use_deep_decoder' to config
        vars(config)['use_deep_decoder'] = config.use_deep_decoder
        model = setup_model(vars(config), device_manager)
        print(f"Model initialized on device: {next(model.parameters()).device}")

        print("\n=== Verifying model device placement ===")
        try:
            verify_device_placement(model, device)
            print("All model parameters and buffers are on the correct device")
        except RuntimeError as e:
            print(f"Device verification failed: {str(e)}")
            print("Attempting to fix device placement...")
            model = move_to_device(model, device)
            print("Model explicitly moved to device again")

        model_stats = calculate_model_statistics(model, device=device)
        print("\n=== Model Statistics ===")
        print(f"Trainable parameters: {model_stats['trainable_parameters']:,}")
        print(f"Total parameters: {model_stats['total_parameters']:,}")
        if 'gflops_ptflops' in model_stats:
            print(f"GFLOPs: {model_stats['gflops_ptflops']:.2f}")
        elif 'gflops_estimated' in model_stats:
            print(f"GFLOPs (estimated): {model_stats['gflops_estimated']:.2f}")

        print("\n=== Collecting density distribution data ===")
        train_densities = collect_density_percentages(train_dataloader, device)
        valid_densities = collect_density_percentages(valid_dataloader, device)

        # Generate density distribution plot
        try:
            distribution_plot_path = create_distribution_plot(
                train_densities, valid_densities,
                config.logs_file_path, paths['timestamp']
            )
            print(f"Density distribution plot saved to: {distribution_plot_path}")
        except Exception as e:
            print(f"Warning: Could not create distribution plot: {e}")
            distribution_plot_path = None

        print("\n=== Setting up training components ===")
        multi_task_loss, optimizer, lr_scheduler = setup_training(config, model, device)

        if hasattr(multi_task_loss, 'device'):
            print(f"MultiTaskLoss device: {multi_task_loss.device}")

        # Max_grad_norm due to using gradient accumulation
        max_grad_norm = 5.0
        print(f"Setting max_grad_norm={max_grad_norm} for gradient clipping")

        # Initialize training history
        train_history = {
            'train_loss': [], 'valid_loss': [],
            'train_dice': [], 'valid_dice': [],
            'train_reg_mae': [], 'valid_reg_mae': [],
            'train_birads_acc': [], 'valid_birads_acc': [],
        }

        lr_history = []  # For tracking learning rate changes
        val_predictions_history = []  # For storing validation predictions
        val_targets_history = []  # For storing validation targets
        best_epoch_idx = 0  # Keep track of best epoch index

        best_score = 0.0
        early_stopping_patience = getattr(config, 'early_stopping', 20)
        early_stopping_counter = 0
        prev_valid_loss = float('inf')

        # Save initial checkpoint
        save_checkpoint(
            model, optimizer, 0, None, train_history,
            config, paths['checkpoint']
        )

        print("\n=== Performing final device verification ===")
        for name, param in model.named_parameters():
            if param.device != device:
                print(f"Warning: Model parameter {name} on {param.device}, should be on {device}")
                param.data = param.data.to(device)

        if hasattr(multi_task_loss, 'parameters'):
            for name, param in multi_task_loss.named_parameters():
                if param.device != device:
                    print(f"Warning: Loss parameter {name} on {param.device}, should be on {device}")
                    param.data = param.data.to(device)

        print("\n=== Starting Training Session ===")
        start_time = time.time()

        with open(config.logs_file_path, 'a+') as logs_file:
            print("=== Training Configuration ===", file=logs_file)
            for key, value in vars(config).items():
                print(f"{key}: {value}", file=logs_file)

            print("\n=== Training Progress ===", file=logs_file)
            print("Epoch\tTrain_Loss\tTrain_Dice\tTrain_RegMAE\tTrain_BIRADS\tVal_Loss\tVal_Dice\tVal_RegMAE\tVal_BIRADS\tComb_Score",
                  file=logs_file)

            for epoch in range(config.num_epochs):
                print(f"\nEpoch: {epoch+1}/{config.num_epochs}")

                # FIXED: Better phased training strategy that more consistently reaches good solutions
                if epoch < 5:
                    # First 5 epochs: Just train segmentation to establish a good foundation
                    print(f"segmentation-only phase {epoch+1}/5")
                    multi_task_loss.lambda_reg = 0.0
                    multi_task_loss.lambda_cls = 0.0
                elif epoch < 15:
                    # Next 10 epochs: Gradually add regression while maintaining segmentation focus
                    progress = (epoch - 5) / 10
                    print(f"regression ramp-up phase {epoch+1-5}/10")
                    multi_task_loss.lambda_reg = config.lambda_reg * progress  # Gradual increase
                    multi_task_loss.lambda_cls = 0.0  # Still no classification
                elif epoch < 25:
                    # Next 10 epochs: Add classification task
                    progress = (epoch - 15) / 10
                    print(f"classification ramp-up phase {epoch+1-15}/10")
                    multi_task_loss.lambda_reg = config.lambda_reg  # Full regression weight
                    multi_task_loss.lambda_cls = config.lambda_cls * progress  # Gradually increase classification
                else:
                    # Full training phase with balanced weights
                    print(f"full training phase {epoch+1-25}/{config.num_epochs-25}")
                    multi_task_loss.lambda_reg = config.lambda_reg  # Full weights
                    multi_task_loss.lambda_cls = config.lambda_cls

                # Print current loss weights for monitoring
                print(f"Current loss weights: lambda(λ)_reg={multi_task_loss.lambda_reg:.4f}, lambda(λ)_cls={multi_task_loss.lambda_cls:.4f}")

                # Run training epoch with gradient accumulation
                train_metrics = train_epoch(
                    model, train_dataloader, optimizer, multi_task_loss, device,
                    max_grad_norm, config.accumulation_steps
                )

                # Run validation epoch
                val_metrics = validate_epoch(
                    model, valid_dataloader, multi_task_loss, device
                )

                # Track predictions and targets for confusion matrices
                if 'predictions' in val_metrics and 'targets' in val_metrics:
                    val_predictions_history.append(val_metrics['predictions'])
                    val_targets_history.append(val_metrics['targets'])

                # Step learning rate scheduler based on validation loss or epoch
                if hasattr(lr_scheduler, 'get_last_lr'):
                    current_lr = lr_scheduler.get_last_lr()[0]
                else:
                    current_lr = optimizer.param_groups[0]['lr']

                print(f"Current learning rate: {current_lr:.8f}")
                lr_history.append(current_lr)

                # Update scheduler
                if config.lr_schedule == 'reducelr':
                    lr_scheduler.step(val_metrics['total_loss'])
                else:
                    lr_scheduler.step()

                # Update training history
                train_history['train_loss'].append(train_metrics['total_loss'])
                train_history['valid_loss'].append(val_metrics['total_loss'])
                train_history['train_dice'].append(train_metrics['dice_score'])
                train_history['valid_dice'].append(val_metrics['dice_score'])
                train_history['train_reg_mae'].append(train_metrics['reg_mae'])
                train_history['valid_reg_mae'].append(val_metrics['reg_mae'])
                train_history['train_birads_acc'].append(train_metrics['birads_accuracy'])
                train_history['valid_birads_acc'].append(val_metrics['birads_accuracy'])

                # Print epoch results
                print(f"Epoch [{epoch+1}/{config.num_epochs}]:")
                print(f"  Train: Loss={train_metrics['total_loss']:.4f} (Seg={train_metrics['seg_loss']:.4f}, "
                      f"Reg={train_metrics['reg_loss']:.4f}, Cls={train_metrics['cls_loss']:.4f}), "
                      f"Dice={train_metrics['dice_score']:.4f}, RegMAE={train_metrics['reg_mae']:.2f}%, "
                      f"BIRADS Acc={train_metrics['birads_accuracy']:.2f}%")
                print(f"  Valid: Loss={val_metrics['total_loss']:.4f} (Seg={val_metrics['seg_loss']:.4f}, "
                      f"Reg={val_metrics['reg_loss']:.4f}, Cls={val_metrics['cls_loss']:.4f}), "
                      f"Dice={val_metrics['dice_score']:.4f}, RegMAE={val_metrics['reg_mae']:.2f}%, "
                      f"BIRADS Acc={val_metrics['birads_accuracy']:.2f}%")

                # Weighted combined score for model selection (focus on segmentation quality)
                combined_score = (0.5 * val_metrics['dice_score'] +
                                  0.25 * (100 - min(val_metrics['reg_mae'], 100)) / 100 +
                                  0.25 * (val_metrics['birads_accuracy'] / 100))

                # Logging (include combined score)
                print(f"{epoch+1}\t{train_metrics['total_loss']:.4f}\t{train_metrics['dice_score']:.4f}\t"
                      f"{train_metrics['reg_mae']:.2f}\t{train_metrics['birads_accuracy']:.2f}\t"
                      f"{val_metrics['total_loss']:.4f}\t{val_metrics['dice_score']:.4f}\t"
                      f"{val_metrics['reg_mae']:.2f}\t{val_metrics['birads_accuracy']:.2f}\t"
                      f"{combined_score:.4f}",
                      file=logs_file)

                # Early stopping check
                if val_metrics['total_loss'] >= prev_valid_loss:
                    early_stopping_counter += 1
                    print(f"  Early stopping counter: {early_stopping_counter}/{early_stopping_patience}")
                else:
                    early_stopping_counter = 0

                prev_valid_loss = val_metrics['total_loss']

                # Trigger early stopping if no improvement for many epochs
                if early_stopping_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break

                # Save periodic checkpoint (every 10 epochs)
                if (epoch + 1) % 10 == 0:
                    save_checkpoint(
                        model, optimizer, epoch, val_metrics, train_history,
                        config, paths['checkpoint']
                    )

                # Generate confusion matrix periodically?
                if (epoch + 1) % config.confusion_matrix_epochs == 0 or epoch == 0:
                    # Generate confusion matrix if available
                    if 'confusion_matrix' in val_metrics and val_metrics['confusion_matrix'] is not None:
                        try:
                            cm_path = Path(config.logs_file_path).parent / 'plots' / f"confusion_matrix_epoch{epoch+1}_{paths['timestamp']}.png"
                            create_confusion_matrix_plot(
                                val_metrics['confusion_matrix'],
                                str(cm_path),
                                f"BI-RADS Confusion Matrix (Epoch {epoch+1})"
                            )
                            print(f"  Confusion matrix saved to: {cm_path}")
                        except Exception as e:
                            print(f"  Could not create confusion matrix: {e}")

                # Print combined score breakdown
                print("  Combined score breakdown:")
                print(f"    - Dice contribution: {0.5 * val_metrics['dice_score']:.4f}")
                print(f"    - Regression contribution: {0.25 * (100 - min(val_metrics['reg_mae'], 100)) / 100:.4f}")
                print(f"    - BI-RADS contribution: {0.25 * (val_metrics['birads_accuracy'] / 100):.4f}")
                print(f"    - Total combined score: {combined_score:.4f}")

                # Save best model when combined score improves
                if combined_score > best_score:
                    best_score = combined_score
                    for path in [config.model_save_path, paths['best']]:
                        save_checkpoint(
                            model, optimizer, epoch,
                            {**val_metrics, 'combined_score': combined_score},
                            train_history, config, path
                        )
                    print(f"Best model saved! Combined Score: {combined_score:.4f}")
                    best_epoch_idx = epoch  # Keep track of best epoch index

        # Save final model at end of training
        save_checkpoint(model, optimizer, epoch, val_metrics, train_history, config, paths['final'])

        # Calculate total training time
        training_time = time.time() - start_time
        print("\n=== Training Session Completed! ===")
        print(f"Total training time: {training_time/3600:.2f} hours")
        print(f"Best Combined Score: {best_score:.4f}")
        print("Model files saved:")
        print(f"- Best model: {paths['best']}")
        print(f"- Final model: {paths['final']}")
        print(f"- Latest checkpoint: {paths['checkpoint']}")

        # Hook for t-SNE (if user sets --tsne_plot) 

        if config.tsne_plot:
            print("\n[INFO] Generating t-SNE plot from validation embeddings...")

            # 1) Extract embeddings from validation set
            embeddings, birads_labels = extract_encoder_embeddings(
                model,
                valid_dataloader,
                device,
                max_batches=50   # optional limit to avoid huge memory usage
            )

            # 2) Create t-SNE plot
            tsne_path = create_tsne_plot(
                embeddings=embeddings,
                birads_labels=birads_labels,
                output_path='output/plots',   # or wherever you want to save
                filename='tsne_birads.png'
            )
            print(f"[INFO] t-SNE plot saved at: {tsne_path}")

        # Generate and save training plots
        try:
            # Create plots directory if it doesn't exist
            plot_dir = Path(config.logs_file_path).parent / 'plots'
            plot_dir.mkdir(parents=True, exist_ok=True)

            # Generate training history plot
            training_plot_path = create_training_plots(
                train_history['train_loss'],
                train_history['valid_loss'],
                train_history['train_dice'],
                train_history['valid_dice'],
                train_history['train_reg_mae'],
                train_history['valid_reg_mae'],
                config.logs_file_path,
                paths['timestamp']
            )

            # Plot BI-RADS accuracy separately
            plt.figure(figsize=(10, 6))
            plt.plot(train_history['train_birads_acc'], label='Train Accuracy')
            plt.plot(train_history['valid_birads_acc'], label='Valid Accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.title('BI-RADS Classification Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)

            birads_plot_path = plot_dir / f"birads_accuracy_{paths['timestamp']}.png"
            plt.savefig(birads_plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Training history plots saved to: {training_plot_path}")
            print(f"BI-RADS accuracy plot saved to: {birads_plot_path}")

            # Generate learning rate plot
            lr_plot_path = create_learning_rate_plot(
                lr_history,
                config.logs_file_path,
                paths['timestamp']
            )
            print(f"Learning rate plot saved to: {lr_plot_path}")

            # Generate best and final epoch confusion matrices
            best_cm_path, final_cm_path = create_epoch_confusion_matrices(
                val_predictions_history,
                val_targets_history,
                best_epoch_idx,
                len(val_predictions_history) - 1,  # final epoch index
                config.logs_file_path,
                paths['timestamp']
            )
            print(f"Best epoch confusion matrix saved to: {best_cm_path}")
            print(f"Final epoch confusion matrix saved to: {final_cm_path}")

            # Generate segmentation visualizations
            viz_path = visualize_segmentations(
                model,
                valid_dataloader,
                device,
                config.logs_file_path,
                paths['timestamp']
            )
            print(f"Segmentation visualization saved to: {viz_path}")

        except Exception as e:
            print(f"Error creating plots: {e}")
            import traceback
            traceback.print_exc()
            # Set default values if plots weren't created
            training_plot_path = None
            birads_plot_path = None
            lr_plot_path = None
            best_cm_path = None
            final_cm_path = None
            viz_path = None

        # Write summary report
        write_summary(
            model_stats,
            training_time,
            best_score,
            val_metrics,
            {
                **paths,
                'distribution_plot': distribution_plot_path,
                'plots': training_plot_path if 'training_plot_path' in locals() else None,
                'birads_plot': birads_plot_path if 'birads_plot_path' in locals() else None,
                'lr_plot': lr_plot_path if 'lr_plot_path' in locals() else None,
                'seg_viz': viz_path if 'viz_path' in locals() else None,
                'best_cm': best_cm_path if 'best_cm_path' in locals() else None,
                'final_cm': final_cm_path if 'final_cm_path' in locals() else None
            },
            config
        )

        return paths['best']

    except Exception as e:
        print(f"\n!!! ERROR in training process: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
