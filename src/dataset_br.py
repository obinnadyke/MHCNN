"""
dataset_br.py - Dataset classes for mammogram image processing (c) itrustal.com
Provisions:
Dataset classes for loading, preprocessing, and augmenting 
mammogram images and their corresponding density masks. Handles DICOM format 
images and supports both the training and inference workflows 
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# Import from losses_metrics for BI-RADS calculation
from losses_metrics import percentage_to_birads_category


class MammoDataset(Dataset):
    """
    Dataset for loading and preprocessing mammogram images and density masks 
    Returns - tuple: (image_tensor, mask_tensor, density_percentage_tensor, birads_category_tensor)
    """

    def __init__(
        self,
        image_paths,
        mask_paths=None,
        augmentations=False,
        target_size=(512, 512),
        to_3channels=True
    ):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augmentations = augmentations
        self.target_size = target_size
        self.to_3channels = to_3channels

        # (A) Geometry transforms for both image and mask
        self.geom_aug = A.Compose([
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.3),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.3),
        ], p=0.8)

        # (B) Color/noise transforms for IMAGE ONLY
        self.color_aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 5), p=0.3),
        ], p=0.8)

        # (C) Final resize+normalize for the image
        self.image_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406] if to_3channels else [0.5],
                std=[0.229, 0.224, 0.225] if to_3channels else [0.5],
                max_pixel_value=255.0
            ),
        ])

        # (D) Only resize the mask (no color normalization)
        self.mask_transform = A.Compose([
            A.Resize(height=target_size[0], width=target_size[1]),
        ])

    def __len__(self):
        return len(self.image_paths)

    
    def __getitem__(self, index):
        # 1) Load DICOM image
        try:
            image, _ = self.load_dicom(self.image_paths[index], is_mask=False)
        except Exception as e:
            print(f"Error loading image {self.image_paths[index]}: {e}")
            return self.get_default_item()

        # 2) If mask paths exist
        if self.mask_paths is not None:
            try:
                mask, _ = self.load_dicom(self.mask_paths[index], is_mask=True)
                
                # For mammogram masks, we need to separately identify:
                # - Breast tissue (non-zero pixels)
                # - Dense tissue (higher intensity values)
                
                # Create breast mask (all non-zero pixels)
                breast_mask = np.zeros_like(mask, dtype=np.float32)
                breast_mask[mask > 0] = 1.0
                
                # Better thresholding for dense tissue
                # Skip Otsu and use simple threshold - more stable
                # Determine the maximum mask value (safely)
                if np.any(mask):  # Check if there are any non-zero values
                    max_val = float(np.max(mask))
                    min_val = float(np.min(mask[mask > 0])) if np.any(mask > 0) else 0.0
                else:
                    max_val = 1.0
                    min_val = 0.0
                    
                # Use simple percentage-based thresholding
                if max_val > 1.0:  # If mask is not normalized to 0-1
                    # Use 50% of the range between min and max (of non-zero values)
                    if max_val > min_val:
                        dense_threshold = min_val + 0.5 * (max_val - min_val)
                    else:
                        dense_threshold = max_val * 0.5
                else:
                    # For already normalized masks
                    dense_threshold = 0.5
                    
                # Safety check - ensure we have a scalar threshold
                if not np.isscalar(dense_threshold):
                    print(f"Warning: Non-scalar threshold for {self.mask_paths[index]}")
                    dense_threshold = 0.5 * float(max_val)
                
                # Create dense tissue mask
                dense_mask = np.zeros_like(mask, dtype=np.float32)
                dense_mask[mask > dense_threshold] = 1.0
                
                # Compute breast density properly
                total_breast_pixels = np.count_nonzero(breast_mask)
                if total_breast_pixels > 0:
                    dense_pixels = np.count_nonzero(dense_mask)
                    density_percentage = (dense_pixels / total_breast_pixels) * 100
                    
                    # Apply reasonable limits to density percentage
                    density_percentage = min(max(density_percentage, 0.0), 100.0)
                else:
                    density_percentage = 0.0
                    
                # Use dense_mask for segmentation training
                mask = dense_mask

                # BI-RADS category calculation (0-indexed for modeling)
                birads_category = percentage_to_birads_category(density_percentage) - 1
                birads_tensor = torch.tensor(birads_category, dtype=torch.long)
                density_tensor = torch.tensor([density_percentage / 100.0], dtype=torch.float32)

                # Apply geometry transforms if requested
                if self.augmentations:
                    geom_data = self.geom_aug(image=image, mask=mask)
                    image, mask = geom_data['image'], geom_data['mask']

                    # Color transforms only on the image
                    color_data = self.color_aug(image=image)
                    image = color_data['image']
                    # mask is unchanged

                # Final transforms
                img_data = self.image_transform(image=image)
                msk_data = self.mask_transform(image=mask)  # 'image' param name is standard in Albumentations

                image = img_data['image']
                mask = msk_data['image']

                # Convert to torch Tensors
                image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

                # Ensure mask is single-channel
                if mask.ndim == 2:
                    mask = np.expand_dims(mask, axis=-1)
                mask_tensor = torch.from_numpy(mask.transpose(2, 0, 1)).float()

                return image_tensor, mask_tensor, density_tensor, birads_tensor

            except Exception as e:
                print(f"Error loading mask {self.mask_paths[index]}: {e}")
                import traceback
                traceback.print_exc()  # Print the full stack trace for debugging
                return self.get_default_item()

        # 3) If no mask paths (inference mode)
        else:
            img_data = self.image_transform(image=image)
            image = img_data['image']
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

            # Return placeholders for mask, density, birads
            return image_tensor, torch.zeros((1, *self.target_size)), torch.zeros(1), torch.zeros(1, dtype=torch.long)

    
    def load_dicom(self, path, is_mask=False):
        """
        Load and preprocess DICOM image with improved mask handling 
        """
        try:
            # Load DICOM file with proper error handling
            dcm = pydicom.dcmread(path, force=True)
            
            # Handle compressed formats
            if hasattr(dcm, 'file_meta') and hasattr(dcm.file_meta, 'TransferSyntaxUID'):
                transfer_syntax = dcm.file_meta.TransferSyntaxUID
                compressed_syntaxes = [
                    pydicom.uid.JPEG2000Lossless,
                    pydicom.uid.JPEG2000,
                    pydicom.uid.JPEGLossless,
                    pydicom.uid.JPEGLSLossless,
                    pydicom.uid.RLELossless,
                ]
                
                if transfer_syntax in compressed_syntaxes:
                    try:
                        # Decompress image
                        dcm.decompress()
                    except Exception as e:
                        print(f"Warning: Could not decompress {path}: {e}")
            
            # Get pixel data
            try:
                pixel_array = dcm.pixel_array
            except Exception as e:
                print(f"Error accessing pixel_array in {path}: {e}")
                # Create a default black image of the expected size
                if hasattr(dcm, 'Rows') and hasattr(dcm, 'Columns'):
                    pixel_array = np.zeros((dcm.Rows, dcm.Columns), dtype=np.uint8)
                else:
                    # Default size if dimensions not available
                    pixel_array = np.zeros((512, 512), dtype=np.uint8)

            if not is_mask:
                # For images, apply intensity transformation
                try:
                    pixel_array = apply_modality_lut(pixel_array, dcm)
                except Exception as e:
                    print(f"Warning: Could not apply modality LUT for {path}: {e}")
                
                # Scale to 0-255
                if pixel_array.dtype != np.uint8:
                    pixel_array = pixel_array.astype(np.float32)
                    min_val, max_val = float(np.min(pixel_array)), float(np.max(pixel_array))
                    if max_val > min_val:
                        # Normalize to [0, 1] range
                        pixel_array = (pixel_array - min_val) / (max_val - min_val)
                    else:
                        # Flat image with all pixels the same value
                        pixel_array = np.zeros_like(pixel_array, dtype=np.float32)
                    
                    # Scale to [0, 255] and convert to uint8
                    pixel_array = (pixel_array * 255).astype(np.uint8)
                    
                # Convert to 3 channels if needed
                if len(pixel_array.shape) == 2 and self.to_3channels:
                    pixel_array = cv2.cvtColor(pixel_array, cv2.COLOR_GRAY2RGB)
            else:
                # For masks, just normalize to uint8 for consistent thresholding
                if pixel_array.dtype != np.uint8:
                    # Scale to [0, 255] range
                    min_val, max_val = float(np.min(pixel_array)), float(np.max(pixel_array))
                    if max_val > min_val:
                        pixel_array = ((pixel_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)
                    else:
                        pixel_array = np.zeros_like(pixel_array, dtype=np.uint8)
                
                # Keep as single channel for masks
                if len(pixel_array.shape) > 2:
                    # Take first channel if multi-channel
                    pixel_array = pixel_array[:, :, 0]

            return pixel_array, dcm

        except Exception as e:
            error_msg = f"Error reading DICOM {path}: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            raise RuntimeError(error_msg)


    def get_default_item(self):
        """Return default tensors in case of errors"""
        if self.to_3channels:
            image = torch.zeros((3, *self.target_size), dtype=torch.float32)
        else:
            image = torch.zeros((1, *self.target_size), dtype=torch.float32)
        mask = torch.zeros((1, *self.target_size), dtype=torch.float32)
        density = torch.zeros(1, dtype=torch.float32)
        birads = torch.zeros(1, dtype=torch.long)

        return image, mask, density, birads


class MammoEvaluation(Dataset):
    """
    Dataset for inference/evaluation only (no masks required) 
    Returns - tuple: (img_id, image_tensor)
    """
    def __init__(self, path, dataset, split='test'):
        self.images_dir = os.path.join(path, dataset, split, 'input_image')
        self.image_paths = sorted([
            os.path.join(self.images_dir, f)
            for f in os.listdir(self.images_dir)
            if f.endswith('.dcm')
        ])
        self.transform = A.Compose([
            A.Resize(height=512, width=512),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img_id = os.path.basename(img_path).split('.')[0]

        try:
            # Load image and convert to RGB
            dcm = pydicom.dcmread(img_path, force=True)

            # Handle compressed formats
            transfer_syntax = dcm.file_meta.TransferSyntaxUID if hasattr(dcm, 'file_meta') else None
            compressed_syntaxes = [
                pydicom.uid.JPEG2000Lossless,
                pydicom.uid.JPEG2000,
                pydicom.uid.JPEGLossless,
                pydicom.uid.JPEGLSLossless,
                pydicom.uid.RLELossless,
            ]
            if transfer_syntax in compressed_syntaxes:
                try:
                    dcm.decompress()
                except Exception as e:
                    print(f"Warning: Could not decompress {img_path}: {e}")

            image = dcm.pixel_array
            
            try:
                image = apply_modality_lut(image, dcm)
            except Exception as e:
                print(f"Warning: Could not apply modality LUT for {img_path}: {e}")

            # Normalize to 0-255
            if image.dtype != np.uint8:
                image = image.astype(np.float32)
                if image.max() > image.min():
                    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
                    image = (image * 255).astype(np.uint8)
                else:
                    image = np.zeros_like(image, dtype=np.uint8)

            # Convert to RGB
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

            # Apply transformations
            transformed = self.transform(image=image)
            image = transformed['image']

            # Convert to tensor (CHW format)
            image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float()

            return img_id, image_tensor

        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Return a default tensor
            return img_id, torch.zeros((3, 512, 512), dtype=torch.float32)


def create_data_loaders(config, device_manager=None):
    """
    Create train and validation data loaders with optimal configuration 
    Returns - tuple: (train_dataloader, valid_dataloader, num_train_samples, num_valid_samples)
    """
    import glob
    from sklearn.model_selection import train_test_split
    
    # Extract paths from config
    data_root = config['data_path']
    image_dir = os.path.join(data_root, 'input_image')
    mask_dir = os.path.join(data_root, 'dense_mask')

    # Find all DICOM images and masks
    all_image_paths = sorted(glob.glob(os.path.join(image_dir, '*.dcm')))
    all_mask_paths = sorted(glob.glob(os.path.join(mask_dir, '*.dcm')))

    # Validate data
    if len(all_image_paths) == 0:
        raise FileNotFoundError(f"No DICOM images found in {image_dir}")
    if len(all_mask_paths) == 0:
        raise FileNotFoundError(f"No DICOM masks found in {mask_dir}")
    if len(all_image_paths) != len(all_mask_paths):
        raise ValueError(f"Number of images ({len(all_image_paths)}) and masks ({len(all_mask_paths)}) do not match.")

    # Split data: 80% train, 20% validation
    train_imgs, val_imgs, train_masks, val_masks = train_test_split(
        all_image_paths, all_mask_paths, test_size=0.2, random_state=42
    )
    print(f"Train samples: {len(train_imgs)}, Val samples: {len(val_imgs)}")

    # Get target size from config, default to 512
    target_size = config.get('output_size', 512)
    if not isinstance(target_size, tuple):
        target_size = (target_size, target_size)

    # Create datasets
    train_dataset = MammoDataset(
        image_paths=train_imgs,
        mask_paths=train_masks,
        augmentations=True,
        target_size=target_size
    )
    valid_dataset = MammoDataset(
        image_paths=val_imgs,
        mask_paths=val_masks,
        augmentations=False,
        target_size=target_size
    )

    # Analyze and plot density distribution | validate the density calculation
    plot_density_distribution(train_dataset, valid_dataset, 
                             output_dir=os.path.join(data_root, 'analysis'))

    # Determine optimal batch size and workers for parallel processing
    num_gpus = 1 if device_manager is None else max(1, torch.cuda.device_count())

    # Scale batch size if using multiple GPUs and config values are unchanged from default
    train_batch = config.get('train_batch_size', 4)
    valid_batch = config.get('valid_batch_size', 4)
    if num_gpus > 1 and not config.get('fixed_batch_size', False):
        train_batch *= num_gpus
        valid_batch *= num_gpus
        print(f"Scaled batch size for {num_gpus} GPUs: train={train_batch}, valid={valid_batch}")

    # Adjust workers based on CPU cores and GPU count
    workers = config.get('num_workers', 0)
    if workers == 0:  # Default value
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        # Use 2 workers per GPU with a maximum of half available CPUs
        workers = min(2 * num_gpus, cpu_count // 2)
        print(f"Using {workers} workers for data loading")

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=train_batch,
        shuffle=True,
        num_workers=workers,
        pin_memory=True,  # Speeds up host to GPU transfers
        drop_last=True    # Avoid batch size issues with last batch
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=valid_batch,
        shuffle=False,
        num_workers=workers,
        pin_memory=True
    )
    
    return train_dataloader, valid_dataloader, len(train_dataset), len(valid_dataset)


def plot_density_distribution(train_dataset, valid_dataset, output_dir='./analysis', num_samples=500):
    """
    Analyze and plot the density distribution in the datasets 
    Args:
        train_dataset (MammoDataset): Training dataset
        valid_dataset (MammoDataset): Validation dataset
        output_dir (str): Directory to save plots
        num_samples (int): Number of samples to analyze (for speed)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    train_densities = []
    valid_densities = []
    train_birads = []
    valid_birads = []
    
    # Sample from training set
    num_train = min(num_samples, len(train_dataset))
    indices = np.random.choice(len(train_dataset), num_train, replace=False)
    
    for idx in indices:
        _, _, density, birads = train_dataset[idx]
        train_densities.append(density.item() * 100)  # Convert to percentage
        train_birads.append(birads.item() + 1)  # Convert from 0-indexed to 1-indexed
    
    # Sample from validation set
    num_valid = min(num_samples, len(valid_dataset))
    indices = np.random.choice(len(valid_dataset), num_valid, replace=False)
    
    for idx in indices:
        _, _, density, birads = valid_dataset[idx]
        valid_densities.append(density.item() * 100)  # Convert to percentage
        valid_birads.append(birads.item() + 1)  # Convert from 0-indexed to 1-indexed
    
    # 1. Plot density distribution histograms
    plt.figure(figsize=(12, 6))
    plt.hist(train_densities, bins=20, alpha=0.7, label='Training')
    plt.hist(valid_densities, bins=20, alpha=0.7, label='Validation')
    plt.xlabel('Breast Density Percentage')
    plt.ylabel('Number of Samples')
    plt.legend()
    plt.title('Distribution of Breast Density Percentages')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'density_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Plot BI-RADS category distribution
    plt.figure(figsize=(10, 6))
    
    # Count occurrences of each BI-RADS category
    birads_counts_train = {i: train_birads.count(i) for i in range(1, 5)}
    birads_counts_valid = {i: valid_birads.count(i) for i in range(1, 5)}
    
    # Calculate percentages
    train_total = len(train_birads)
    valid_total = len(valid_birads)
    birads_pct_train = {k: v / train_total * 100 for k, v in birads_counts_train.items()}
    birads_pct_valid = {k: v / valid_total * 100 for k, v in birads_counts_valid.items()}
    
    # Plot
    categories = list(range(1, 5))
    width = 0.35
    
    plt.bar([x - width/2 for x in categories], 
            [birads_pct_train.get(i, 0) for i in categories], 
            width, label='Training')
    plt.bar([x + width/2 for x in categories], 
            [birads_pct_valid.get(i, 0) for i in categories], 
            width, label='Validation')
    
    plt.xlabel('BI-RADS Category')
    plt.ylabel('Percentage')
    plt.title('Distribution of BI-RADS Categories')
    plt.xticks(categories)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'birads_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Save summary statistics
    with open(os.path.join(output_dir, 'density_stats.txt'), 'w') as f:
        f.write("=== Breast Density Statistics ===\n\n")
        
        f.write("Training Set:\n")
        f.write(f"  Samples: {len(train_densities)}\n")
        f.write(f"  Mean: {np.mean(train_densities):.2f}%\n")
        f.write(f"  Median: {np.median(train_densities):.2f}%\n")
        f.write(f"  Min: {np.min(train_densities):.2f}%\n")
        f.write(f"  Max: {np.max(train_densities):.2f}%\n")
        f.write(f"  BI-RADS Distribution: {birads_counts_train}\n\n")
        
        f.write("Validation Set:\n")
        f.write(f"  Samples: {len(valid_densities)}\n")
        f.write(f"  Mean: {np.mean(valid_densities):.2f}%\n")
        f.write(f"  Median: {np.median(valid_densities):.2f}%\n")
        f.write(f"  Min: {np.min(valid_densities):.2f}%\n")
        f.write(f"  Max: {np.max(valid_densities):.2f}%\n")
        f.write(f"  BI-RADS Distribution: {birads_counts_valid}\n")
    
    print(f"Density distribution analysis saved to {output_dir}")
    
    return {
        'train_densities': train_densities,
        'valid_densities': valid_densities,
        'train_birads': train_birads,
        'valid_birads': valid_birads
    }
