#!/usr/bin/env python
"""
predict.py - Breast Density Analysis Inference Script |(c) itrustal.com
Provisions:
1. Segment breast density tissue
2. Calculate percentage breast density (PBD)
3. Directly classify BI-RADS category
4. Create visual overlays and save results as DICOM files 
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np
import cv2
import pydicom
from pydicom.uid import ExplicitVRLittleEndian, generate_uid
from pydicom.pixel_data_handlers.util import apply_modality_lut
from pathlib import Path
from tqdm import tqdm
import warnings
import time
import json
from collections import Counter
import multiprocessing
import matplotlib.pyplot as plt

# Import from refactored modules
from device_checker import DeviceManager, move_to_device
from density_model import load_model_for_inference
from losses_metrics import get_birads_from_percentage, dice_coef


# Suppress warnings
warnings.filterwarnings("ignore")

# Default parameters
OUTPUT_SIZE = 512  # Consistent with training


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Breast Density Analysis Inference')

    # Input/output paths
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing input DICOM images')
    parser.add_argument('--output_dir', type=str, default='Results',
                        help='Directory to save results')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')

    # Model parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run inference on')
    parser.add_argument('--segmentation_model', default='Unet', type=str,
                        help='Segmentation model architecture (default: Unet)')
    parser.add_argument('--encoder', default='resnet101', type=str,
                        help='Encoder backbone for segmentation model (default: resnet101)')
    parser.add_argument('--use_deep_decoder', action='store_true',
                        help='Use deeper custom UNet decoder (if used during training)')

    # Output options
    parser.add_argument('--overlay_alpha', type=float, default=0.7,
                        help='Alpha value for overlay blending (0.0-1.0)')
    parser.add_argument('--save_images', action='store_true',
                        help='Save PNG images in addition to DICOM files')
    parser.add_argument('--save_csv', action='store_true',
                        help='Save results as CSV file')
    parser.add_argument('--batch_size', type=int, default=2,
                        help='Batch size for inference')
    parser.add_argument('--debug_segmentation', action='store_true',
                        help='Save raw segmentation masks for debugging')

    # Ground truth (optional)
    parser.add_argument('--ground_truth_dir', type=str, default=None,
                        help='Directory containing ground truth mask DICOM files (for evaluation)')

    # GPU settings
    parser.add_argument('--device_ids', default=None, type=str,
                        help='Comma-separated list of GPU IDs to use (e.g., "0,1,2").')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for data loading (0=auto)')

    return parser.parse_args()


def load_dicom(path):
    """
    Load DICOM image and return preprocessed array 
    Returns - tuple: (preprocessed_image, original_dicom_object, window_center, window_width)
    """
    try:
        dcm = pydicom.dcmread(path, force=True)

        # Handle compressed DICOM formats
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
                print(f"Warning: Failed to decompress {path}: {e}")

        # Get pixel array
        image = dcm.pixel_array

        # Extract window settings if available (for better visualization)
        window_center = window_width = None
        if hasattr(dcm, 'WindowCenter') and hasattr(dcm, 'WindowWidth'):
            try:
                window_center = dcm.WindowCenter
                if isinstance(window_center, pydicom.multival.MultiValue):
                    window_center = window_center[0]

                window_width = dcm.WindowWidth
                if isinstance(window_width, pydicom.multival.MultiValue):
                    window_width = window_width[0]
            except:
                pass

        # Apply modality LUT (window leveling)
        try:
            if hasattr(dcm, 'RescaleSlope') or hasattr(dcm, 'RescaleIntercept'):
                image = apply_modality_lut(image, dcm)
        except Exception as e:
            print(f"Warning: Could not apply modality LUT for {path}: {e}")

        # Apply custom windowing if available
        if window_center is not None and window_width is not None:
            try:
                min_value = window_center - window_width // 2
                max_value = window_center + window_width // 2
                image = np.clip(image, min_value, max_value)
            except:
                pass

        # Normalize to 0-255 range
        if image.dtype != np.uint8:
            image = image.astype(np.float32)
            image_min = float(np.min(image))
            image_max = float(np.max(image))

            # Prevent division by zero
            if image_max > image_min:
                image = ((image - image_min) / (image_max - image_min)) * 255.0
            else:
                image = np.zeros_like(image)

            image = np.clip(image, 0, 255).astype(np.uint8)

        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 1:
            image = cv2.cvtColor(image[:,:,0], cv2.COLOR_GRAY2RGB)

        return image, dcm, window_center, window_width

    except Exception as e:
        raise RuntimeError(f"Error reading DICOM {path}: {str(e)}")


def preprocess_image(image, target_size=(OUTPUT_SIZE, OUTPUT_SIZE)):
    """
    Preprocess image for model input 
    Returns - tuple: (tensor, original_size, padding_info)
    """
    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_h, new_w = int(h * scale), int(w * scale)

    # Resize while maintaining aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Create canvas of target size
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    y_offset = (target_size[0] - new_h) // 2
    x_offset = (target_size[1] - new_w) // 2

    # Place resized image on canvas
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    # Normalize and convert to tensor
    tensor = torch.from_numpy(canvas.transpose(2, 0, 1)).float() / 255.0

    # Normalize with ImageNet means and stds to match training
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    tensor = (tensor - mean) / std

    return tensor, (h, w), (y_offset, x_offset, new_h, new_w)


def get_colormap_for_birads(birads_category):
    """
    Return appropriate colormap based on BI-RADS category
    Args - birads_category (int): BI-RADS category (1-4)
    Returns - int: OpenCV colormap code
    """
    # Use different colormaps for different BI-RADS categories
    # BI-RADS 1-2: Green to Blue spectrum (less dense)
    # BI-RADS 3-4: Yellow to Red spectrum (more dense)
    if birads_category <= 2:
        return cv2.COLORMAP_COOL  # Blue-cyan-green
    else:
        return cv2.COLORMAP_JET  # Blue-cyan-green-yellow-red


def enhance_segmentation_visibility(mask, threshold=0.2):
    """
    Enhance the visibility of the segmentation mask 
    Returns - numpy.ndarray: Enhanced mask
    """
    # Apply threshold to create a binary segmentation
    binary_mask = (mask > threshold).astype(np.float32)

    # Enhance the original mask using the binary mask
    enhanced_mask = mask.copy()
    enhanced_mask[binary_mask > 0] = np.maximum(enhanced_mask[binary_mask > 0], 0.5)

    return enhanced_mask


def create_overlay(original_image, density_mask, predicted_percentage, birads_category, birads_text, blend_alpha=0.7):
    """
    Create colored overlay with BI-RADS text and visualization.
    Args:
        original_image (numpy.ndarray): Original mammogram image
        density_mask (numpy.ndarray): Predicted density mask
        predicted_percentage (float): Predicted percentage breast density (PBD)
        birads_category (int): Predicted BI-RADS category (1-4)
        birads_text (str): BI-RADS classification text
        blend_alpha (float): Alpha value for blending
    Returns:
        numpy.ndarray: Overlay image with improved quality and larger text
    """
    # Ensure we're working with RGB images and not empty (safety checks)
    if original_image is None or original_image.size == 0:
        print("Warning: (Empty) No original image received")
        # Return a simple placeholder image to avoid crashes
        return np.zeros((512, 512, 3), dtype=np.uint8)

    if original_image.shape[2] != 3:
        # Convert grayscale to RGB
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)

    # Additional safety check for density mask
    if density_mask is None or density_mask.size == 0:
        print("Warning: Empty density mask received")
        # Use a placeholder mask (all zeros)
        density_mask = np.zeros(original_image.shape[:2], dtype=np.float32)

    try:
        # Create a clean copy of the original image
        original_copy = original_image.copy()

        # Create a mask for the entire breast area (to isolate from background)
        # First threshold the grayscale image to find the breast region
        gray = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
        _, breast_mask = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)

        # Clean up the breast mask with morphological operations
        kernel = np.ones((5, 5), np.uint8)
        breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_CLOSE, kernel)
        breast_mask = cv2.morphologyEx(breast_mask, cv2.MORPH_OPEN, kernel)

        # Create black background image (for areas outside breast)
        h, w = original_image.shape[:2]
        black_background = np.zeros((h, w, 3), dtype=np.uint8)

        # Blend breast region from original image onto black background
        breast_mask_3ch = cv2.cvtColor(breast_mask, cv2.COLOR_GRAY2BGR)
        breast_mask_3ch = breast_mask_3ch.astype(np.float32) / 255.0
        background_image = black_background * (1 - breast_mask_3ch) + original_copy * breast_mask_3ch
        background_image = background_image.astype(np.uint8)

        # Enhance the segmentation mask for better visibility
        enhanced_mask = enhance_segmentation_visibility(density_mask, threshold=0.15)

        # Create dynamic color map based on BI-RADS category
        if birads_category <= 2:
            # Use cool colors for lower categories (blue/green)
            colormap = cv2.COLORMAP_COOL
        else:
            # Use hot colors for higher categories (yellow/red)
            colormap = cv2.COLORMAP_JET

        # Create heatmap from enhanced density mask
        heatmap = cv2.applyColorMap(
            (enhanced_mask * 255).astype(np.uint8),
            colormap
        )

        # Only apply heatmap within the breast area
        heatmap_masked = heatmap * (breast_mask_3ch > 0).astype(np.uint8)

        # Create soft mask for density areas only
        density_area_mask = np.expand_dims(enhanced_mask, axis=2)
        density_area_mask = np.repeat(density_area_mask, 3, axis=2)

        # Create the final overlay by blending:
        # 1. Use black background where there's no breast
        # 2. Use original image where breast is present but no density
        # 3. Blend heatmap only over density areas
        overlay = background_image * (1 - density_area_mask) + heatmap_masked * density_area_mask * blend_alpha + background_image * (1 - blend_alpha) * density_area_mask
        overlay = overlay.astype(np.uint8)

        # Add contour lines around segmented areas for better visibility
        binary_mask = (enhanced_mask > 0.15).astype(np.uint8)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

        # Use a bold, clear font
        font = cv2.FONT_HERSHEY_TRIPLEX  # A bolder, more visible font

        # Create text with good size and format
        text = f"PBD: {predicted_percentage:.1f}% | BI-RADS {birads_category} ({birads_text})"

        # Calculate text size for centering
        font_scale = w / 750.0 if w > 100 else 1.0  # Scale based on image width
        font_scale = min(max(font_scale, 0.8), 2.5)  # Keep between 0.8 and 2.5
        thickness = 2  # Text thickness
        text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

        # Position text at the top center
        text_x = (w - text_size[0]) // 2  # Center horizontally
        text_y = text_size[1] + 20  # Position near top with margin

        # Add a multi-directional outline for maximum visibility
        shadow_thickness = 7  # Thicker shadow

        # Draw thick black outline in multiple directions
        offsets = [(-2,-2), (-2,0), (-2,2), (0,-2), (0,2), (2,-2), (2,0), (2,2)]
        for dx, dy in offsets:
            cv2.putText(
                overlay,
                text,
                (text_x + dx, text_y + dy),
                font,
                font_scale,
                (0, 0, 0),
                shadow_thickness,
                cv2.LINE_AA
            )

        # Draw white text on top of shadow with increased thickness
        cv2.putText(
            overlay,
            text,
            (text_x, text_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness + 1,  # Slightly thicker for better visibility
            cv2.LINE_AA
        )

        return overlay

    except Exception as e:
        print(f"Error in create_overlay: {e}")
        import traceback
        traceback.print_exc()
        # Return the original image as fallback
        return original_image



def save_dicom(original_dcm, image, output_path):
    """
    Save result as DICOM with original metadata.

    Args:
        original_dcm (Dataset): Original DICOM object
        image (np.ndarray): Image array (RGB format) to save
        output_path: Path to save DICOM file
    """
    try:
        # Ensure image is exactly 512x512
        if image.shape[0] != 512 or image.shape[1] != 512:
            image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_CUBIC)

        # Convert image from RGB to BGR (OpenCV standard)
        if image.shape[2] == 3:  # Make sure it's color
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image

        # Copy original dataset to preserve metadata
        ds = original_dcm.copy()

        # Set new SOP Instance UID
        ds.SOPInstanceUID = generate_uid()

        # Set required image-related tags
        ds.Rows, ds.Columns = image_bgr.shape[:2]
        ds.SamplesPerPixel = 3
        ds.PhotometricInterpretation = "RGB"
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PlanarConfiguration = 0  # Color-by-pixel (essential for RGB)

        # Set transfer syntax
        ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds.is_little_endian = True
        ds.is_implicit_VR = False

        # Remove potentially conflicting attributes
        for attr in ['NumberOfFrames', 'PixelSpacing', 'ImagerPixelSpacing']:
            if hasattr(ds, attr):
                delattr(ds, attr)

        # Set to Secondary Capture
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
        if hasattr(ds, 'Modality'):
            ds.Modality = 'OT'  # Other

        # Add annotations to describe processing
        ds.DerivationDescription = "Breast density analysis with segmentation overlay"

        # Set pixel data
        ds.PixelData = image_bgr.tobytes()

        # Save the file
        ds.save_as(str(output_path))

    except Exception as e:
        print(f"Error saving DICOM to {output_path}: {e}")
        # Try saving as PNG instead
        try:
            output_png = str(output_path).replace('.dcm', '.png')
            cv2.imwrite(output_png, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            print(f"Saved as PNG instead: {output_png}")
        except Exception as png_error:
            print(f"Could not save as PNG either: {png_error}")


def process_batch(model, images, original_sizes, padding_infos, device, alpha=0.7):
    """
    Process a batch of images with robust error handling.

    Args:
        model: Model to use for inference
        images: Batch of images
        original_sizes: Original image sizes
        padding_infos: Padding information
        device: Device to run inference on
        alpha: Alpha value for overlay blending

    Returns:
        list: List of tuples (density_mask, predicted_percentage, birads_category, birads_text)
    """
    try:
        with torch.no_grad():
            # Forward pass through model
            outputs = model(images)

            # Handle different output formats from the model
            if isinstance(outputs, tuple) and len(outputs) == 3:
                seg_outputs, reg_outputs, cls_outputs = outputs
            elif isinstance(outputs, tuple) and len(outputs) == 2:
                seg_outputs, reg_outputs = outputs
                cls_outputs = None
            else:
                raise ValueError(f"Unexpected model output format: {type(outputs)}")

            # Handle tuple outputs from segmentation models (e.g., DeepLabV3+)
            if isinstance(seg_outputs, tuple):
                seg_outputs = seg_outputs[0]  # Use the main segmentation output

            results = []
            for i in range(images.size(0)):
                try:
                    # Get segmentation prediction
                    if seg_outputs.dim() == 4 and seg_outputs.size(1) == 1:
                        # Single channel output (standard segmentation)
                        mask = torch.sigmoid(seg_outputs[i, 0]).cpu().numpy()
                    elif seg_outputs.dim() == 4 and seg_outputs.size(1) > 1:
                        # Multi-channel output (e.g., background + foreground)
                        probs = torch.softmax(seg_outputs[i], dim=0).cpu().numpy()
                        mask = probs[1] if probs.shape[0] > 1 else probs[0]
                    else:
                        raise ValueError(f"Unexpected segmentation output shape: {seg_outputs.shape}")

                    # Get percentage prediction
                    predicted_percentage = reg_outputs[i, 0].item() * 100

                    # --- Option 1: Use regression-based label (this is our preferred method)
                    birads_category, birads_text = get_birads_from_percentage(predicted_percentage)

                    # For logging, print the regression outcome:
                    print(f"Image {i}: Predicted Percentage (regression) = {predicted_percentage:.1f}%, "
                          f"BI-RADS (from regression) = {birads_category} ({birads_text})")

                    # If you want to also see the classification output (for analysis), do:
                    if cls_outputs is not None:
                        _, cls_birads = torch.max(cls_outputs[i], 0)
                        cls_birads = cls_birads.item() + 1  # Convert from 0-indexed to 1-indexed
                        print(f"Image {i}: BI-RADS (classification) = {cls_birads}")

                    # --- Option 2: A Weighted Hybrid Approach (comment out) ---#
                    # This section demonstrates a way to combine the regression and classification outputs.
                    # Uncomment to test if desired.
                    #
                    # if cls_outputs is not None:
                    #     _, cls_birads = torch.max(cls_outputs[i], 0)
                    #     cls_birads = cls_birads.item() + 1  # 1-indexed classification result
                    #
                    #     # Convert regression percentage to a BI-RADS label using your threshold function
                    #     reg_birads, _ = get_birads_from_percentage(predicted_percentage)
                    #
                    #     # Set a weighting factor (alpha: weight for classification, 1 - alpha: weight for regression)
                    #     alpha = 0.5  # Adjust between 0.0 and 1.0 as needed
                    #     combined_value = alpha * cls_birads + (1 - alpha) * reg_birads
                    #     combined_birads = int(round(combined_value))
                    #     birads_category = combined_birads
                    #
                    #     # You can derive descriptive text from the final label (assuming you have a helper for that)
                    #     birads_text = get_birads_text_from_class(birads_category)
                    #
                    #     print(f"Image {i}: Weighted BI-RADS (combined) = {birads_category} ({birads_text})")
                    #
                    # Else, if classification is missing, continue to use regression:
                    # else:
                    #     birads_category, birads_text = get_birads_from_percentage(predicted_percentage)

                #=============================================Update Ends here ====================================================#

                    # Get BI-RADS prediction from classifier (former_method)
                    #if cls_outputs is not None:
                    #    _, birads_pred = torch.max(cls_outputs[i], 0)
                    #    birads_category = birads_pred.item() + 1  # Convert from 0-3 to 1-4
                    #else:
                        # Fallback to regression-based category if no classifier output
                    #    birads_category, _ = get_birads_from_percentage(predicted_percentage)

                    # Get BI-RADS description
                    _, birads_text = get_birads_from_percentage(predicted_percentage)

                    # Restore original size with error handling
                    try:
                        y_offset, x_offset, new_h, new_w = padding_infos[i]
                        cropped_mask = mask[y_offset:y_offset+new_h, x_offset:x_offset+new_w]
                        h, w = original_sizes[i]

                        # Verify mask isn't empty
                        if cropped_mask.size == 0:
                            print(f"Warning: Empty cropped mask for image {i}")
                            resized_mask = np.zeros((h, w), dtype=np.float32)
                        else:
                            # Use cubic interpolation for better quality upscaling
                            resized_mask = cv2.resize(cropped_mask, (w, h), interpolation=cv2.INTER_CUBIC)
                    except Exception as resize_error:
                        print(f"Error resizing mask: {resize_error}")
                        # Create an empty mask as fallback
                        h, w = original_sizes[i]
                        resized_mask = np.zeros((h, w), dtype=np.float32)

                    results.append((resized_mask, predicted_percentage, birads_category, birads_text))

                except Exception as e:
                    print(f"Error processing image {i} in batch: {e}")
                    # Create a placeholder result for this image to maintain batch alignment
                    h, w = original_sizes[i]
                    empty_mask = np.zeros((h, w), dtype=np.float32)
                    results.append((empty_mask, 0.0, 1, "Insufficient dense tissue"))

            return results
    except Exception as e:
        print(f"Batch processing error: {e}")
        import traceback
        traceback.print_exc()

        # Create placeholder results for the entire batch
        results = []
        for size in original_sizes:
            h, w = size
            empty_mask = np.zeros((h, w), dtype=np.float32)
            results.append((empty_mask, 0.0, 1, "Insufficient dense tissue"))

        return results


def main():
    """Main inference function"""
    args = parse_args()

    print("\n=== Breast Density Analysis Inference ===")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set up directories for different output types
    dicom_dir = output_dir / "dicom"
    dicom_dir.mkdir(exist_ok=True)

    if args.save_images:
        image_dir = output_dir / "images"
        image_dir.mkdir(exist_ok=True)

    if args.debug_segmentation:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(exist_ok=True)

    # Determine optimal batch size and workers for parallel processing
    if args.num_workers == 0:  # Auto-detect
        cpu_count = multiprocessing.cpu_count()
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        # Use 2 workers per GPU with a maximum of half available CPUs
        args.num_workers = min(2 * num_gpus, cpu_count // 2)
        print(f"Auto-configured {args.num_workers} workers for data loading")

    # Set up device
    device_manager = DeviceManager(args.device_ids)
    device = device_manager.device
    print(f"Using device: {device}")

    # Load model
    try:
        model_config = {
            'segmentation_model': args.segmentation_model,
            'encoder': args.encoder,
            'output_size': OUTPUT_SIZE,
            'use_deep_decoder': args.use_deep_decoder  # Pass this flag if model was trained with it
        }

        model = load_model_for_inference(args.model_path, model_config, device)

        # Apply DataParallel if multiple GPUs available
        if device_manager.multi_gpu:
            model = device_manager.setup_dataparallel(model)

        model.eval()
        print(f"Model loaded successfully from {args.model_path}")

    except Exception as e:
        print(f"Error loading model: {str(e)}")
        import traceback
        traceback.print_exc()
        return

    # Get all DICOM files
    input_dir = Path(args.input_dir)
    input_files = list(input_dir.glob('**/*.dcm')) + list(input_dir.glob('**/*.dicom'))  # Include subdirectories

    if not input_files:
        print(f"No DICOM files found in {args.input_dir} or its subdirectories")
        print("Checking for other image formats...")
        # Try looking for other image formats
        other_formats = list(input_dir.glob('**/*.png')) + list(input_dir.glob('**/*.jpg')) + list(input_dir.glob('**/*.jpeg'))
        if other_formats:
            print(f"Found {len(other_formats)} non-DICOM images. This script only processes DICOM files.")
            return
        else:
            print("No image files found.")
            return

    print(f"Processing {len(input_files)} DICOM images...")

    # Initialize metrics tracking
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    metrics_file = output_dir / f"inference_metrics_{timestamp}.json"

    all_metrics = {
        "timestamp": timestamp,
        "model_path": args.model_path,
        "input_dir": str(args.input_dir),
        "num_images": len(input_files),
        "per_image_results": {},
        "birads_distribution": {},
        "average_metrics": {}
    }

    # Set up for ground truth comparison (if provided)
    has_ground_truth = False
    if args.ground_truth_dir:
        ground_truth_dir = Path(args.ground_truth_dir)
        if ground_truth_dir.exists():
            has_ground_truth = True
            print(f"Ground truth directory found: {args.ground_truth_dir}")

    # Process images
    all_percentages = []
    all_birads = []
    successful_predictions = 0

    # Use batching for faster inference
    batch_size = args.batch_size
    progress_bar = tqdm(total=len(input_files), desc="Processing")

    for i in range(0, len(input_files), batch_size):
        batch_files = input_files[i:i+batch_size]
        batch_images = []
        batch_dcms = []
        batch_original_sizes = []
        batch_padding_infos = []
        batch_errors = []
        batch_window_params = []

        # Prepare batch
        for input_path in batch_files:
            try:
                # Load and preprocess image
                image, original_dcm, window_center, window_width = load_dicom(input_path)
                tensor, original_size, padding_info = preprocess_image(image)

                batch_images.append(tensor)
                batch_dcms.append((input_path, original_dcm, image))
                batch_original_sizes.append(original_size)
                batch_padding_infos.append(padding_info)
                batch_window_params.append((window_center, window_width))
                batch_errors.append(None)

            except Exception as e:
                print(f"Error processing {input_path}: {str(e)}")
                batch_errors.append(str(e))
                continue

        if not batch_images:
            # Skip batch if all images failed to load
            progress_bar.update(len(batch_files))
            continue

        # Stack tensors into batch
        batch_tensor = torch.stack(batch_images).to(device)

        # Process batch
        try:
            batch_results = process_batch(
                model, batch_tensor, batch_original_sizes, batch_padding_infos,
                device, args.overlay_alpha
            )

            # Process batch results
            result_idx = 0
            for file_idx, ((input_path, original_dcm, original_image), error) in enumerate(zip(batch_dcms, batch_errors)):
                if error:
                    # Skip files that failed to load
                    progress_bar.update(1)
                    continue

                try:
                    # Get results for this image
                    density_mask, percentage, birads_category, birads_text = batch_results[result_idx]
                    result_idx += 1

                    # Debugging: Save raw segmentation mask
                    if args.debug_segmentation:
                        # Create a nicer visualization of the raw mask
                        plt.figure(figsize=(10, 10))
                        plt.imshow(density_mask, cmap='jet')
                        plt.colorbar(label='Density Probability')
                        plt.title(f"Raw Segmentation Mask - {input_path.stem}")

                        debug_path = debug_dir / f"{input_path.stem}_segmask.png"
                        debug_path = debug_dir / f"{input_path.stem}_segmask.dcm"
                        plt.savefig(debug_path, dpi=150, bbox_inches='tight')
                        plt.close()

                        # Also save as a heatmap overlay
                        heatmap = cv2.applyColorMap(
                            (density_mask * 255).astype(np.uint8),
                            cv2.COLORMAP_JET
                        )
                        debug_overlay = cv2.addWeighted(
                            cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR),
                            0.7,
                            heatmap,
                            0.3,
                            0
                        )
                        cv2.imwrite(
                            str(debug_dir / f"{input_path.stem}_heatmap.png"),
                            debug_overlay
                        )

                    # Create overlay with improved visualization
                    overlay = create_overlay(
                        original_image,
                        density_mask,
                        percentage,
                        birads_category,
                        birads_text,
                        args.overlay_alpha
                    )

                    # Check if overlay is valid before saving
                    if overlay is None or overlay.size == 0:
                        print(f"Warning: Empty overlay generated for {input_path}. Using original image.")
                        # Use original image as fallback
                        overlay = original_image.copy()
                        # Add simple text to original image
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        cv2.putText(overlay, f"PBD: {percentage:.1f}% | BI-RADS {birads_category}",
                                    (20, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

                    # Save results
                    ext = input_path.suffix     # e.g., .dcm or .dicom
                    output_path = dicom_dir / f"{input_path.stem}_result{ext}"
                    save_dicom(original_dcm, overlay, output_path)

                    if args.save_images and overlay is not None and overlay.size > 0:
                        image_path = image_dir / f"{input_path.stem}_result.png"
                        try:
                            cv2.imwrite(str(image_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
                        except Exception as img_error:
                            print(f"Error saving PNG: {img_error}")
                            # Fallback to direct save without color conversion
                            try:
                                cv2.imwrite(str(image_path), overlay)
                            except:
                                print(f"Could not save image for {input_path} even with fallback")

                    # Track metrics
                    all_percentages.append(percentage)
                    all_birads.append(birads_category)
                    successful_predictions += 1

                    # Add to per-image results
                    all_metrics["per_image_results"][input_path.stem] = {
                        "percentage": percentage,
                        "birads_category": birads_category,
                        "birads_text": birads_text
                    }

                    progress_bar.update(1)

                except Exception as process_error:
                    print(f"Error in post-processing {input_path}: {process_error}")
                    import traceback
                    traceback.print_exc()
                    progress_bar.update(1)

        except Exception as batch_error:
            print(f"Error processing batch: {batch_error}")
            import traceback
            traceback.print_exc()
            # Continue with the next batch
            progress_bar.update(len(batch_files) - sum(1 for e in batch_errors if e is not None))

    progress_bar.close()

    print(f"\nSuccessfully processed {successful_predictions} out of {len(input_files)} images")

    # Calculate BI-RADS distribution
    if all_birads:
        birads_counter = Counter(all_birads)
        total_count = len(all_birads)
        for category in range(1, 5):
            count = birads_counter.get(category, 0)
            percentage = (count / total_count) * 100 if total_count > 0 else 0
            all_metrics["birads_distribution"][f"BI-RADS {category}"] = {
                "count": count,
                "percentage": percentage
            }

        # Print summary
        print("\n=== BI-RADS Distribution ===")
        for category, stats in all_metrics["birads_distribution"].items():
            print(f"{category}: {stats['count']} images ({stats['percentage']:.1f}%)")

    # Calculate average metrics
    if all_percentages:
        all_metrics["average_metrics"]["mean_density_percentage"] = sum(all_percentages) / len(all_percentages)
        all_metrics["average_metrics"]["median_density_percentage"] = sorted(all_percentages)[len(all_percentages) // 2]

        print(f"\nAverage Density: {all_metrics['average_metrics']['mean_density_percentage']:.2f}%")
        print(f"Median Density: {all_metrics['average_metrics']['median_density_percentage']:.2f}%")

    # Save metrics to JSON file
    with open(metrics_file, 'w') as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nProcessing completed!")
    print(f"Results saved to {args.output_dir}")
    print(f"Metrics saved to {metrics_file}")

    if args.debug_segmentation:
        print(f"Debug visualizations saved to {debug_dir}")

    # Save results to CSV if requested
    if args.save_csv:
        import csv
        csv_file = output_dir / f"results_{timestamp}.csv"
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Image', 'Density_Percentage', 'BI-RADS_Category', 'Description'])
            for image_id, result in all_metrics["per_image_results"].items():
                writer.writerow([
                    image_id,
                    f"{result['percentage']:.2f}%",
                    result['birads_category'],
                    result['birads_text']
                ])
        print(f"CSV report saved to {csv_file}")

    # Generate summary of most common issues (if any errors)
    errors_count = len(input_files) - successful_predictions
    if errors_count > 0:
        print(f"\nWARNING: {errors_count} images failed to process correctly.")
        print("Common issues with breast density prediction:")
        print("1. Low contrast in original mammograms")
        print("2. Model not recognizing breast tissue correctly")
        print("3. Incompatible DICOM format/metadata")
        print("Use --debug_segmentation flag to see raw segmentation masks for troubleshooting")

if __name__ == "__main__":
    main()
