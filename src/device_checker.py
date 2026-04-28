"""
device_checker.py - Utility for device placement and multi-GPU management |(c) itrustal.com 
"""

import os
import torch
import torch.nn as nn

class DeviceManager:
    """
    Manages device selection, setup, and multi-GPU configurations for training and inference 
    """

    def __init__(self, device_ids=None):
        """Initialize the device manager and set up available GPUs."""
        self.total_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        self.gpu_ids = list(range(self.total_gpus)) if self.total_gpus > 0 else []

        # Parse user-specified device IDs if provided
        if device_ids and self.total_gpus > 0:
            parsed_ids = []
            for i_str in device_ids.split(','):
                i = int(i_str)
                if 0 <= i < self.total_gpus:
                    parsed_ids.append(i)
            if parsed_ids:
                self.gpu_ids = parsed_ids

            # Update CUDA_VISIBLE_DEVICES
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in self.gpu_ids)

        # If we have valid GPUs, pick the first one as primary; otherwise CPU
        if self.gpu_ids:
            self.device = torch.device(f"cuda:{self.gpu_ids[0]}")
        else:
            self.device = torch.device("cpu")

        # Flag for multi-GPU
        self.multi_gpu = len(self.gpu_ids) > 1

        print(f"[DeviceManager] Using device: {self.device} (Multi-GPU: {self.multi_gpu}, GPUs: {self.gpu_ids})")

    def setup_dataparallel(self, model):
        """
        Wrap model with DataParallel if multiple GPUs are available 
        Returns - nn.Module: Possibly wrapped in DataParallel.
        """
        # Check if model is already wrapped in DataParallel
        if isinstance(model, nn.DataParallel):
            return model
            
        # Move model to the primary device first
        model.to(self.device)

        if self.multi_gpu:
            # Store device_ids_list as an attribute for later use
            self.device_ids_list = self.gpu_ids
            print(f"[DeviceManager] DataParallel activated on GPUs {self.gpu_ids}")
            model = nn.DataParallel(model, device_ids=self.gpu_ids)
        
        return model

    # Utility function to handle state_dict from DataParallel models
    def fix_state_dict_for_loading(state_dict, model):
        """
        Fixes state dictionary keys for loading into a model (handles DataParallel prefix) 
        Returns - ct: Fixed state dictionary
        """
        new_dict = {}
        
        # Check if we're loading a DataParallel dict into a non-DataParallel model
        loading_dp_into_normal = False
        if any(k.startswith('module.') for k in state_dict.keys()) and not isinstance(model, nn.DataParallel):
            loading_dp_into_normal = True
            
        # Check if we're loading a normal dict into a DataParallel model
        loading_normal_into_dp = False  
        if not any(k.startswith('module.') for k in state_dict.keys()) and isinstance(model, nn.DataParallel):
            loading_normal_into_dp = True
        
        # Fix keys accordingly
        for k, v in state_dict.items():
            if loading_dp_into_normal:
                # Remove 'module.' prefix for DataParallel state dict
                new_k = k[7:] if k.startswith('module.') else k
                new_dict[new_k] = v
            elif loading_normal_into_dp:
                # Add 'module.' prefix for normal state dict
                new_k = 'module.' + k
                new_dict[new_k] = v
            else:
                # No changes needed
                new_dict[k] = v
                
        return new_dict

    '''
    def setup_dataparallel(self, model):
        """
        Wrap model with DataParallel if multiple GPUs are available       
        Returns - nn.Module: Possibly wrapped in DataParallel.
        """
        # Move model to the primary device first
        model.to(self.device)

        if self.multi_gpu:
            print(f"[DeviceManager] DataParallel activated on GPUs {self.gpu_ids}")
            model = nn.DataParallel(model, device_ids=self.gpu_ids)

        return model
    '''

    def print_device_info(self):
        """Prints detailed information about the available computing devices."""
        print("\n=== Device Information ===")
        print(f"Primary Device: {self.device}")
        print(f"Multiple GPUs: {'Yes' if self.multi_gpu else 'No'}")
        print(f"Total GPUs: {self.total_gpus}")
        print(f"Using GPUs: {self.gpu_ids}")

        if self.total_gpus > 0:
            print("\n=== GPU Memory Status ===")
            for i in self.gpu_ids:
                props = torch.cuda.get_device_properties(i)
                total_mem = props.total_memory / (1024**3)
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                free = total_mem - allocated

                print(f"  GPU {i}: {props.name} | Free: {free:.2f} GB / {total_mem:.2f} GB total")
                print(f"    Reserved: {reserved:.2f} GB | Allocated: {allocated:.2f} GB")
                print(f"    Compute Capability: {props.major}.{props.minor}")

    def enable_memory_optimizations(self):
        """Enable memory optimizations for CUDA operations."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.backends.cuda.matmul.allow_tf32 = True
            if hasattr(torch.backends.cudnn, 'allow_tf32'):
                torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
            print("[DeviceManager] Memory optimizations enabled.")

def move_to_device(model, device):
    """
    Moves all model parameters and buffers to the specified device 
    Returns - nn.Module: Model on the correct device.
    """
    model.to(device)
    for param in model.parameters():
        if param.device != device:
            param.data = param.data.to(device)
    for buffer in model.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)
    return model

def verify_device_placement(model, device, verbose=True):
    """
    Verifies that all model parameters and buffers are on the specified device 
    Returns - bool: True if all components are on the correct device, otherwise raises an error 
    """
    mismatches = []
    for name, param in model.named_parameters():
        if param.device != device:
            mismatches.append(f"Param {name} on {param.device}, expected {device}")

    for name, buf in model.named_buffers():
        if buf.device != device:
            mismatches.append(f"Buffer {name} on {buf.device}, expected {device}")

    if mismatches:
        if verbose:
            print("[verify_device_placement] Device mismatch found:")
            for msg in mismatches[:10]:
                print("  -", msg)
            if len(mismatches) > 10:
                print(f"  ... and {len(mismatches) - 10} more.")
        raise RuntimeError("\n".join(mismatches))

    if verbose:
        print(f"[verify_device_placement] All parameters and buffers are on {device}")
    return True
