"""
param_stats.py - Utilities for model parameter counting and statistics (c) itrustal.com
"""
import torch
import numpy as np


def count_parameters(model):
    """
    Count trainable and total parameters in the model 
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    return trainable_params, total_params


def format_number(num):
    """
    Format large numbers for readability
    Returns - str: Formatted number string (e.g., "10.5M")
    """
    if num >= 1e9:
        return f"{num/1e9:.2f}B"
    elif num >= 1e6:
        return f"{num/1e6:.2f}M"
    elif num >= 1e3:
        return f"{num/1e3:.2f}K"
    return f"{num:.2f}"


def calculate_memory_usage(model):
    """
    Calculate estimated memory usage for model parameters and buffers
    Returns - tuple: (param_size_mb, buffer_size_mb, total_size_mb)
    """
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1024**2
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers()) / 1024**2
    total_size = param_size + buffer_size
    
    return param_size, buffer_size, total_size


def calculate_model_statistics(model, input_shape=(3, 512, 512), device=None):
    """
    Calculate and return comprehensive model statistics 
    Returns - dict: Dictionary of model statistics
    """
    try:
        from ptflops import get_model_complexity_info
        from thop import profile
        has_complexity_tools = True
    except ImportError:
        print("Note: ptflops and/or thop not installed; FLOP estimation will be limited")
        print("Tip: Install with 'pip install ptflops thop' for enhanced complexity metrics")
        has_complexity_tools = False
    
    # Use model's device if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Count parameters
    trainable_params, total_params = count_parameters(model)
    
    # Calculate memory usage
    param_size, buffer_size, total_static_memory = calculate_memory_usage(model)
    
    # Initialize stats dictionary
    stats = {
        'trainable_parameters': trainable_params,
        'total_parameters': total_params,
        'parameter_memory_mb': param_size,
        'buffer_memory_mb': buffer_size,
        'total_static_memory_mb': total_static_memory,
        'input_shape': input_shape
    }
    
    # Calculate FLOPs if complexity tools are available
    if has_complexity_tools:
        try:
            # Prepare for complexity calculation
            input_tensor = torch.randn(1, *input_shape).to(device)
            
            # Get model for profiling (handle DataParallel)
            model_to_profile = model.module if hasattr(model, 'module') else model
            
            # Using ptflops
            macs, params = get_model_complexity_info(
                model_to_profile, input_shape, as_strings=False, 
                print_per_layer_stat=False, verbose=False
            )
            stats['gflops_ptflops'] = macs * 2 / 1e9
            
            # Using thop
            macs_thop, params_thop = profile(model_to_profile, inputs=(input_tensor,), verbose=False)
            stats['gflops_thop'] = macs_thop * 2 / 1e9
        except Exception as e:
            print(f"Warning: Error calculating FLOPs: {e}")
            stats['gflops_estimated'] = (trainable_params * 2) / 1e9  # Rough estimate
    else:
        # Rough FLOP estimation if tools not available
        stats['gflops_estimated'] = (trainable_params * 2) / 1e9
    
    return stats


def print_model_summary(model, input_shape=(3, 512, 512)):
    """
    Print comprehensive model summary with statistics 
    Returns - dict: Dictionary of model statistics
    """
    # Calculate stats
    stats = calculate_model_statistics(model, input_shape)
    
    # Print summary
    print("\n" + "="*50)
    print("MODEL SUMMARY")
    print("="*50)
    
    print("\nParameters:")
    print(f"- Trainable parameters: {format_number(stats['trainable_parameters'])}")
    print(f"- Total parameters: {format_number(stats['total_parameters'])}")
    
    print("\nMemory:")
    print(f"- Parameters memory: {stats['parameter_memory_mb']:.2f} MB")
    print(f"- Buffers memory: {stats['buffer_memory_mb']:.2f} MB")
    print(f"- Total static memory: {stats['total_static_memory_mb']:.2f} MB")
    
    print("\nComputation:")
    if 'gflops_ptflops' in stats:
        print(f"- GFLOPs (ptflops): {stats['gflops_ptflops']:.2f}")
    if 'gflops_thop' in stats:
        print(f"- GFLOPs (thop): {stats['gflops_thop']:.2f}")
    if 'gflops_estimated' in stats:
        print(f"- GFLOPs (estimated): {stats['gflops_estimated']:.2f}")
    
    if hasattr(model, 'seg_model'):
        print("\nModel Structure:")
        print(f"- Base segmentation model: {model.seg_model.__class__.__name__}")
        if hasattr(model.seg_model, 'encoder'):
            print(f"- Encoder: {model.seg_model.encoder.__class__.__name__}")
        if hasattr(model.seg_model, 'decoder'):
            print(f"- Decoder: {model.seg_model.decoder.__class__.__name__}")
    
    print("\nInput/Output:")
    print(f"- Input shape: {input_shape}")
    print(f"- Output size: {getattr(model, 'output_size', 'N/A')}")
    
    # Try to use torchsummary if available
    try:
        from torchsummary import summary
        device = next(model.parameters()).device
        print("\nDetailed Layer Structure:")
        summary(model, input_size=input_shape, device=str(device).split(':')[0])
    except ImportError:
        print("\nNote: torchsummary not installed; install with 'pip install torchsummary' for detailed layer structure")
    except Exception as e:
        print(f"\nError generating detailed summary: {e}")
    
    print("="*50)
    
    return stats
