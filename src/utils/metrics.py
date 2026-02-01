"""Video compression metrics and evaluation utilities."""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim


def calculate_psnr(img1: torch.Tensor, img2: torch.Tensor, max_val: float = 1.0) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        img1: First image tensor.
        img2: Second image tensor.
        max_val: Maximum possible pixel value.
        
    Returns:
        float: PSNR value in dB.
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(max_val / torch.sqrt(mse)).item()


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """Calculate Structural Similarity Index (SSIM).
    
    Args:
        img1: First image tensor.
        img2: Second image tensor.
        
    Returns:
        float: SSIM value between 0 and 1.
    """
    # Convert to numpy for skimage
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    # Handle batch dimension
    if len(img1_np.shape) == 4:  # Batch dimension
        ssim_values = []
        for i in range(img1_np.shape[0]):
            ssim_val = ssim(
                img1_np[i].transpose(1, 2, 0),
                img2_np[i].transpose(1, 2, 0),
                multichannel=True,
                channel_axis=-1
            )
            ssim_values.append(ssim_val)
        return np.mean(ssim_values)
    else:
        return ssim(
            img1_np.transpose(1, 2, 0),
            img2_np.transpose(1, 2, 0),
            multichannel=True,
            channel_axis=-1
        )


def calculate_compression_ratio(original_size: int, compressed_size: int) -> float:
    """Calculate compression ratio.
    
    Args:
        original_size: Size of original data in bytes.
        compressed_size: Size of compressed data in bytes.
        
    Returns:
        float: Compression ratio.
    """
    return original_size / compressed_size if compressed_size > 0 else float('inf')


def calculate_bpp(compressed_size: int, num_pixels: int) -> float:
    """Calculate bits per pixel (bpp).
    
    Args:
        compressed_size: Size of compressed data in bits.
        num_pixels: Number of pixels in the image/video.
        
    Returns:
        float: Bits per pixel.
    """
    return compressed_size / num_pixels


def calculate_bd_rate(
    psnr_values: List[float],
    bpp_values: List[float],
    reference_psnr: List[float],
    reference_bpp: List[float]
) -> float:
    """Calculate BD-Rate (Bjontegaard Delta Rate).
    
    Args:
        psnr_values: PSNR values for the method.
        bpp_values: BPP values for the method.
        reference_psnr: Reference PSNR values.
        reference_bpp: Reference BPP values.
        
    Returns:
        float: BD-Rate percentage.
    """
    # Simple linear interpolation for BD-Rate calculation
    # In practice, you'd want to use proper polynomial fitting
    def interpolate_rate(psnr_target: float, psnr_vals: List[float], bpp_vals: List[float]) -> float:
        if psnr_target <= min(psnr_vals):
            return bpp_vals[psnr_vals.index(min(psnr_vals))]
        elif psnr_target >= max(psnr_vals):
            return bpp_vals[psnr_vals.index(max(psnr_vals))]
        else:
            # Linear interpolation
            for i in range(len(psnr_vals) - 1):
                if psnr_vals[i] <= psnr_target <= psnr_vals[i + 1]:
                    t = (psnr_target - psnr_vals[i]) / (psnr_vals[i + 1] - psnr_vals[i])
                    return bpp_vals[i] + t * (bpp_vals[i + 1] - bpp_vals[i])
        return 0.0
    
    # Calculate average rate difference
    psnr_range = np.linspace(min(min(psnr_values), min(reference_psnr)),
                            max(max(psnr_values), max(reference_psnr)), 100)
    
    rate_diff = 0.0
    for psnr_val in psnr_range:
        rate_method = interpolate_rate(psnr_val, psnr_values, bpp_values)
        rate_ref = interpolate_rate(psnr_val, reference_psnr, reference_bpp)
        if rate_ref > 0:
            rate_diff += (rate_method - rate_ref) / rate_ref
    
    return (rate_diff / len(psnr_range)) * 100


def evaluate_compression_metrics(
    original: torch.Tensor,
    reconstructed: torch.Tensor,
    original_size: int,
    compressed_size: int
) -> Dict[str, float]:
    """Evaluate comprehensive compression metrics.
    
    Args:
        original: Original image/video tensor.
        reconstructed: Reconstructed image/video tensor.
        original_size: Size of original data in bytes.
        compressed_size: Size of compressed data in bytes.
        
    Returns:
        Dict containing all compression metrics.
    """
    psnr = calculate_psnr(original, reconstructed)
    ssim_val = calculate_ssim(original, reconstructed)
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)
    
    num_pixels = original.numel()
    bpp = calculate_bpp(compressed_size * 8, num_pixels)  # Convert bytes to bits
    
    return {
        'psnr': psnr,
        'ssim': ssim_val,
        'compression_ratio': compression_ratio,
        'bpp': bpp,
        'original_size_mb': original_size / (1024 * 1024),
        'compressed_size_mb': compressed_size / (1024 * 1024)
    }
