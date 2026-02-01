"""Tests for video compression models."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.video_compressor import VideoAutoencoder, TemporalVideoAutoencoder, HyperpriorVideoCompressor
from src.models.losses import CompressionLoss, PerceptualLoss, RateLoss, TemporalConsistencyLoss
from src.utils.device import get_device, count_parameters, get_model_size_mb
from src.utils.metrics import calculate_psnr, calculate_ssim, evaluate_compression_metrics


class TestVideoCompressor:
    """Test video compression models."""
    
    def test_video_autoencoder(self):
        """Test VideoAutoencoder model."""
        model = VideoAutoencoder()
        device = get_device()
        model.to(device)
        
        # Test input
        x = torch.randn(2, 3, 224, 224).to(device)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == x.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0
        
        # Check parameters
        assert count_parameters(model) > 0
        assert get_model_size_mb(model) > 0
    
    def test_temporal_video_autoencoder(self):
        """Test TemporalVideoAutoencoder model."""
        model = TemporalVideoAutoencoder()
        device = get_device()
        model.to(device)
        
        # Test input (video with temporal dimension)
        x = torch.randn(2, 3, 8, 224, 224).to(device)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == x.shape
        assert output.min() >= 0.0
        assert output.max() <= 1.0
    
    def test_hyperprior_video_compressor(self):
        """Test HyperpriorVideoCompressor model."""
        model = HyperpriorVideoCompressor()
        device = get_device()
        model.to(device)
        
        # Test input
        x = torch.randn(2, 3, 224, 224).to(device)
        
        # Forward pass
        outputs = model(x)
        
        # Check output keys
        expected_keys = ['y', 'z', 'mu', 'sigma', 'x_recon']
        assert all(key in outputs for key in expected_keys)
        
        # Check reconstruction shape
        assert outputs['x_recon'].shape == x.shape
        assert outputs['x_recon'].min() >= 0.0
        assert outputs['x_recon'].max() <= 1.0


class TestLosses:
    """Test loss functions."""
    
    def test_compression_loss(self):
        """Test CompressionLoss."""
        loss_fn = CompressionLoss()
        device = get_device()
        
        # Test inputs
        x_recon = torch.randn(2, 3, 224, 224).to(device)
        x_target = torch.randn(2, 3, 224, 224).to(device)
        rate_loss = torch.tensor(0.1).to(device)
        
        # Forward pass
        loss_dict = loss_fn(x_recon, x_target, rate_loss)
        
        # Check loss components
        assert 'total_loss' in loss_dict
        assert 'mse_loss' in loss_dict
        assert 'perceptual_loss' in loss_dict
        assert 'rate_loss' in loss_dict
        
        # Check loss values
        assert loss_dict['total_loss'] > 0
        assert loss_dict['mse_loss'] > 0
        assert loss_dict['perceptual_loss'] > 0
        assert loss_dict['rate_loss'] == rate_loss
    
    def test_rate_loss(self):
        """Test RateLoss."""
        loss_fn = RateLoss()
        device = get_device()
        
        # Test inputs
        y = torch.randn(2, 64, 14, 14).to(device)
        z = torch.randn(2, 32, 7, 7).to(device)
        
        # Forward pass
        loss = loss_fn(y, z)
        
        # Check loss value
        assert loss > 0
        assert isinstance(loss, torch.Tensor)
    
    def test_temporal_consistency_loss(self):
        """Test TemporalConsistencyLoss."""
        loss_fn = TemporalConsistencyLoss()
        device = get_device()
        
        # Test input (video)
        x = torch.randn(2, 3, 8, 224, 224).to(device)
        
        # Forward pass
        loss = loss_fn(x)
        
        # Check loss value
        assert loss >= 0
        assert isinstance(loss, torch.Tensor)


class TestMetrics:
    """Test evaluation metrics."""
    
    def test_psnr_calculation(self):
        """Test PSNR calculation."""
        device = get_device()
        
        # Test identical images (should give infinite PSNR)
        img1 = torch.randn(1, 3, 224, 224).to(device)
        img2 = img1.clone()
        
        psnr = calculate_psnr(img1, img2)
        assert psnr == float('inf')
        
        # Test different images
        img2 = torch.randn(1, 3, 224, 224).to(device)
        psnr = calculate_psnr(img1, img2)
        assert psnr > 0
        assert psnr < float('inf')
    
    def test_ssim_calculation(self):
        """Test SSIM calculation."""
        device = get_device()
        
        # Test identical images (should give SSIM = 1)
        img1 = torch.randn(1, 3, 224, 224).to(device)
        img2 = img1.clone()
        
        ssim = calculate_ssim(img1, img2)
        assert abs(ssim - 1.0) < 1e-6
        
        # Test different images
        img2 = torch.randn(1, 3, 224, 224).to(device)
        ssim = calculate_ssim(img1, img2)
        assert 0 <= ssim <= 1
    
    def test_compression_metrics(self):
        """Test compression metrics evaluation."""
        device = get_device()
        
        # Test inputs
        original = torch.randn(1, 3, 224, 224).to(device)
        reconstructed = torch.randn(1, 3, 224, 224).to(device)
        original_size = 1000000  # 1MB
        compressed_size = 100000  # 100KB
        
        # Calculate metrics
        metrics = evaluate_compression_metrics(
            original, reconstructed, original_size, compressed_size
        )
        
        # Check metric keys
        expected_keys = ['psnr', 'ssim', 'compression_ratio', 'bpp', 'original_size_mb', 'compressed_size_mb']
        assert all(key in metrics for key in expected_keys)
        
        # Check metric values
        assert metrics['psnr'] > 0
        assert 0 <= metrics['ssim'] <= 1
        assert metrics['compression_ratio'] > 0
        assert metrics['bpp'] > 0
        assert metrics['original_size_mb'] > 0
        assert metrics['compressed_size_mb'] > 0


class TestDeviceUtils:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda', 'mps']
    
    def test_count_parameters(self):
        """Test parameter counting."""
        model = VideoAutoencoder()
        param_count = count_parameters(model)
        assert param_count > 0
        assert isinstance(param_count, int)
    
    def test_get_model_size(self):
        """Test model size calculation."""
        model = VideoAutoencoder()
        size_mb = get_model_size_mb(model)
        assert size_mb > 0
        assert isinstance(size_mb, float)


if __name__ == "__main__":
    pytest.main([__file__])
