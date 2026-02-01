"""Loss functions for video compression."""

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class CompressionLoss(nn.Module):
    """Combined loss function for video compression."""
    
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_perceptual: float = 0.1,
        lambda_rate: float = 0.01
    ):
        """Initialize compression loss.
        
        Args:
            lambda_mse: Weight for MSE reconstruction loss.
            lambda_perceptual: Weight for perceptual loss.
            lambda_rate: Weight for rate loss.
        """
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_perceptual = lambda_perceptual
        self.lambda_rate = lambda_rate
        
        self.mse_loss = nn.MSELoss()
        self.perceptual_loss = PerceptualLoss()
    
    def forward(
        self,
        x_recon: torch.Tensor,
        x_target: torch.Tensor,
        rate_loss: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Calculate compression loss.
        
        Args:
            x_recon: Reconstructed image/video.
            x_target: Target image/video.
            rate_loss: Rate loss from entropy coding.
            
        Returns:
            Dictionary containing individual loss components and total loss.
        """
        # Reconstruction loss
        mse_loss = self.mse_loss(x_recon, x_target)
        
        # Perceptual loss
        perceptual_loss = self.perceptual_loss(x_recon, x_target)
        
        # Total loss
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_perceptual * perceptual_loss +
            self.lambda_rate * rate_loss
        )
        
        return {
            'total_loss': total_loss,
            'mse_loss': mse_loss,
            'perceptual_loss': perceptual_loss,
            'rate_loss': rate_loss
        }


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""
    
    def __init__(self, feature_layers: list = None):
        """Initialize perceptual loss.
        
        Args:
            feature_layers: List of layer indices to use for perceptual loss.
        """
        super().__init__()
        
        if feature_layers is None:
            feature_layers = [4, 9, 18, 27]  # VGG16 feature layers
        
        self.feature_layers = feature_layers
        
        # Load pretrained VGG16
        vgg = torchvision.models.vgg16(pretrained=True)
        self.features = vgg.features
        
        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Extract feature layers
        self.feature_extractors = nn.ModuleList()
        for i in range(max(feature_layers) + 1):
            if i in feature_layers:
                self.feature_extractors.append(self.features[i])
            else:
                self.feature_extractors.append(nn.Identity())
    
    def forward(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Calculate perceptual loss.
        
        Args:
            x_recon: Reconstructed image.
            x_target: Target image.
            
        Returns:
            Perceptual loss value.
        """
        # Ensure images are in [0, 1] range
        x_recon = torch.clamp(x_recon, 0, 1)
        x_target = torch.clamp(x_target, 0, 1)
        
        # Normalize to ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(x_recon.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(x_recon.device)
        
        x_recon_norm = (x_recon - mean) / std
        x_target_norm = (x_target - mean) / std
        
        # Extract features
        recon_features = []
        target_features = []
        
        for layer in self.feature_extractors:
            x_recon_norm = layer(x_recon_norm)
            x_target_norm = layer(x_target_norm)
            
            if isinstance(layer, nn.Conv2d):
                recon_features.append(x_recon_norm)
                target_features.append(x_target_norm)
        
        # Calculate perceptual loss
        perceptual_loss = 0.0
        for recon_feat, target_feat in zip(recon_features, target_features):
            perceptual_loss += F.mse_loss(recon_feat, target_feat)
        
        return perceptual_loss / len(recon_features)


class RateLoss(nn.Module):
    """Rate loss for entropy coding."""
    
    def __init__(self):
        """Initialize rate loss."""
        super().__init__()
    
    def forward(self, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Calculate rate loss.
        
        Args:
            y: Main latent representation.
            z: Hyperprior representation.
            
        Returns:
            Rate loss value.
        """
        # Simplified rate loss (in practice, you'd use actual entropy coding)
        # This is a proxy for the actual bit rate
        rate_y = torch.mean(torch.abs(y))
        rate_z = torch.mean(torch.abs(z))
        
        return rate_y + rate_z


class TemporalConsistencyLoss(nn.Module):
    """Temporal consistency loss for video compression."""
    
    def __init__(self, alpha: float = 1.0):
        """Initialize temporal consistency loss.
        
        Args:
            alpha: Weight for temporal consistency.
        """
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x_recon: torch.Tensor) -> torch.Tensor:
        """Calculate temporal consistency loss.
        
        Args:
            x_recon: Reconstructed video tensor of shape (B, C, T, H, W).
            
        Returns:
            Temporal consistency loss value.
        """
        if x_recon.dim() != 5:
            return torch.tensor(0.0, device=x_recon.device)
        
        B, C, T, H, W = x_recon.shape
        
        # Calculate frame differences
        frame_diff = 0.0
        for t in range(T - 1):
            diff = F.mse_loss(x_recon[:, :, t, :, :], x_recon[:, :, t + 1, :, :])
            frame_diff += diff
        
        return self.alpha * frame_diff / (T - 1)


class CharbonnierLoss(nn.Module):
    """Charbonnier loss (robust L1 loss)."""
    
    def __init__(self, eps: float = 1e-6):
        """Initialize Charbonnier loss.
        
        Args:
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
    
    def forward(self, x_recon: torch.Tensor, x_target: torch.Tensor) -> torch.Tensor:
        """Calculate Charbonnier loss.
        
        Args:
            x_recon: Reconstructed image/video.
            x_target: Target image/video.
            
        Returns:
            Charbonnier loss value.
        """
        diff = x_recon - x_target
        return torch.mean(torch.sqrt(diff * diff + self.eps))
