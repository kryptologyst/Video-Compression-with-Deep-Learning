"""Advanced video compression models."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual block with skip connection."""
    
    def __init__(self, in_channels: int, out_channels: int):
        """Initialize residual block.
        
        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through residual block."""
        residual = self.skip(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        return F.relu(out + residual)


class VideoAutoencoder(nn.Module):
    """Advanced video autoencoder for compression."""
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 64,
        num_residual_blocks: int = 4
    ):
        """Initialize video autoencoder.
        
        Args:
            input_channels: Number of input channels (RGB=3).
            latent_channels: Number of channels in latent space.
            num_residual_blocks: Number of residual blocks in encoder/decoder.
        """
        super().__init__()
        self.latent_channels = latent_channels
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),  # 224x224 -> 112x112
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 112x112 -> 56x56
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),  # 56x56 -> 28x28
            nn.ReLU(inplace=True),
            nn.Conv2d(256, latent_channels, 4, stride=2, padding=1),  # 28x28 -> 14x14
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks in latent space
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(latent_channels, latent_channels)
            for _ in range(num_residual_blocks)
        ])
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, 4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),  # 28x28 -> 56x56
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.Sigmoid(),  # Output in [0, 1]
        )
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to latent representation.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Latent representation of shape (B, latent_channels, H', W').
        """
        z = self.encoder(x)
        
        # Apply residual blocks
        for block in self.residual_blocks:
            z = block(z)
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to output.
        
        Args:
            z: Latent tensor of shape (B, latent_channels, H', W').
            
        Returns:
            Reconstructed tensor of shape (B, C, H, W).
        """
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through autoencoder.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Reconstructed tensor of shape (B, C, H, W).
        """
        z = self.encode(x)
        return self.decode(z)


class TemporalVideoAutoencoder(VideoAutoencoder):
    """Video autoencoder with temporal modeling."""
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 64,
        num_residual_blocks: int = 4,
        temporal_kernel_size: int = 3
    ):
        """Initialize temporal video autoencoder.
        
        Args:
            input_channels: Number of input channels (RGB=3).
            latent_channels: Number of channels in latent space.
            num_residual_blocks: Number of residual blocks in encoder/decoder.
            temporal_kernel_size: Kernel size for temporal convolutions.
        """
        super().__init__(input_channels, latent_channels, num_residual_blocks)
        
        # Temporal modeling layers
        self.temporal_encoder = nn.Conv3d(
            latent_channels, latent_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            padding=(temporal_kernel_size // 2, 0, 0)
        )
        
        self.temporal_decoder = nn.Conv3d(
            latent_channels, latent_channels,
            kernel_size=(temporal_kernel_size, 1, 1),
            padding=(temporal_kernel_size // 2, 0, 0)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with temporal modeling.
        
        Args:
            x: Input tensor of shape (B, C, T, H, W).
            
        Returns:
            Reconstructed tensor of shape (B, C, T, H, W).
        """
        B, C, T, H, W = x.shape
        
        # Process each frame through encoder
        encoded_frames = []
        for t in range(T):
            frame_encoded = self.encode(x[:, :, t, :, :])
            encoded_frames.append(frame_encoded)
        
        # Stack encoded frames
        z = torch.stack(encoded_frames, dim=2)  # (B, latent_channels, T, H', W')
        
        # Apply temporal modeling
        z_temporal = self.temporal_encoder(z)
        z_temporal = F.relu(z_temporal)
        z_temporal = self.temporal_decoder(z_temporal)
        z_temporal = F.relu(z_temporal)
        
        # Decode each frame
        decoded_frames = []
        for t in range(T):
            frame_decoded = self.decode(z_temporal[:, :, t, :, :])
            decoded_frames.append(frame_decoded)
        
        # Stack decoded frames
        output = torch.stack(decoded_frames, dim=2)  # (B, C, T, H, W)
        
        return output


class HyperpriorVideoCompressor(nn.Module):
    """Video compressor with hyperprior for learned compression."""
    
    def __init__(
        self,
        input_channels: int = 3,
        latent_channels: int = 64,
        hyperprior_channels: int = 32
    ):
        """Initialize hyperprior video compressor.
        
        Args:
            input_channels: Number of input channels (RGB=3).
            latent_channels: Number of channels in latent space.
            hyperprior_channels: Number of channels in hyperprior.
        """
        super().__init__()
        
        # Main encoder/decoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, latent_channels, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, input_channels, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )
        
        # Hyperprior encoder/decoder
        self.hyper_encoder = nn.Sequential(
            nn.Conv2d(latent_channels, hyperprior_channels, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyperprior_channels, hyperprior_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        
        self.hyper_decoder = nn.Sequential(
            nn.ConvTranspose2d(hyperprior_channels, hyperprior_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hyperprior_channels, latent_channels * 2, 3, stride=1, padding=1),
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through hyperprior compressor.
        
        Args:
            x: Input tensor of shape (B, C, H, W).
            
        Returns:
            Dictionary containing compressed representation and reconstruction.
        """
        # Encode to latent representation
        y = self.encoder(x)
        
        # Encode hyperprior
        z = self.hyper_encoder(y)
        
        # Decode hyperprior to get parameters for latent distribution
        hyper_params = self.hyper_decoder(z)
        mu, sigma = torch.chunk(hyper_params, 2, dim=1)
        sigma = torch.exp(sigma)  # Ensure positive
        
        # Decode main representation
        x_recon = self.decoder(y)
        
        return {
            'y': y,  # Main latent representation
            'z': z,  # Hyperprior representation
            'mu': mu,  # Mean of latent distribution
            'sigma': sigma,  # Standard deviation of latent distribution
            'x_recon': x_recon,  # Reconstructed image
        }
