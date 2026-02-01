"""Training script for video compression models."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.video_compressor import VideoAutoencoder, TemporalVideoAutoencoder, HyperpriorVideoCompressor
from src.models.losses import CompressionLoss, RateLoss, TemporalConsistencyLoss
from src.data.video_dataset import VideoCompressionDataset, create_sample_video
from src.utils.device import get_device, set_seed, count_parameters, get_model_size_mb
from src.utils.metrics import evaluate_compression_metrics


class VideoCompressionTrainer:
    """Trainer class for video compression models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary.
        """
        self.config = config
        self.device = get_device()
        
        # Set random seed
        set_seed(config.get('seed', 42))
        
        # Setup logging
        self._setup_logging()
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Initialize loss function
        self.criterion = self._create_loss_function()
        
        # Initialize optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
        
        # Initialize scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=config['training'].get('patience', 10),
            verbose=True
        )
        
        # Initialize data loaders
        self.train_loader, self.val_loader = self._create_data_loaders()
        
        # Initialize logging
        self.writer = SummaryWriter(config['logging']['log_dir'])
        
        # Initialize wandb if enabled
        if config['logging'].get('use_wandb', False):
            wandb.init(
                project=config['logging'].get('project_name', 'video-compression'),
                config=config
            )
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
    
    def _create_model(self) -> nn.Module:
        """Create model based on configuration."""
        model_type = self.config['model']['type']
        
        if model_type == 'video_autoencoder':
            return VideoAutoencoder(
                input_channels=self.config['model'].get('input_channels', 3),
                latent_channels=self.config['model'].get('latent_channels', 64),
                num_residual_blocks=self.config['model'].get('num_residual_blocks', 4)
            )
        elif model_type == 'temporal_autoencoder':
            return TemporalVideoAutoencoder(
                input_channels=self.config['model'].get('input_channels', 3),
                latent_channels=self.config['model'].get('latent_channels', 64),
                num_residual_blocks=self.config['model'].get('num_residual_blocks', 4),
                temporal_kernel_size=self.config['model'].get('temporal_kernel_size', 3)
            )
        elif model_type == 'hyperprior':
            return HyperpriorVideoCompressor(
                input_channels=self.config['model'].get('input_channels', 3),
                latent_channels=self.config['model'].get('latent_channels', 64),
                hyperprior_channels=self.config['model'].get('hyperprior_channels', 32)
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _create_loss_function(self) -> nn.Module:
        """Create loss function based on configuration."""
        loss_config = self.config['training']['loss']
        
        if loss_config['type'] == 'compression':
            return CompressionLoss(
                lambda_mse=loss_config.get('lambda_mse', 1.0),
                lambda_perceptual=loss_config.get('lambda_perceptual', 0.1),
                lambda_rate=loss_config.get('lambda_rate', 0.01)
            )
        else:
            return nn.MSELoss()
    
    def _create_data_loaders(self) -> tuple[DataLoader, DataLoader]:
        """Create data loaders for training and validation."""
        # Create sample videos if no data exists
        data_dir = Path(self.config['data']['data_dir'])
        if not data_dir.exists() or not list(data_dir.glob('*.mp4')):
            self.logger.info("Creating sample videos for training...")
            data_dir.mkdir(parents=True, exist_ok=True)
            for i in range(10):  # Create 10 sample videos
                create_sample_video(
                    data_dir / f"sample_{i:03d}.mp4",
                    duration=2.0,
                    fps=30,
                    size=(224, 224)
                )
        
        # Create datasets
        video_paths = list(data_dir.glob('*.mp4'))
        train_size = int(0.8 * len(video_paths))
        
        train_dataset = VideoCompressionDataset(
            video_paths[:train_size],
            frame_size=self.config['data']['frame_size'],
            sequence_length=self.config['data'].get('sequence_length', 8),
            max_frames=self.config['data'].get('max_frames', None)
        )
        
        val_dataset = VideoCompressionDataset(
            video_paths[train_size:],
            frame_size=self.config['data']['frame_size'],
            sequence_length=self.config['data'].get('sequence_length', 8),
            max_frames=self.config['data'].get('max_frames', None)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['data'].get('num_workers', 4),
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 4),
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, (input_frames, target_frames) in enumerate(pbar):
            input_frames = input_frames.to(self.device)
            target_frames = target_frames.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            if self.config['model']['type'] == 'hyperprior':
                outputs = self.model(input_frames)
                reconstructed = outputs['x_recon']
                
                # Calculate rate loss
                rate_loss_fn = RateLoss()
                rate_loss = rate_loss_fn(outputs['y'], outputs['z'])
                
                # Calculate compression loss
                loss_dict = self.criterion(reconstructed, target_frames, rate_loss)
                loss = loss_dict['total_loss']
            else:
                reconstructed = self.model(input_frames)
                loss = self.criterion(reconstructed, target_frames)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Log to tensorboard
            if batch_idx % 100 == 0:
                self.writer.add_scalar(
                    'Train/Loss',
                    loss.item(),
                    epoch * num_batches + batch_idx
                )
        
        return {'train_loss': total_loss / num_batches}
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for input_frames, target_frames in tqdm(self.val_loader, desc="Validation"):
                input_frames = input_frames.to(self.device)
                target_frames = target_frames.to(self.device)
                
                # Forward pass
                if self.config['model']['type'] == 'hyperprior':
                    outputs = self.model(input_frames)
                    reconstructed = outputs['x_recon']
                    
                    # Calculate rate loss
                    rate_loss_fn = RateLoss()
                    rate_loss = rate_loss_fn(outputs['y'], outputs['z'])
                    
                    # Calculate compression loss
                    loss_dict = self.criterion(reconstructed, target_frames, rate_loss)
                    loss = loss_dict['total_loss']
                else:
                    reconstructed = self.model(input_frames)
                    loss = self.criterion(reconstructed, target_frames)
                
                total_loss += loss.item()
                
                # Calculate metrics
                metrics = evaluate_compression_metrics(
                    target_frames,
                    reconstructed,
                    target_frames.numel() * 4,  # Approximate original size
                    reconstructed.numel() * 4   # Approximate compressed size
                )
                
                total_psnr += metrics['psnr']
                total_ssim += metrics['ssim']
        
        avg_loss = total_loss / num_batches
        avg_psnr = total_psnr / num_batches
        avg_ssim = total_ssim / num_batches
        
        # Log metrics
        self.writer.add_scalar('Val/Loss', avg_loss, epoch)
        self.writer.add_scalar('Val/PSNR', avg_psnr, epoch)
        self.writer.add_scalar('Val/SSIM', avg_ssim, epoch)
        
        if self.config['logging'].get('use_wandb', False):
            wandb.log({
                'val_loss': avg_loss,
                'val_psnr': avg_psnr,
                'val_ssim': avg_ssim,
                'epoch': epoch
            })
        
        return {
            'val_loss': avg_loss,
            'val_psnr': avg_psnr,
            'val_ssim': avg_ssim
        }
    
    def train(self) -> None:
        """Main training loop."""
        self.logger.info(f"Starting training with {count_parameters(self.model)} parameters")
        self.logger.info(f"Model size: {get_model_size_mb(self.model):.2f} MB")
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config['training']['num_epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update scheduler
            self.scheduler.step(val_metrics['val_loss'])
            
            # Log metrics
            self.logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val PSNR: {val_metrics['val_psnr']:.2f}, "
                f"Val SSIM: {val_metrics['val_ssim']:.4f}"
            )
            
            # Save best model
            if val_metrics['val_loss'] < best_val_loss:
                best_val_loss = val_metrics['val_loss']
                self.save_checkpoint(epoch, is_best=True)
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= self.config['training'].get('early_stopping_patience', 20):
                self.logger.info("Early stopping triggered")
                break
        
        self.logger.info("Training completed")
        self.writer.close()
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        checkpoint_dir = Path(self.config['logging']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save latest checkpoint
        torch.save(checkpoint, checkpoint_dir / 'latest.pth')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, checkpoint_dir / 'best.pth')


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train video compression model')
    parser.add_argument('--config', type=str, default='configs/train.yaml',
                       help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Create trainer
    trainer = VideoCompressionTrainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        checkpoint = torch.load(args.resume)
        trainer.model.load_state_dict(checkpoint['model_state_dict'])
        trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        trainer.logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Start training
    trainer.train()


if __name__ == '__main__':
    main()
