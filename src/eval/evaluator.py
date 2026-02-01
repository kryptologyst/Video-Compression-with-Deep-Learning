"""Evaluation script for video compression models."""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm import tqdm

from src.models.video_compressor import VideoAutoencoder, TemporalVideoAutoencoder, HyperpriorVideoCompressor
from src.data.video_dataset import VideoCompressionDataset
from src.utils.device import get_device, set_seed
from src.utils.metrics import evaluate_compression_metrics


class VideoCompressionEvaluator:
    """Evaluator class for video compression models."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize evaluator.
        
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
        
        # Load checkpoint
        self._load_checkpoint()
        
        # Initialize data loader
        self.data_loader = self._create_data_loader()
        
        # Create output directory
        self.output_dir = Path(config['evaluation']['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
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
    
    def _load_checkpoint(self) -> None:
        """Load model checkpoint."""
        checkpoint_path = self.config['model']['checkpoint_path']
        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"Loaded checkpoint from: {checkpoint_path}")
    
    def _create_data_loader(self) -> DataLoader:
        """Create data loader for evaluation."""
        data_dir = Path(self.config['data']['data_dir'])
        video_paths = list(data_dir.glob('*.mp4'))
        
        if not video_paths:
            raise FileNotFoundError(f"No video files found in: {data_dir}")
        
        dataset = VideoCompressionDataset(
            video_paths,
            frame_size=self.config['data']['frame_size'],
            sequence_length=self.config['data'].get('sequence_length', 8),
            max_frames=self.config['data'].get('max_frames', None)
        )
        
        return DataLoader(
            dataset,
            batch_size=self.config['evaluation']['batch_size'],
            shuffle=False,
            num_workers=self.config['data'].get('num_workers', 4),
            pin_memory=True
        )
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate the model on the dataset."""
        self.logger.info("Starting evaluation...")
        
        all_metrics = []
        
        with torch.no_grad():
            for batch_idx, (input_frames, target_frames) in enumerate(tqdm(self.data_loader, desc="Evaluating")):
                input_frames = input_frames.to(self.device)
                target_frames = target_frames.to(self.device)
                
                # Forward pass
                if self.config['model']['type'] == 'hyperprior':
                    outputs = self.model(input_frames)
                    reconstructed = outputs['x_recon']
                else:
                    reconstructed = self.model(input_frames)
                
                # Calculate metrics
                metrics = evaluate_compression_metrics(
                    target_frames,
                    reconstructed,
                    target_frames.numel() * 4,  # Approximate original size
                    reconstructed.numel() * 4     # Approximate compressed size
                )
                
                all_metrics.append(metrics)
                
                # Save reconstructions if requested
                if self.config['evaluation'].get('save_reconstructions', False):
                    self._save_reconstructions(
                        input_frames, reconstructed, target_frames, batch_idx
                    )
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        
        # Log results
        self.logger.info("Evaluation Results:")
        for key, value in avg_metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
        
        # Save results
        self._save_results(avg_metrics)
        
        return avg_metrics
    
    def _save_reconstructions(self, input_frames: torch.Tensor, reconstructed: torch.Tensor, 
                            target_frames: torch.Tensor, batch_idx: int) -> None:
        """Save reconstruction visualizations."""
        # Convert tensors to numpy
        input_np = input_frames[0].cpu().numpy()  # Take first sample
        recon_np = reconstructed[0].cpu().numpy()
        target_np = target_frames[0].cpu().numpy()
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original frames (first frame)
        axes[0].imshow(input_np[:, 0, :, :].transpose(1, 2, 0))
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        # Reconstructed frames (first frame)
        axes[1].imshow(recon_np[:, 0, :, :].transpose(1, 2, 0))
        axes[1].set_title('Reconstructed')
        axes[1].axis('off')
        
        # Difference
        diff = np.abs(input_np[:, 0, :, :] - recon_np[:, 0, :, :])
        axes[2].imshow(diff.transpose(1, 2, 0))
        axes[2].set_title('Difference')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'reconstruction_{batch_idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def _save_results(self, metrics: Dict[str, float]) -> None:
        """Save evaluation results to file."""
        results_file = self.output_dir / 'evaluation_results.txt'
        
        with open(results_file, 'w') as f:
            f.write("Video Compression Evaluation Results\n")
            f.write("=" * 40 + "\n\n")
            
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
        
        self.logger.info(f"Results saved to: {results_file}")
    
    def create_compression_curve(self, metrics_list: List[Dict[str, float]]) -> None:
        """Create compression curve visualization."""
        if len(metrics_list) < 2:
            self.logger.warning("Need at least 2 models for compression curve")
            return
        
        psnr_values = [m['psnr'] for m in metrics_list]
        bpp_values = [m['bpp'] for m in metrics_list]
        
        plt.figure(figsize=(10, 6))
        plt.plot(bpp_values, psnr_values, 'o-', linewidth=2, markersize=8)
        plt.xlabel('Bits per Pixel (bpp)')
        plt.ylabel('PSNR (dB)')
        plt.title('Rate-Distortion Curve')
        plt.grid(True, alpha=0.3)
        
        # Add model labels
        model_names = [f"Model {i+1}" for i in range(len(metrics_list))]
        for i, (bpp, psnr) in enumerate(zip(bpp_values, psnr_values)):
            plt.annotate(model_names[i], (bpp, psnr), xytext=(5, 5), 
                        textcoords='offset points')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'compression_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info("Compression curve saved")


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate video compression model')
    parser.add_argument('--config', type=str, default='configs/eval.yaml',
                       help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (overrides config)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Override checkpoint path if specified
    if args.checkpoint:
        config.model.checkpoint_path = args.checkpoint
    
    # Create evaluator
    evaluator = VideoCompressionEvaluator(config)
    
    # Run evaluation
    metrics = evaluator.evaluate()
    
    print("\nEvaluation completed successfully!")


if __name__ == '__main__':
    main()
