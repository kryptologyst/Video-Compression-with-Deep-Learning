# Video Compression with Deep Learning

A production-ready implementation of deep learning-based video compression using autoencoders, temporal modeling, and learned compression techniques.

## Overview

This project implements advanced video compression algorithms using neural networks, including:

- **Video Autoencoder**: Basic frame-by-frame compression using convolutional autoencoders
- **Temporal Video Autoencoder**: Compression with temporal consistency modeling
- **Hyperprior Video Compressor**: Learned compression with hyperprior modeling for rate-distortion optimization

## Features

- **Multiple Model Architectures**: From basic autoencoders to advanced learned compression
- **Comprehensive Metrics**: PSNR, SSIM, compression ratio, bits per pixel (bpp)
- **Modern Stack**: PyTorch 2.x, Hydra configuration, TensorBoard logging
- **Production Ready**: Type hints, comprehensive testing, CI/CD pipeline
- **Interactive Demo**: Streamlit-based web interface for real-time compression
- **Evaluation Tools**: Automated evaluation with visualization and reporting

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Video-Compression-with-Deep-Learning.git
cd Video-Compression-with-Deep-Learning

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e ".[dev]"
```

### Basic Usage

1. **Create sample data** (if you don't have video files):
```bash
python -c "from src.data.video_dataset import create_sample_video; create_sample_video('data/raw/sample.mp4')"
```

2. **Train a model**:
```bash
python scripts/train.py --config configs/train.yaml
```

3. **Evaluate the model**:
```bash
python scripts/eval.py --config configs/eval.yaml
```

4. **Run the demo**:
```bash
python scripts/demo.py
```

## Project Structure

```
video-compression-dl/
├── src/                    # Source code
│   ├── models/            # Model implementations
│   │   ├── video_compressor.py
│   │   └── losses.py
│   ├── data/              # Data loading and preprocessing
│   │   └── video_dataset.py
│   ├── utils/             # Utility functions
│   │   ├── device.py
│   │   └── metrics.py
│   ├── train/             # Training scripts
│   │   └── trainer.py
│   └── eval/              # Evaluation scripts
│       └── evaluator.py
├── configs/               # Configuration files
│   ├── train.yaml
│   └── eval.yaml
├── scripts/               # Executable scripts
│   ├── train.py
│   ├── eval.py
│   └── demo.py
├── demo/                  # Demo interface
│   └── streamlit_app.py
├── tests/                 # Unit tests
│   └── test_models.py
├── data/                  # Data directory
│   ├── raw/              # Raw video files
│   └── processed/        # Processed data
├── assets/               # Generated assets
├── checkpoints/          # Model checkpoints
├── logs/                 # Training logs
└── requirements.txt      # Dependencies
```

## Model Architectures

### 1. Video Autoencoder
Basic convolutional autoencoder for frame-by-frame compression:
- Encoder: Progressive downsampling with residual blocks
- Decoder: Progressive upsampling with skip connections
- Latent space: Configurable compression ratio

### 2. Temporal Video Autoencoder
Extension with temporal modeling:
- 3D convolutions for temporal consistency
- Frame interpolation capabilities
- Reduced temporal artifacts

### 3. Hyperprior Video Compressor
Advanced learned compression:
- Hyperprior modeling for rate-distortion optimization
- Entropy coding simulation
- BD-Rate evaluation support

## Configuration

The project uses Hydra for configuration management. Key configuration files:

- `configs/train.yaml`: Training configuration
- `configs/eval.yaml`: Evaluation configuration

Example configuration:
```yaml
model:
  type: "video_autoencoder"
  latent_channels: 64
  num_residual_blocks: 4

training:
  batch_size: 8
  learning_rate: 0.001
  num_epochs: 100

data:
  frame_size: [224, 224]
  sequence_length: 8
```

## Training

### Basic Training
```bash
python scripts/train.py --config configs/train.yaml
```

### Resume Training
```bash
python scripts/train.py --config configs/train.yaml --resume checkpoints/latest.pth
```

### Training with Custom Configuration
```bash
python scripts/train.py --config configs/train.yaml model.latent_channels=128 training.batch_size=16
```

## Evaluation

### Basic Evaluation
```bash
python scripts/eval.py --config configs/eval.yaml
```

### Evaluation with Custom Checkpoint
```bash
python scripts/eval.py --config configs/eval.yaml --checkpoint checkpoints/best.pth
```

### Metrics
The evaluation provides comprehensive metrics:
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index (0-1)
- **Compression Ratio**: Original size / Compressed size
- **BPP**: Bits per pixel
- **BD-Rate**: Bjontegaard Delta Rate for rate-distortion comparison

## Demo Interface

The Streamlit demo provides an interactive interface for:
- Video upload and processing
- Real-time compression visualization
- Metric calculation and display
- Side-by-side comparison
- Model parameter adjustment

```bash
python scripts/demo.py
```

Access the demo at: http://localhost:8501

## Dataset Schema

### Input Format
- **Video files**: MP4, AVI, MOV, MKV
- **Frame size**: Configurable (default: 224x224)
- **Sequence length**: Number of consecutive frames (default: 8)
- **Channels**: RGB (3 channels)

### Data Organization
```
data/raw/
├── video1.mp4
├── video2.mp4
└── ...
```

### Sample Data Generation
If no video data is available, the system automatically generates sample videos for testing:
```python
from src.data.video_dataset import create_sample_video
create_sample_video('data/raw/sample.mp4', duration=2.0, fps=30)
```

## Performance Metrics

### Compression Efficiency
- **Compression Ratio**: 2-10x typical compression
- **PSNR**: 25-35 dB for good quality
- **SSIM**: 0.8-0.95 for perceptual quality

### Computational Performance
- **Training Time**: ~2-4 hours on GPU for 100 epochs
- **Inference Speed**: ~10-50 FPS depending on model complexity
- **Memory Usage**: 2-8 GB VRAM for training

### Model Sizes
- **Video Autoencoder**: ~50-200 MB
- **Temporal Autoencoder**: ~100-400 MB
- **Hyperprior Compressor**: ~200-800 MB

## Development

### Code Quality
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google/NumPy style docstrings
- **Formatting**: Black + Ruff for code formatting
- **Testing**: Comprehensive unit tests with pytest

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ tests/
ruff check src/ tests/
```

### Pre-commit Hooks
```bash
pre-commit install
pre-commit run --all-files
```

## Advanced Features

### Mixed Precision Training
Enable automatic mixed precision for faster training:
```yaml
training:
  use_amp: true
```

### Multi-GPU Training
Support for distributed training:
```bash
python -m torch.distributed.launch --nproc_per_node=2 scripts/train.py
```

### Custom Loss Functions
Implement custom loss functions by extending the base classes:
```python
from src.models.losses import CompressionLoss

class CustomLoss(CompressionLoss):
    def forward(self, x_recon, x_target, rate_loss):
        # Custom loss implementation
        pass
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in configuration
   - Use gradient accumulation
   - Enable gradient checkpointing

2. **Video Loading Errors**
   - Ensure video files are in supported formats
   - Check file permissions
   - Verify OpenCV installation

3. **Model Loading Issues**
   - Check checkpoint file paths
   - Verify model architecture matches checkpoint
   - Ensure PyTorch version compatibility

### Performance Optimization

1. **Training Speed**
   - Use mixed precision training
   - Increase batch size if memory allows
   - Use multiple workers for data loading

2. **Inference Speed**
   - Use model quantization
   - Enable TensorRT optimization
   - Reduce input resolution

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{video_compression_dl,
  title={Video Compression with Deep Learning},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/Video-Compression-with-Deep-Learning}
}
```

## Acknowledgments

- PyTorch team for the excellent deep learning framework
- OpenCV for video processing capabilities
- Streamlit for the demo interface
- The computer vision research community for foundational work

## Future Work

- [ ] Integration with actual entropy coding
- [ ] Support for more video formats
- [ ] Real-time compression pipeline
- [ ] Mobile optimization
- [ ] Advanced temporal modeling techniques
- [ ] Integration with existing codecs (H.264, H.265)
# Video-Compression-with-Deep-Learning
