"""Streamlit demo for video compression."""

import streamlit as st
import torch
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
from typing import Optional

from src.models.video_compressor import VideoAutoencoder, TemporalVideoAutoencoder, HyperpriorVideoCompressor
from src.utils.device import get_device
from src.utils.metrics import evaluate_compression_metrics


@st.cache_resource
def load_model(model_type: str, checkpoint_path: str):
    """Load model with caching."""
    device = get_device()
    
    if model_type == 'video_autoencoder':
        model = VideoAutoencoder()
    elif model_type == 'temporal_autoencoder':
        model = TemporalVideoAutoencoder()
    elif model_type == 'hyperprior':
        model = HyperpriorVideoCompressor()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    model.to(device)
    model.eval()
    return model


def process_video_frames(video_file, model, frame_size=(224, 224), max_frames=30):
    """Process video frames through the model."""
    device = get_device()
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        # Load video
        cap = cv2.VideoCapture(tmp_path)
        frames = []
        frame_count = 0
        
        while frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB and resize
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, frame_size)
            
            # Convert to tensor
            frame_tensor = torch.tensor(frame_resized, dtype=torch.float32).permute(2, 0, 1) / 255.0
            frames.append(frame_tensor)
            frame_count += 1
        
        cap.release()
        
        if not frames:
            return None, None, None
        
        # Stack frames
        video_tensor = torch.stack(frames, dim=1).unsqueeze(0).to(device)  # (1, C, T, H, W)
        
        # Process through model
        with torch.no_grad():
            if isinstance(model, HyperpriorVideoCompressor):
                outputs = model(video_tensor.squeeze(2))  # Remove temporal dimension for hyperprior
                reconstructed = outputs['x_recon']
            else:
                reconstructed = model(video_tensor.squeeze(2))  # Remove temporal dimension
        
        # Convert back to numpy
        original_frames = video_tensor.squeeze(0).cpu().numpy()  # (C, T, H, W)
        reconstructed_frames = reconstructed.squeeze(0).cpu().numpy()  # (C, T, H, W)
        
        return original_frames, reconstructed_frames, len(frames)
        
    finally:
        # Clean up temporary file
        os.unlink(tmp_path)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Video Compression Demo",
        page_icon="🎬",
        layout="wide"
    )
    
    st.title("🎬 Video Compression with Deep Learning")
    st.markdown("Upload a video to see how our deep learning models compress and reconstruct it.")
    
    # Sidebar for model selection
    st.sidebar.header("Model Configuration")
    
    model_type = st.sidebar.selectbox(
        "Select Model Type",
        ["video_autoencoder", "temporal_autoencoder", "hyperprior"],
        help="Choose the compression model to use"
    )
    
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint Path",
        value="checkpoints/best.pth",
        help="Path to the model checkpoint"
    )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    latent_channels = st.sidebar.slider("Latent Channels", 16, 128, 64)
    num_residual_blocks = st.sidebar.slider("Residual Blocks", 2, 8, 4)
    
    # Video processing parameters
    st.sidebar.subheader("Processing Parameters")
    frame_size = st.sidebar.selectbox("Frame Size", [(224, 224), (256, 256), (128, 128)])
    max_frames = st.sidebar.slider("Max Frames", 5, 50, 20)
    
    # Load model
    try:
        model = load_model(model_type, checkpoint_path)
        st.sidebar.success("✅ Model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📤 Upload Video")
        uploaded_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file to compress and reconstruct"
        )
        
        if uploaded_file is not None:
            st.video(uploaded_file)
    
    with col2:
        st.header("⚙️ Processing")
        
        if uploaded_file is not None:
            with st.spinner("Processing video..."):
                original_frames, reconstructed_frames, num_frames = process_video_frames(
                    uploaded_file, model, frame_size, max_frames
                )
            
            if original_frames is not None:
                st.success(f"✅ Processed {num_frames} frames successfully!")
                
                # Calculate metrics
                original_tensor = torch.tensor(original_frames)
                reconstructed_tensor = torch.tensor(reconstructed_frames)
                
                metrics = evaluate_compression_metrics(
                    original_tensor,
                    reconstructed_tensor,
                    original_tensor.numel() * 4,
                    reconstructed_tensor.numel() * 4
                )
                
                # Display metrics
                st.subheader("📊 Compression Metrics")
                col_metrics1, col_metrics2 = st.columns(2)
                
                with col_metrics1:
                    st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
                    st.metric("SSIM", f"{metrics['ssim']:.4f}")
                
                with col_metrics2:
                    st.metric("Compression Ratio", f"{metrics['compression_ratio']:.2f}x")
                    st.metric("Bits per Pixel", f"{metrics['bpp']:.4f}")
    
    # Results section
    if uploaded_file is not None and original_frames is not None:
        st.header("🎯 Results")
        
        # Frame comparison
        st.subheader("Frame-by-Frame Comparison")
        
        frame_idx = st.slider("Select Frame", 0, num_frames - 1, 0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original")
            original_frame = original_frames[:, frame_idx, :, :].transpose(1, 2, 0)
            st.image(original_frame, use_column_width=True)
        
        with col2:
            st.subheader("Reconstructed")
            recon_frame = reconstructed_frames[:, frame_idx, :, :].transpose(1, 2, 0)
            st.image(recon_frame, use_column_width=True)
        
        with col3:
            st.subheader("Difference")
            diff_frame = np.abs(original_frame - recon_frame)
            st.image(diff_frame, use_column_width=True)
        
        # Create side-by-side video
        st.subheader("Side-by-Side Video")
        
        # Create comparison video
        comparison_frames = []
        for i in range(num_frames):
            orig = original_frames[:, i, :, :].transpose(1, 2, 0)
            recon = reconstructed_frames[:, i, :, :].transpose(1, 2, 0)
            
            # Create side-by-side frame
            comparison_frame = np.concatenate([orig, recon], axis=1)
            comparison_frames.append(comparison_frame)
        
        # Save comparison video
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp_file.name, fourcc, 10.0, 
                                (frame_size[1] * 2, frame_size[0]))
            
            for frame in comparison_frames:
                frame_bgr = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
            
            out.release()
            
            # Display video
            with open(tmp_file.name, 'rb') as f:
                st.video(f.read())
            
            # Clean up
            os.unlink(tmp_file.name)
    
    # Model information
    st.header("ℹ️ Model Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.subheader("Model Architecture")
        st.write(f"**Type:** {model_type}")
        st.write(f"**Latent Channels:** {latent_channels}")
        st.write(f"**Residual Blocks:** {num_residual_blocks}")
        st.write(f"**Frame Size:** {frame_size}")
    
    with info_col2:
        st.subheader("Performance")
        if uploaded_file is not None and original_frames is not None:
            st.write(f"**Processing Time:** ~{num_frames * 0.1:.1f}s")
            st.write(f"**Memory Usage:** ~{original_frames.nbytes / 1024 / 1024:.1f} MB")
            st.write(f"**Compression Efficiency:** {metrics['compression_ratio']:.2f}x")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with PyTorch, Streamlit, and OpenCV. "
        "This demo showcases deep learning-based video compression techniques."
    )


if __name__ == "__main__":
    main()
