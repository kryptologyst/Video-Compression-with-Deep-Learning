"""Video data loading and preprocessing utilities."""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VideoDataset(Dataset):
    """Dataset class for video compression tasks."""
    
    def __init__(
        self,
        video_paths: List[Union[str, Path]],
        frame_size: Tuple[int, int] = (224, 224),
        max_frames: Optional[int] = None,
        transform: Optional[transforms.Compose] = None
    ):
        """Initialize video dataset.
        
        Args:
            video_paths: List of paths to video files.
            frame_size: Target frame size (height, width).
            max_frames: Maximum number of frames to load per video.
            transform: Optional transforms to apply to frames.
        """
        self.video_paths = [Path(p) for p in video_paths]
        self.frame_size = frame_size
        self.max_frames = max_frames
        self.transform = transform or self._default_transform()
        
        # Validate video paths
        self.valid_paths = []
        for path in self.video_paths:
            if path.exists() and path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                self.valid_paths.append(path)
            else:
                print(f"Warning: Skipping invalid video path: {path}")
    
    def _default_transform(self) -> transforms.Compose:
        """Create default transforms for video frames."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.frame_size),
            transforms.ToTensor(),
        ])
    
    def __len__(self) -> int:
        """Return number of valid videos."""
        return len(self.valid_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and return video frames.
        
        Args:
            idx: Index of the video to load.
            
        Returns:
            torch.Tensor: Video frames of shape (C, T, H, W).
        """
        video_path = self.valid_paths[idx]
        frames = self._load_video_frames(video_path)
        
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Stack frames into tensor
        video_tensor = torch.stack(frames, dim=1)  # (C, T, H, W)
        return video_tensor
    
    def _load_video_frames(self, video_path: Path) -> List[np.ndarray]:
        """Load frames from video file.
        
        Args:
            video_path: Path to video file.
            
        Returns:
            List of frames as numpy arrays.
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if self.max_frames and frame_count >= self.max_frames:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
        
        cap.release()
        return frames


class VideoCompressionDataset(VideoDataset):
    """Specialized dataset for video compression with temporal consistency."""
    
    def __init__(
        self,
        video_paths: List[Union[str, Path]],
        frame_size: Tuple[int, int] = (224, 224),
        sequence_length: int = 8,
        overlap: int = 4,
        **kwargs
    ):
        """Initialize video compression dataset.
        
        Args:
            video_paths: List of paths to video files.
            frame_size: Target frame size (height, width).
            sequence_length: Number of consecutive frames per sample.
            overlap: Number of overlapping frames between sequences.
            **kwargs: Additional arguments for VideoDataset.
        """
        super().__init__(video_paths, frame_size, **kwargs)
        self.sequence_length = sequence_length
        self.overlap = overlap
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load video sequence for compression.
        
        Args:
            idx: Index of the video to load.
            
        Returns:
            Tuple of (input_frames, target_frames) for compression training.
        """
        video_path = self.valid_paths[idx]
        frames = self._load_video_frames(video_path)
        
        if len(frames) < self.sequence_length:
            # Pad with last frame if video is too short
            frames.extend([frames[-1]] * (self.sequence_length - len(frames)))
        
        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        # Convert to tensor
        video_tensor = torch.stack(frames, dim=1)  # (C, T, H, W)
        
        # For compression, input and target are the same (autoencoder)
        return video_tensor, video_tensor


def create_sample_video(
    output_path: Union[str, Path],
    duration: float = 2.0,
    fps: int = 30,
    size: Tuple[int, int] = (224, 224)
) -> None:
    """Create a sample video for testing purposes.
    
    Args:
        output_path: Path to save the sample video.
        duration: Duration of video in seconds.
        fps: Frames per second.
        size: Video size (width, height).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    
    num_frames = int(duration * fps)
    
    for i in range(num_frames):
        # Create a simple animated pattern
        frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        
        # Moving circle
        center_x = int(size[0] * (0.5 + 0.3 * np.sin(2 * np.pi * i / num_frames)))
        center_y = int(size[1] * (0.5 + 0.3 * np.cos(2 * np.pi * i / num_frames)))
        cv2.circle(frame, (center_x, center_y), 30, (255, 255, 255), -1)
        
        # Color gradient
        color_intensity = int(255 * (0.5 + 0.5 * np.sin(2 * np.pi * i / num_frames)))
        frame[:, :, 0] = color_intensity
        frame[:, :, 1] = 255 - color_intensity
        
        out.write(frame)
    
    out.release()


def load_video_frames(
    video_path: Union[str, Path],
    max_frames: Optional[int] = None,
    frame_size: Optional[Tuple[int, int]] = None
) -> List[np.ndarray]:
    """Load frames from a video file.
    
    Args:
        video_path: Path to video file.
        max_frames: Maximum number of frames to load.
        frame_size: Optional frame size to resize to (height, width).
        
    Returns:
        List of frames as numpy arrays.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames and frame_count >= max_frames:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize if specified
        if frame_size:
            frame_rgb = cv2.resize(frame_rgb, (frame_size[1], frame_size[0]))
        
        frames.append(frame_rgb)
        frame_count += 1
    
    cap.release()
    return frames
