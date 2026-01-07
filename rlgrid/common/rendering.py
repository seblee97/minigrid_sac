from __future__ import annotations

import os
import numpy as np
import gymnasium as gym
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2

def render_static_env_image(env: gym.Env, save_path: str, title: Optional[str] = None) -> None:
    """
    Render a static image of the environment at its current state and save it.
    
    Args:
        env: The environment to render
        save_path: Path where to save the image
        title: Optional title for the image
    """
    try:
        # Reset to initial state for consistent rendering
        obs, _ = env.reset()
        
        # For MiniGrid environments, we need to use rgb_array render mode
        if hasattr(env.unwrapped, 'render'):
            # Try different render modes for MiniGrid
            try:
                rgb_array = env.unwrapped.render(mode='rgb_array')
            except:
                try:
                    rgb_array = env.render(mode='rgb_array')
                except:
                    rgb_array = env.render()
            
            if rgb_array is not None and isinstance(rgb_array, np.ndarray):
                plt.figure(figsize=(8, 8))
                plt.imshow(rgb_array)
                plt.axis('off')
                if title:
                    plt.title(title)
                plt.tight_layout()
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close()
                return
        
        # Fallback: use observation if it's an image
        if isinstance(obs, dict) and 'image' in obs:
            img = obs['image']
        else:
            img = obs
            
        if isinstance(img, np.ndarray) and len(img.shape) == 3:
            plt.figure(figsize=(8, 8))
            plt.imshow(img)
            plt.axis('off')
            if title:
                plt.title(title)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
            
    except Exception as e:
        print(f"Warning: Could not render environment image: {e}")


class EpisodeVideoRecorder:
    """
    Records video of episodes during training.
    """
    
    def __init__(self, env: gym.Env, save_dir: str, fps: int = 30):
        self.env = env
        self.save_dir = save_dir
        self.fps = fps
        self.frames: List[np.ndarray] = []
        self.recording = False
        
        # Try to set up the environment for RGB rendering
        if hasattr(env.unwrapped, 'render_mode'):
            try:
                env.unwrapped.render_mode = 'rgb_array'
            except:
                pass
        
        os.makedirs(save_dir, exist_ok=True)
    
    def start_recording(self) -> None:
        """Start recording a new episode."""
        self.frames = []
        self.recording = True
    
    def capture_frame(self) -> None:
        """Capture the current frame if recording."""
        if not self.recording:
            return
            
        try:
            # Try to get RGB rendering from MiniGrid
            frame = None
            
            # Method 1: Try unwrapped render with rgb_array mode
            if hasattr(self.env, 'unwrapped') and hasattr(self.env.unwrapped, 'render'):
                try:
                    frame = self.env.unwrapped.render(mode='rgb_array')
                except:
                    try:
                        frame = self.env.unwrapped.render()
                    except:
                        pass
            
            # Method 2: Try wrapped env render
            if frame is None and hasattr(self.env, 'render'):
                try:
                    frame = self.env.render(mode='rgb_array')
                except:
                    try:
                        frame = self.env.render()
                    except:
                        pass
            
            # Method 3: Use observation as fallback
            if frame is None:
                try:
                    if hasattr(self.env.unwrapped, 'gen_obs'):
                        obs = self.env.unwrapped.gen_obs()
                    else:
                        # Get last observation if available
                        obs = getattr(self.env, '_last_obs', None)
                    
                    if obs is not None:
                        if isinstance(obs, dict) and 'image' in obs:
                            frame = obs['image']
                        else:
                            frame = obs
                except:
                    return
            
            if frame is not None and isinstance(frame, np.ndarray):
                # Ensure frame is in correct format (H, W, C) with values 0-255
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                    
                # Ensure 3 channels for video
                if len(frame.shape) == 3:
                    if frame.shape[2] > 3:
                        frame = frame[:, :, :3]
                    elif frame.shape[2] == 1:
                        frame = np.repeat(frame, 3, axis=2)
                elif len(frame.shape) == 2:
                    frame = np.stack([frame] * 3, axis=-1)
                
                # Make sure frame is not empty/black
                if frame.max() > 0:
                    self.frames.append(frame)
                
        except Exception as e:
            print(f"Warning: Could not capture frame: {e}")
    
    def stop_recording(self, filename: str) -> bool:
        """Stop recording and save the video."""
        if not self.recording or not self.frames:
            self.recording = False
            return False
        
        self.recording = False
        
        try:
            video_path = os.path.join(self.save_dir, f"{filename}.mp4")
            
            # Get frame dimensions
            h, w = self.frames[0].shape[:2]
            
            # Create video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(video_path, fourcc, self.fps, (w, h))
            
            for frame in self.frames:
                # Convert RGB to BGR for OpenCV
                bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(bgr_frame)
            
            out.release()
            self.frames = []
            return True
            
        except Exception as e:
            print(f"Warning: Could not save video: {e}")
            return False


def should_record_video(step: int, video_freq: int, last_video_step: int = 0) -> bool:
    """
    Determine if we should record a video at this step using threshold crossing.
    
    Args:
        step: Current training step
        video_freq: Frequency of video recording in steps
        last_video_step: Step when last video was recorded
        
    Returns:
        True if a video should be recorded
    """
    if video_freq <= 0:
        return False
    
    # Record at regular intervals
    return step - last_video_step >= video_freq