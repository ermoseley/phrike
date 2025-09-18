#!/usr/bin/env python3
"""
Create video from frames using Python (alternative to ffmpeg).
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image

def create_video_from_frames_python(frame_dir="sod_video_frames", output_file="sod_shock_tube_python.mp4", fps=30):
    """Create video from frames using Python matplotlib animation."""
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return False
    
    print(f"Found {len(frame_files)} frames")
    
    # Read first frame to get dimensions
    first_frame = Image.open(frame_files[0])
    width, height = first_frame.size
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Create image object
    im = ax.imshow(np.zeros((height, width, 3)), aspect='auto')
    
    def animate(frame_num):
        """Animation function."""
        if frame_num < len(frame_files):
            # Load and display frame
            img = Image.open(frame_files[frame_num])
            img_array = np.array(img)
            im.set_array(img_array)
            ax.set_title(f'Frame {frame_num+1}/{len(frame_files)}')
        return [im]
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(frame_files), 
                                 interval=1000/fps, blit=True, repeat=True)
    
    # Save as video
    print(f"Creating video: {output_file}")
    try:
        anim.save(output_file, writer='pillow', fps=fps, bitrate=1800)
        print(f"Video created successfully: {output_file}")
        return True
    except Exception as e:
        print(f"Error creating video: {e}")
        return False

def create_gif_from_frames(frame_dir="sod_video_frames", output_file="sod_shock_tube.gif", fps=30):
    """Create GIF from frames using PIL."""
    
    # Get all frame files
    frame_files = sorted(glob.glob(os.path.join(frame_dir, "frame_*.png")))
    
    if not frame_files:
        print(f"No frames found in {frame_dir}")
        return False
    
    print(f"Found {len(frame_files)} frames")
    
    # Load frames
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        frames.append(img)
    
    # Save as GIF
    print(f"Creating GIF: {output_file}")
    try:
        frames[0].save(output_file, 
                      save_all=True, 
                      append_images=frames[1:], 
                      duration=1000//fps,  # Duration in milliseconds
                      loop=0)  # Infinite loop
        print(f"GIF created successfully: {output_file}")
        return True
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return False

if __name__ == "__main__":
    # Try to create both video and GIF
    print("Creating video using Python...")
    success1 = create_video_from_frames_python()
    
    print("\nCreating GIF...")
    success2 = create_gif_from_frames()
    
    if success1 or success2:
        print("\nVideo/GIF creation completed!")
    else:
        print("\nVideo/GIF creation failed!")
