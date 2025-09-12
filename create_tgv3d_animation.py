#!/usr/bin/env python3
"""
Create an animation from the TGV3D frame sequence.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image

def create_animation():
    """Create an animated GIF from the frame sequence."""
    frames_dir = "outputs/tgv3d/frames"
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    
    if len(frame_files) == 0:
        print("No frame files found!")
        return
    
    print(f"Found {len(frame_files)} frames")
    
    # Load all frames
    frames = []
    for frame_file in frame_files:
        img = Image.open(frame_file)
        frames.append(np.array(img))
    
    # Create animation
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    def animate(frame):
        ax.clear()
        ax.axis('off')
        ax.imshow(frames[frame])
        # Extract time from filename
        time_str = os.path.basename(frame_files[frame]).split('_')[1].split('.')[0]
        time_val = float(time_str)
        ax.set_title(f'3D Taylor-Green Vortex Evolution (GPU/MPS)\nTime: {time_val:.3f}', 
                    fontsize=14, fontweight='bold')
    
    anim = FuncAnimation(fig, animate, frames=len(frames), interval=100, repeat=True)
    
    # Save as GIF
    output_path = "outputs/tgv3d/tgv3d_animation.gif"
    anim.save(output_path, writer=PillowWriter(fps=10))
    print(f"Animation saved to: {output_path}")
    
    plt.close()

if __name__ == "__main__":
    create_animation()
