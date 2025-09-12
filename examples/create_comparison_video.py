#!/usr/bin/env python3
"""
Create comparison videos from the generated comparison frames.
"""

import os
import subprocess
import sys


def create_video_from_frames(frames_dir, output_video, fps=10):
    """Create a video from comparison frames using ffmpeg."""
    
    if not os.path.exists(frames_dir):
        print(f"Frames directory {frames_dir} does not exist")
        return False
    
    # Check if there are any frames
    frames = [f for f in os.listdir(frames_dir) if f.endswith('.png')]
    if not frames:
        print(f"No PNG frames found in {frames_dir}")
        return False
    
    print(f"Found {len(frames)} frames in {frames_dir}")
    
    # Try different codecs
    codecs = ["libopenh264", "h264_videotoolbox", "mpeg4", "libvpx-vp9"]
    
    for codec in codecs:
        cmd = [
            "ffmpeg", "-y", "-framerate", str(fps),
            "-i", os.path.join(frames_dir, "comparison_frame_%05d.png"),
            "-c:v", codec, "-crf", "18", "-pix_fmt", "yuv420p",
            output_video
        ]
        
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            if result.returncode == 0:
                print(f"Successfully created video: {output_video} using codec {codec}")
                return True
            else:
                print(f"Codec {codec} failed: {result.stderr.decode()}")
        except FileNotFoundError:
            print(f"ffmpeg not found")
            return False
    
    print("All codecs failed")
    return False


def main():
    """Create videos for all comparison variables."""
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    comparison_dir = os.path.join(base_dir, 'outputs', 'khi2d', 'comparison_frames')
    
    variables = ["density", "momentum_x", "momentum_y", "energy", "pressure"]
    
    for var_name in variables:
        var_dir = os.path.join(comparison_dir, var_name)
        output_video = os.path.join(comparison_dir, f"khi_comparison_{var_name}.mp4")
        
        print(f"\nCreating video for {var_name}...")
        if create_video_from_frames(var_dir, output_video):
            print(f"Video saved: {output_video}")
        else:
            print(f"Failed to create video for {var_name}")


if __name__ == "__main__":
    main()
