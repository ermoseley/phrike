"""Base problem class for Hydra simulations."""

import os
import subprocess
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from hydra.io import load_config, ensure_outdir, save_solution_snapshot


class BaseProblem(ABC):
    """Base class for all Hydra problems."""
    
    @staticmethod
    def clear_numba_cache():
        """Clear numba cache to prevent module name conflicts.
        
        This is primarily useful as a one-time fix after package renames.
        Use the --clear-cache CLI flag when needed.
        """
        try:
            # Clear user's numba cache
            numba_cache = Path.home() / ".numba_cache"
            if numba_cache.exists():
                shutil.rmtree(numba_cache)
            
            # Clear local __pycache__ directories
            current_dir = Path(__file__).parent.parent
            for pycache in current_dir.rglob("__pycache__"):
                if pycache.is_dir():
                    shutil.rmtree(pycache)
                    
        except Exception:
            # Silently fail if cache clearing doesn't work
            pass
    
    def __init__(self, config_path: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """Initialize problem with configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (takes precedence over config_path)
        """
        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")
        
        self.setup_output_directory()
        self.setup_common_parameters()
    
    def setup_output_directory(self) -> None:
        """Setup output directory from configuration."""
        if "io" in self.config and "outdir" in self.config["io"]:
            self.outdir = os.path.abspath(self.config["io"]["outdir"])
        else:
            # Default to problem name in outputs/
            problem_name = self.config.get("problem", "unknown")
            self.outdir = os.path.abspath(f"outputs/{problem_name}")
        
        ensure_outdir(self.outdir)
    
    def setup_common_parameters(self) -> None:
        """Extract common parameters from configuration."""
        self.gamma = float(self.config["physics"]["gamma"])
        
        # Integration parameters
        self.t0 = float(self.config["integration"]["t0"])
        self.t_end = float(self.config["integration"]["t_end"])
        self.cfl = float(self.config["integration"]["cfl"])
        self.scheme = str(self.config["integration"].get("scheme", "rk4"))
        self.output_interval = float(self.config["integration"].get("output_interval", 0.1))
        self.checkpoint_interval = float(self.config["integration"].get("checkpoint_interval", 0.0))
        self.filter_config = self.config["integration"].get("spectral_filter", {"enabled": False})
        
        # FFT workers
        self.fft_workers = int(self.config["grid"].get("fft_workers", os.cpu_count() or 1))
    
    @abstractmethod
    def create_grid(self, backend: str = "numpy", device: Optional[str] = None):
        """Create the computational grid."""
        pass
    
    @abstractmethod
    def create_equations(self):
        """Create the equation system."""
        pass
    
    @abstractmethod
    def create_initial_conditions(self, grid):
        """Create initial conditions."""
        pass
    
    @abstractmethod
    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        pass
    
    def create_final_visualization(self, solver):
        """Create final visualization plots."""
        pass
    
    def setup_video_generation(self) -> Dict[str, Any]:
        """Setup video generation parameters from config."""
        video_config = self.config.get("video", {})
        return {
            "fps": int(video_config.get("fps", 30)),
            "codec": str(video_config.get("codec", "h264_videotoolbox")),
            "crf": int(video_config.get("crf", 18)),
            "pix_fmt": str(video_config.get("pix_fmt", "yuv420p"))
        }
    
    def generate_video(self, frames_dir: str, video_name: str) -> str:
        """Generate video from frames."""
        video_config = self.setup_video_generation()
        video_path = os.path.join(self.outdir, f"{video_name}.mp4")
        
        # Normalize frame indices
        frames = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        for i, fname in enumerate(frames):
            src = os.path.join(frames_dir, fname)
            dst = os.path.join(frames_dir, f"frame_{i:08d}.png")
            if src != dst:
                os.replace(src, dst)
        
        def try_ffmpeg(codec_name: str) -> bool:
            cmd = [
                "ffmpeg", "-y", "-framerate", str(video_config["fps"]),
                "-i", os.path.join(frames_dir, "frame_%08d.png"),
                "-c:v", codec_name, "-crf", str(video_config["crf"]), 
                "-pix_fmt", video_config["pix_fmt"],
                video_path,
            ]
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except FileNotFoundError:
                return False
            return proc.returncode == 0
        
        for codec in [video_config["codec"], "libopenh264", "mpeg4", "libvpx-vp9", "libaom-av1"]:
            if try_ffmpeg(codec):
                print(f"Wrote video: {video_path} using codec {codec}")
                return video_path
        
        print("Warning: Could not create video. Frames in:", frames_dir)
        return frames_dir
    
    def convert_torch_to_numpy(self, *arrays):
        """Convert torch tensors to numpy arrays for plotting."""
        try:
            import torch
            result = []
            for arr in arrays:
                if isinstance(arr, torch.Tensor):
                    result.append(arr.detach().cpu().numpy())
                else:
                    result.append(arr)
            return result
        except ImportError:
            return arrays
    
    def run(self, backend: str = "numpy", device: Optional[str] = None, 
            generate_video: bool = True) -> Any:
        """Run the simulation."""
        # Create components
        grid = self.create_grid(backend, device)
        equations = self.create_equations()
        U0 = self.create_initial_conditions(grid)
        
        # Create solver
        solver_class = self.get_solver_class()
        solver = solver_class(grid=grid, equations=equations, scheme=self.scheme, cfl=self.cfl)
        
        # Setup visualization
        frames_dir = os.path.join(self.outdir, "frames")
        if generate_video:
            ensure_outdir(frames_dir)
        
        # Run simulation
        if hasattr(solver, 'run'):
            # New solver interface
            history = solver.run(
                U0, 
                t0=self.t0, 
                t_end=self.t_end, 
                output_interval=self.output_interval,
                checkpoint_interval=self.checkpoint_interval,
                outdir=self.outdir,
                on_output=self.create_visualization if generate_video else None
            )
        else:
            # Legacy solver interface
            history = solver.run(
                U0, 
                t0=self.t0, 
                t_end=self.t_end, 
                output_interval=self.output_interval,
                outdir=self.outdir
            )
        
        # Final visualization
        self.create_final_visualization(solver)
        
        # Generate video if requested
        if generate_video and os.path.exists(frames_dir):
            problem_name = self.config.get("problem", "simulation")
            self.generate_video(frames_dir, problem_name)
        
        return solver, history
    
    @abstractmethod
    def get_solver_class(self):
        """Get the appropriate solver class for this problem."""
        pass
