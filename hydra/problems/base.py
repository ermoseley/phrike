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
        
        # Monitoring parameters - enabled by default
        self.monitoring_config = self.config.get("monitoring", {})
        self.monitoring_enabled = bool(self.monitoring_config.get("enabled", True))  # Default: True
        self.monitoring_step_interval = int(self.monitoring_config.get("step_interval", 10))  # Default: 10
        self.monitoring_output_file = self.monitoring_config.get("output_file", None)  # Default: None (stdout)
        self.monitoring_include_conservation = bool(self.monitoring_config.get("include_conservation", True))  # Default: True
        self.monitoring_include_timestep = bool(self.monitoring_config.get("include_timestep", True))  # Default: True
        self.monitoring_include_time = bool(self.monitoring_config.get("include_time", True))  # Default: True
        self.monitoring_include_velocity_stats = bool(self.monitoring_config.get("include_velocity_stats", True))  # Default: True
        self.monitoring_include_density_stats = bool(self.monitoring_config.get("include_density_stats", True))  # Default: True
        
        # Initialize monitoring state
        self.monitoring_step_count = 0
        self.monitoring_initial_values = None
    
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
            "width": int(video_config.get("width", 0)),  # 0 = auto
            "height": int(video_config.get("height", 0)),  # 0 = auto
            "scale": float(video_config.get("scale", 1.0)),
            "codec": str(video_config.get("codec", "h264_videotoolbox")),
            "crf": int(video_config.get("crf", 18)),
            "preset": str(video_config.get("preset", "medium")),
            "pix_fmt": str(video_config.get("pix_fmt", "yuv420p")),
            "bitrate": video_config.get("bitrate"),
            "maxrate": video_config.get("maxrate"),
            "bufsize": video_config.get("bufsize"),
            "tune": video_config.get("tune"),
            "profile": str(video_config.get("profile", "high")),
            "level": str(video_config.get("level", "4.1"))
        }
    
    def generate_video(self, frames_dir: str, video_name: str) -> str:
        """Generate video from frames using enhanced ffmpeg parameters."""
        video_config = self.setup_video_generation()
        video_path = os.path.join(self.outdir, f"{video_name}.mp4")
        
        # Normalize frame indices - find all frame files and rename them sequentially
        frames = sorted([f for f in os.listdir(frames_dir) if f.startswith('frame_') and f.endswith('.png')])
        for i, fname in enumerate(frames):
            src = os.path.join(frames_dir, fname)
            dst = os.path.join(frames_dir, f"frame_{i:08d}.png")
            if src != dst:
                os.replace(src, dst)
        
        def build_ffmpeg_cmd(codec_name: str) -> list:
            """Build ffmpeg command with all configured parameters."""
            cmd = [
                "ffmpeg", "-y",
                "-framerate", str(video_config["fps"]),
                "-i", os.path.join(frames_dir, "frame_%08d.png"),
                "-c:v", codec_name,
                "-pix_fmt", video_config["pix_fmt"]
            ]
            
            # Add preset only for codecs that support it
            if codec_name in ["libx264", "libx265"]:
                cmd.extend(["-preset", video_config["preset"]])
            
            # Add profile and level only for H.264/H.265 codecs
            if codec_name in ["libx264", "h264_videotoolbox"]:
                cmd.extend(["-profile:v", video_config["profile"]])
                cmd.extend(["-level", video_config["level"]])
            
            # Add scaling if specified
            if video_config["scale"] != 1.0:
                cmd.extend(["-vf", f"scale=iw*{video_config['scale']}:ih*{video_config['scale']}"])
            elif video_config["width"] > 0 and video_config["height"] > 0:
                cmd.extend(["-vf", f"scale={video_config['width']}:{video_config['height']}"])
            
            # Add bitrate control (CRF or bitrate)
            if video_config["bitrate"]:
                cmd.extend(["-b:v", video_config["bitrate"]])
                if video_config["maxrate"]:
                    cmd.extend(["-maxrate", video_config["maxrate"]])
                if video_config["bufsize"]:
                    cmd.extend(["-bufsize", video_config["bufsize"]])
            else:
                cmd.extend(["-crf", str(video_config["crf"])])
            
            # Add tune if specified
            if video_config["tune"]:
                cmd.extend(["-tune", video_config["tune"]])
            
            # Add output file
            cmd.append(video_path)
            
            return cmd
        
        def try_ffmpeg(codec_name: str) -> bool:
            """Try to run ffmpeg with the given codec."""
            cmd = build_ffmpeg_cmd(codec_name)
            try:
                proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                if proc.returncode == 0:
                    print(f"Wrote video: {video_path} using codec {codec_name}")
                    print(f"Command: {' '.join(cmd)}")
                    return True
                else:
                    print(f"FFmpeg failed with codec {codec_name}: {proc.stderr.decode()}")
                    return False
            except FileNotFoundError:
                print("FFmpeg not found. Please install ffmpeg to generate videos.")
                return False
        
        # Try different codecs in order of preference
        codecs_to_try = [
            video_config["codec"],
            "libx264",
            "libopenh264", 
            "mpeg4",
            "libvpx-vp9",
            "libaom-av1"
        ]
        
        for codec in codecs_to_try:
            if try_ffmpeg(codec):
                return video_path
        
        print("Warning: Could not create video with any codec. Frames in:", frames_dir)
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
    
    def compute_conservation_errors(self, solver, U):
        """Compute conservation errors relative to initial values."""
        if self.monitoring_initial_values is None:
            return {}
        
        # Get current conserved quantities
        if hasattr(solver.equations, 'conserved_quantities'):
            current_cons = solver.equations.conserved_quantities(U)
        else:
            # Fallback: compute manually
            primitive_vars = solver.equations.primitive(U)
            primitive_vars = self.convert_torch_to_numpy(*primitive_vars)
            
            # Compute conserved quantities
            if len(primitive_vars) == 4:  # Could be 1D or 2D
                rho = primitive_vars[0]
                if len(rho.shape) == 1:  # 1D: rho, u, p, a
                    rho, u, p, a = primitive_vars
                    current_cons = {
                        "mass": float(np.sum(rho)),
                        "momentum": float(np.sum(rho * u)),
                        "energy": float(np.sum(p / (solver.equations.gamma - 1.0) + 0.5 * rho * u**2))
                    }
                else:  # 2D: rho, ux, uy, p
                    rho, ux, uy, p = primitive_vars
                    current_cons = {
                        "mass": float(np.sum(rho)),
                        "momentum_x": float(np.sum(rho * ux)),
                        "momentum_y": float(np.sum(rho * uy)),
                        "energy": float(np.sum(p / (solver.equations.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)))
                    }
            elif len(primitive_vars) == 5:  # 3D: rho, ux, uy, uz, p
                rho, ux, uy, uz, p = primitive_vars
                current_cons = {
                    "mass": float(np.sum(rho)),
                    "momentum_x": float(np.sum(rho * ux)),
                    "momentum_y": float(np.sum(rho * uy)),
                    "momentum_z": float(np.sum(rho * uz)),
                    "energy": float(np.sum(p / (solver.equations.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2 + uz**2)))
                }
            else:
                raise ValueError(f"Unexpected number of primitive variables: {len(primitive_vars)}")
        
        # Compute relative errors
        errors = {}
        for key, initial_val in self.monitoring_initial_values.items():
            if key in current_cons and initial_val != 0:
                errors[f"{key}_error"] = abs((current_cons[key] - initial_val) / initial_val)
            elif key in current_cons:
                errors[f"{key}_error"] = abs(current_cons[key] - initial_val)
        
        return errors
    
    def compute_velocity_stats(self, solver, U):
        """Compute velocity statistics."""
        primitive_vars = solver.equations.primitive(U)
        primitive_vars = self.convert_torch_to_numpy(*primitive_vars)
        
        # Handle different dimensions
        if len(primitive_vars) == 4:  # Could be 1D or 2D
            rho = primitive_vars[0]
            if len(rho.shape) == 1:  # 1D: rho, u, p, a
                rho, u, p, a = primitive_vars
                v_mag = np.abs(u)
            else:  # 2D: rho, ux, uy, p
                rho, ux, uy, p = primitive_vars
                v_mag = np.sqrt(ux**2 + uy**2)
        elif len(primitive_vars) == 5:  # 3D: rho, ux, uy, uz, p
            rho, ux, uy, uz, p = primitive_vars
            v_mag = np.sqrt(ux**2 + uy**2 + uz**2)
        else:
            raise ValueError(f"Unexpected number of primitive variables: {len(primitive_vars)}")
        
        # Compute density-weighted statistics
        total_mass = np.sum(rho)
        if total_mass > 0:
            v_rms = np.sqrt(np.sum(rho * v_mag**2) / total_mass)
            v_max = np.max(v_mag)
            v_min = np.min(v_mag)
        else:
            v_rms = v_max = v_min = 0.0
        
        return {
            "v_rms": float(v_rms),
            "v_max": float(v_max),
            "v_min": float(v_min)
        }
    
    def compute_density_stats(self, solver, U):
        """Compute density statistics."""
        primitive_vars = solver.equations.primitive(U)
        primitive_vars = self.convert_torch_to_numpy(*primitive_vars)
        
        # Extract density (first variable)
        rho = primitive_vars[0]
        
        return {
            "rho_mean": float(np.mean(rho)),
            "rho_max": float(np.max(rho)),
            "rho_min": float(np.min(rho)),
            "rho_std": float(np.std(rho))
        }
    
    def output_monitoring_info(self, solver, U, step_count, dt):
        """Output monitoring information."""
        if not self.monitoring_enabled:
            return
        
        # Prepare monitoring data
        info_lines = []
        
        if self.monitoring_include_time:
            info_lines.append(f"Time: {solver.t:.6f}")
        
        if self.monitoring_include_timestep:
            info_lines.append(f"Step: {step_count}, dt: {dt:.2e}")
        
        if self.monitoring_include_conservation:
            errors = self.compute_conservation_errors(solver, U)
            for key, error in errors.items():
                info_lines.append(f"{key}: {error:.2e}")
        
        if self.monitoring_include_velocity_stats:
            v_stats = self.compute_velocity_stats(solver, U)
            for key, value in v_stats.items():
                info_lines.append(f"{key}: {value:.6f}")
        
        if self.monitoring_include_density_stats:
            rho_stats = self.compute_density_stats(solver, U)
            for key, value in rho_stats.items():
                info_lines.append(f"{key}: {value:.6f}")
        
        # Output the information
        output_text = " | ".join(info_lines)
        
        if self.monitoring_output_file:
            # Write to file
            log_path = os.path.join(self.outdir, self.monitoring_output_file)
            with open(log_path, "a") as f:
                f.write(f"{output_text}\n")
        else:
            # Print to stdout
            print(output_text)
    
    def initialize_monitoring(self, solver, U):
        """Initialize monitoring with initial values."""
        if not self.monitoring_enabled:
            return
        
        # Store initial conserved quantities for error computation
        if hasattr(solver.equations, 'conserved_quantities'):
            self.monitoring_initial_values = solver.equations.conserved_quantities(U)
        else:
            # Fallback: compute manually
            primitive_vars = solver.equations.primitive(U)
            primitive_vars = self.convert_torch_to_numpy(*primitive_vars)
            
            if len(primitive_vars) == 4:  # Could be 1D or 2D
                rho = primitive_vars[0]
                if len(rho.shape) == 1:  # 1D: rho, u, p, a
                    rho, u, p, a = primitive_vars
                    self.monitoring_initial_values = {
                        "mass": float(np.sum(rho)),
                        "momentum": float(np.sum(rho * u)),
                        "energy": float(np.sum(p / (solver.equations.gamma - 1.0) + 0.5 * rho * u**2))
                    }
                else:  # 2D: rho, ux, uy, p
                    rho, ux, uy, p = primitive_vars
                    self.monitoring_initial_values = {
                        "mass": float(np.sum(rho)),
                        "momentum_x": float(np.sum(rho * ux)),
                        "momentum_y": float(np.sum(rho * uy)),
                        "energy": float(np.sum(p / (solver.equations.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)))
                    }
            elif len(primitive_vars) == 5:  # 3D: rho, ux, uy, uz, p
                rho, ux, uy, uz, p = primitive_vars
                self.monitoring_initial_values = {
                    "mass": float(np.sum(rho)),
                    "momentum_x": float(np.sum(rho * ux)),
                    "momentum_y": float(np.sum(rho * uy)),
                    "momentum_z": float(np.sum(rho * uz)),
                    "energy": float(np.sum(p / (solver.equations.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2 + uz**2)))
                }
            else:
                raise ValueError(f"Unexpected number of primitive variables: {len(primitive_vars)}")
        
        # Reset step counter
        self.monitoring_step_count = 0
    
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
        
        # Initialize monitoring
        self.initialize_monitoring(solver, U0)
        
        # Run simulation
        if hasattr(solver, 'run'):
            # New solver interface
            def visualization_callback(t, U):
                self.create_visualization(solver, t, U)
            
            def monitoring_callback(step_count, dt, U):
                if self.monitoring_enabled and step_count % self.monitoring_step_interval == 0:
                    self.output_monitoring_info(solver, U, step_count, dt)
            
            history = solver.run(
                U0, 
                t0=self.t0, 
                t_end=self.t_end, 
                output_interval=self.output_interval,
                checkpoint_interval=self.checkpoint_interval,
                outdir=self.outdir,
                on_output=visualization_callback if generate_video else None,
                on_step=monitoring_callback
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
