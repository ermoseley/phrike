"""Base problem class for PHRIKE simulations."""

import os
import subprocess
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from phrike.io import load_config, ensure_outdir


class BaseProblem(ABC):
    """Base class for all PHRIKE problems."""

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

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
        restart_from: Optional[str] = None,
    ):
        """Initialize problem with configuration.

        Args:
            config_path: Path to YAML configuration file
            config: Configuration dictionary (takes precedence over config_path)
            restart_from: Path to checkpoint file for restart
        """
        self.restart_from = restart_from

        if config is not None:
            self.config = config
        elif config_path is not None:
            self.config = load_config(config_path)
        else:
            raise ValueError("Either config_path or config must be provided")

        self.setup_output_directory()
        self.setup_common_parameters()

        # Load restart data if specified
        if self.restart_from is not None:
            self.load_restart_data()

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
        self.gamma = float(self.config["equations"]["gamma"])

        # Integration parameters
        self.t0 = float(self.config["integration"]["t0"])
        self.t_end = float(self.config["integration"]["t_end"])
        self.cfl = float(self.config["integration"]["cfl"])
        self.scheme = str(self.config["integration"].get("scheme", "rk4"))
        self.output_interval = float(
            self.config["integration"].get("output_interval", 0.1)
        )
        self.checkpoint_interval = float(
            self.config["integration"].get("checkpoint_interval", 0.0)
        )
        self.filter_config = self.config["integration"].get(
            "spectral_filter", {"enabled": False}
        )
        
        # Grid parameters
        self.precision = str(self.config["grid"].get("precision", "double"))
        
        # Adaptive time-stepping parameters
        adaptive_raw = self.config["integration"].get("adaptive", None)
        if adaptive_raw:
            self.adaptive_config = {
                "enabled": bool(adaptive_raw.get("enabled", False)),
                "scheme": str(adaptive_raw.get("scheme", "rk45")),
                "rtol": float(adaptive_raw.get("rtol", 1e-6)),
                "atol": float(adaptive_raw.get("atol", 1e-8)),
                "safety_factor": float(adaptive_raw.get("safety_factor", 0.9)),
                "min_dt_factor": float(adaptive_raw.get("min_dt_factor", 0.1)),
                "max_dt_factor": float(adaptive_raw.get("max_dt_factor", 5.0)),
                "max_rejections": int(adaptive_raw.get("max_rejections", 10)),
                "fallback_scheme": str(adaptive_raw.get("fallback_scheme", "rk4"))
            }
        else:
            self.adaptive_config = None
        
        # Artificial viscosity parameters
        artificial_viscosity_raw = self.config.get("artificial_viscosity", None)
        if artificial_viscosity_raw:
            self.artificial_viscosity_config = {
                "enabled": bool(artificial_viscosity_raw.get("enabled", False)),
                "nu_max": float(artificial_viscosity_raw.get("nu_max", 1e-3)),
                "s_ref": float(artificial_viscosity_raw.get("s_ref", 1.0)),
                "s_min": float(artificial_viscosity_raw.get("s_min", 0.1)),
                "p": float(artificial_viscosity_raw.get("p", 2.0)),
                "epsilon": float(artificial_viscosity_raw.get("epsilon", 1e-12)),
                "variable_weights": artificial_viscosity_raw.get("variable_weights", {
                    "density": 1.0,
                    "momentum_x": 1.0,
                    "momentum_y": 1.0,
                    "momentum_z": 1.0,
                    "energy": 1.0
                }),
                "sensor_variable": str(artificial_viscosity_raw.get("sensor_variable", "density")),
                "diagnostic_output": bool(artificial_viscosity_raw.get("diagnostic_output", False))
            }
        else:
            self.artificial_viscosity_config = None

        # Initial conditions smoothing parameters
        ic_smoothing_raw = self.config.get("initial_conditions_smoothing", None)
        if ic_smoothing_raw:
            self.ic_smoothing_config = {
                "enabled": bool(ic_smoothing_raw.get("enabled", False)),
                "kernel_size": float(ic_smoothing_raw.get("kernel_size", 0.0)),
                "mode": str(ic_smoothing_raw.get("mode", "wrap")),  # 'wrap' for periodic, 'constant' for non-periodic
                "variables": ic_smoothing_raw.get("variables", ["density", "velocity", "pressure"]),
                "diagnostic_output": bool(ic_smoothing_raw.get("diagnostic_output", False))
            }
        else:
            self.ic_smoothing_config = None

        # Threading / FFT workers (unified via runtime.num_threads, fallback to grid.fft_workers)
        runtime_cfg = self.config.get("runtime", {})
        num_threads_cfg = runtime_cfg.get("num_threads", None)
        threads: int | None
        if num_threads_cfg is not None:
            if isinstance(num_threads_cfg, str) and num_threads_cfg.strip().lower() == "auto":
                threads = os.cpu_count() or 1
            else:
                try:
                    threads = int(num_threads_cfg)
                except Exception:
                    threads = os.cpu_count() or 1
        else:
            threads = None

        grid_threads = self.config["grid"].get("fft_workers", None)
        if grid_threads is not None:
            try:
                self.fft_workers = int(grid_threads)
            except Exception:
                self.fft_workers = os.cpu_count() or 1
        elif threads is not None:
            self.fft_workers = threads
        else:
            self.fft_workers = os.cpu_count() or 1

        # Monitoring parameters - enabled by default
        self.monitoring_config = self.config.get("monitoring", {})
        self.monitoring_enabled = bool(
            self.monitoring_config.get("enabled", True)
        )  # Default: True
        self.monitoring_step_interval = int(
            self.monitoring_config.get("step_interval", 10)
        )  # Default: 10
        self.monitoring_output_file = self.monitoring_config.get(
            "output_file", None
        )  # Default: None (stdout)
        self.monitoring_include_conservation = bool(
            self.monitoring_config.get("include_conservation", True)
        )  # Default: True
        self.monitoring_include_timestep = bool(
            self.monitoring_config.get("include_timestep", True)
        )  # Default: True
        self.monitoring_include_time = bool(
            self.monitoring_config.get("include_time", True)
        )  # Default: True
        self.monitoring_include_velocity_stats = bool(
            self.monitoring_config.get("include_velocity_stats", True)
        )  # Default: True
        self.monitoring_include_density_stats = bool(
            self.monitoring_config.get("include_density_stats", True)
        )  # Default: True

        # Initialize monitoring state
        self.monitoring_step_count = 0
        self.monitoring_initial_values = None

        # Initialize restart data
        self.restart_data = None

    def load_restart_data(self) -> None:
        """Load restart data from checkpoint file."""
        from phrike.io import load_checkpoint

        if self.restart_from is None:
            return

        print(f"Loading restart data from: {self.restart_from}")
        self.restart_data = load_checkpoint(self.restart_from)

        # Update configuration with restart time
        if "integration" not in self.config:
            self.config["integration"] = {}
        self.config["integration"]["t0"] = self.restart_data["t"]

        # Update common parameters
        self.t0 = self.restart_data["t"]

        print(f"Restarting simulation at t = {self.t0:.6f}")

    def validate_restart_data(self, grid, equations) -> None:
        """Validate that restart data is compatible with current problem configuration."""
        if self.restart_data is None:
            return

        # Check gamma parameter
        if "gamma" in self.restart_data["meta"]:
            restart_gamma = self.restart_data["meta"]["gamma"]
            if abs(restart_gamma - equations.gamma) > 1e-10:
                print(
                    f"Warning: Gamma mismatch - restart: {restart_gamma}, current: {equations.gamma}"
                )

        # Check grid dimensions
        restart_meta = self.restart_data["meta"]
        if "N" in restart_meta:  # 1D case
            if restart_meta["N"] != grid.N:
                raise ValueError(
                    f"Grid size mismatch - restart: {restart_meta['N']}, current: {grid.N}"
                )
        elif "Nx" in restart_meta:  # 2D/3D case
            if restart_meta["Nx"] != grid.Nx:
                raise ValueError(
                    f"Grid Nx mismatch - restart: {restart_meta['Nx']}, current: {grid.Nx}"
                )
            if "Ny" in restart_meta and restart_meta["Ny"] != grid.Ny:
                raise ValueError(
                    f"Grid Ny mismatch - restart: {restart_meta['Ny']}, current: {grid.Ny}"
                )
            if "Nz" in restart_meta and restart_meta["Nz"] != grid.Nz:
                raise ValueError(
                    f"Grid Nz mismatch - restart: {restart_meta['Nz']}, current: {grid.Nz}"
                )

    @abstractmethod
    def create_grid(self, backend: str = "numpy", device: Optional[str] = None, debug: bool = False):
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
            "quality": str(video_config.get("quality", "high")),  # low, medium, high
            "crf": int(video_config.get("crf", 23)),  # Default to higher quality
            "preset": str(video_config.get("preset", "medium")),
            "pix_fmt": str(video_config.get("pix_fmt", "yuv420p")),
            "bitrate": video_config.get("bitrate"),
            "maxrate": video_config.get("maxrate"),
            "bufsize": video_config.get("bufsize"),
            "tune": video_config.get("tune"),
            "profile": str(video_config.get("profile", "high")),
            "level": str(video_config.get("level", "4.1")),
        }

    def generate_video(self, frames_dir: str, video_name: str) -> str:
        """Generate video from frames using enhanced ffmpeg parameters based on amr2vid.py approach."""
        video_config = self.setup_video_generation()
        video_path = os.path.join(self.outdir, f"{video_name}.mp4")

        # Normalize frame indices - find all frame files and rename them sequentially
        frames = sorted(
            [
                f
                for f in os.listdir(frames_dir)
                if f.startswith("frame_") and f.endswith(".png")
            ]
        )
        
        print(f"Found {len(frames)} frames to rename")
        for i, fname in enumerate(frames):
            src = os.path.join(frames_dir, fname)
            dst = os.path.join(frames_dir, f"frame_{i:08d}.png")
            if src != dst:
                print(f"Renaming {fname} -> frame_{i:08d}.png")
                os.replace(src, dst)

        # Check if ffmpeg is available
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: ffmpeg not found. Please install ffmpeg to create movies.")
            print("Frames have been generated in:", frames_dir)
            return frames_dir

        # Define encoding options based on quality and available codecs
        quality = video_config["quality"]
        
        # Try different encoding options based on what's available
        encoding_options = [
            # Option 1: H.264 with libx264 (best quality)
            {
                "name": "H.264 (libx264)",
                "cmd": [
                    "ffmpeg", "-y",
                    "-framerate", str(video_config["fps"]),
                    "-i", os.path.join(frames_dir, "frame_%08d.png"),
                    "-c:v", "libx264",
                    "-crf", "18" if quality == "high" else "23" if quality == "medium" else "28",
                    "-pix_fmt", video_config["pix_fmt"],
                ]
            },
            # Option 2: H.264 with h264_videotoolbox (macOS hardware acceleration)
            {
                "name": "H.264 (videotoolbox)",
                "cmd": [
                    "ffmpeg", "-y",
                    "-framerate", str(video_config["fps"]),
                    "-i", os.path.join(frames_dir, "frame_%08d.png"),
                    "-c:v", "h264_videotoolbox",
                    "-b:v", "15M" if quality == "high" else "10M" if quality == "medium" else "5M",
                    "-pix_fmt", video_config["pix_fmt"],
                    "-profile:v", video_config["profile"],
                    "-level", video_config["level"],
                ]
            },
            # Option 3: H.264 with nvenc (NVIDIA hardware acceleration)
            {
                "name": "H.264 (nvenc)",
                "cmd": [
                    "ffmpeg", "-y",
                    "-framerate", str(video_config["fps"]),
                    "-i", os.path.join(frames_dir, "frame_%08d.png"),
                    "-c:v", "h264_nvenc",
                    "-b:v", "15M" if quality == "high" else "10M" if quality == "medium" else "5M",
                    "-pix_fmt", video_config["pix_fmt"],
                ]
            },
            # Option 4: MPEG-4 (more widely supported)
            {
                "name": "MPEG-4",
                "cmd": [
                    "ffmpeg", "-y",
                    "-framerate", str(video_config["fps"]),
                    "-i", os.path.join(frames_dir, "frame_%08d.png"),
                    "-c:v", "mpeg4",
                    "-q:v", "1" if quality == "high" else "2" if quality == "medium" else "5",
                    "-pix_fmt", video_config["pix_fmt"],
                ]
            },
            # Option 5: VP9 (good compression)
            {
                "name": "VP9",
                "cmd": [
                    "ffmpeg", "-y",
                    "-framerate", str(video_config["fps"]),
                    "-i", os.path.join(frames_dir, "frame_%08d.png"),
                    "-c:v", "libvpx-vp9",
                    "-crf", "15" if quality == "high" else "20" if quality == "medium" else "25",
                    "-b:v", "0",  # Use CRF mode
                    "-pix_fmt", video_config["pix_fmt"],
                ]
            }
        ]

        # Add scaling and output file to all options
        for option in encoding_options:
            # Add scaling if specified
            if video_config["scale"] != 1.0:
                option["cmd"].extend(["-vf", f"scale=iw*{video_config['scale']}:ih*{video_config['scale']}"])
            elif video_config["width"] > 0 and video_config["height"] > 0:
                option["cmd"].extend(["-vf", f"scale={video_config['width']}:{video_config['height']}"])
            
            # Add output file
            option["cmd"].append(video_path)

        print(f"Creating movie: {video_path}")
        print(f"Using {len(frames)} frames at {video_config['fps']} fps")
        print(f"Quality setting: {quality}")

        # Try each encoding option until one works
        for option in encoding_options:
            print(f"Trying {option['name']}...")
            try:
                result = subprocess.run(option['cmd'], capture_output=True, text=True, timeout=300)
                if result.returncode == 0:
                    print(f"Movie created successfully using {option['name']}: {video_path}")
                    return video_path
                else:
                    print(f"Failed with {option['name']}: {result.stderr}")
                    continue
            except subprocess.TimeoutExpired:
                print(f"Timeout with {option['name']}")
                continue
            except Exception as e:
                print(f"Exception with {option['name']}: {e}")
                continue

        print("All encoding options failed. Frames are available for manual processing.")
        print("You can try creating the movie manually with:")
        print(f"ffmpeg -framerate {video_config['fps']} -i {frames_dir}/frame_%05d.png -c:v libx264 -crf 23 -pix_fmt yuv420p {video_path}")
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
        if hasattr(solver.equations, "conserved_quantities"):
            current_cons = solver.equations.conserved_quantities(U)
        else:
            # Fallback: compute manually - work directly with torch tensors
            primitive_vars = solver.equations.primitive(U)
            # Only convert to numpy for final computation, not intermediate steps

            # Compute conserved quantities - work with torch tensors directly
            try:
                import torch
                is_torch = any(isinstance(v, torch.Tensor) for v in primitive_vars)
            except ImportError:
                is_torch = False
                torch = None

            if len(primitive_vars) == 4:  # Could be 1D or 2D
                rho = primitive_vars[0]
                if len(rho.shape) == 1:  # 1D: rho, u, p, a
                    rho, u, p, a = primitive_vars
                    if is_torch:
                        current_cons = {
                            "mass": float(torch.sum(rho).item()),
                            "momentum": float(torch.sum(rho * u).item()),
                            "energy": float(
                                torch.sum(
                                    p / (solver.equations.gamma - 1.0) + 0.5 * rho * u**2
                                ).item()
                            ),
                        }
                    else:
                        current_cons = {
                            "mass": float(rho.sum()),
                            "momentum": float((rho * u).sum()),
                            "energy": float(
                                (
                                    p / (solver.equations.gamma - 1.0) + 0.5 * rho * u**2
                                ).sum()
                            ),
                        }
                else:  # 2D: rho, ux, uy, p
                    rho, ux, uy, p = primitive_vars
                    if is_torch:
                        current_cons = {
                            "mass": float(torch.sum(rho).item()),
                            "momentum_x": float(torch.sum(rho * ux).item()),
                            "momentum_y": float(torch.sum(rho * uy).item()),
                            "energy": float(
                                torch.sum(
                                    p / (solver.equations.gamma - 1.0)
                                    + 0.5 * rho * (ux**2 + uy**2)
                                ).item()
                            ),
                        }
                    else:
                        current_cons = {
                            "mass": float(rho.sum()),
                            "momentum_x": float((rho * ux).sum()),
                            "momentum_y": float((rho * uy).sum()),
                            "energy": float(
                                (
                                    p / (solver.equations.gamma - 1.0)
                                    + 0.5 * rho * (ux**2 + uy**2)
                                ).sum()
                            ),
                        }
            elif len(primitive_vars) == 5:  # 3D: rho, ux, uy, uz, p
                rho, ux, uy, uz, p = primitive_vars
                if is_torch:
                    current_cons = {
                        "mass": float(torch.sum(rho).item()),
                        "momentum_x": float(torch.sum(rho * ux).item()),
                        "momentum_y": float(torch.sum(rho * uy).item()),
                        "momentum_z": float(torch.sum(rho * uz).item()),
                        "energy": float(
                            torch.sum(
                                p / (solver.equations.gamma - 1.0)
                                + 0.5 * rho * (ux**2 + uy**2 + uz**2)
                            ).item()
                        ),
                    }
                else:
                    current_cons = {
                        "mass": float(rho.sum()),
                        "momentum_x": float((rho * ux).sum()),
                        "momentum_y": float((rho * uy).sum()),
                        "momentum_z": float((rho * uz).sum()),
                        "energy": float(
                            (
                                p / (solver.equations.gamma - 1.0)
                                + 0.5 * rho * (ux**2 + uy**2 + uz**2)
                            ).sum()
                        ),
                    }
            else:
                raise ValueError(
                    f"Unexpected number of primitive variables: {len(primitive_vars)}"
                )

        # Compute conservation errors relative to initial values.
        # Keep sign for mass and energy; keep momentum errors absolute.
        errors = {}
        for key, initial_val in self.monitoring_initial_values.items():
            if key in current_cons:
                if initial_val != 0:
                    delta = (current_cons[key] - initial_val) / initial_val
                else:
                    delta = current_cons[key] - initial_val

                # Momentum components (including 1D 'momentum') remain absolute
                if key.startswith("momentum"):
                    errors[f"{key}_error"] = abs(delta)
                else:
                    # mass and energy (and any other conserved scalars) keep sign
                    errors[f"{key}_error"] = delta

        return errors

    def compute_velocity_stats(self, solver, U):
        """Compute velocity statistics."""
        primitive_vars = solver.equations.primitive(U)
        # Work directly with torch tensors, only convert final results

        # Handle different dimensions - work with torch tensors directly
        try:
            import torch
            is_torch = any(isinstance(v, torch.Tensor) for v in primitive_vars)
        except ImportError:
            is_torch = False
            torch = None

        if len(primitive_vars) == 4:  # Could be 1D or 2D
            rho = primitive_vars[0]
            if len(rho.shape) == 1:  # 1D: rho, u, p, a
                rho, u, p, a = primitive_vars
                if is_torch:
                    v_mag = torch.abs(u)
                else:
                    v_mag = np.abs(u)
            else:  # 2D: rho, ux, uy, p
                rho, ux, uy, p = primitive_vars
                if is_torch:
                    v_mag = torch.sqrt(ux**2 + uy**2)
                else:
                    v_mag = np.sqrt(ux**2 + uy**2)
        elif len(primitive_vars) == 5:  # 3D: rho, ux, uy, uz, p
            rho, ux, uy, uz, p = primitive_vars
            if is_torch:
                v_mag = torch.sqrt(ux**2 + uy**2 + uz**2)
            else:
                v_mag = np.sqrt(ux**2 + uy**2 + uz**2)
        else:
            raise ValueError(
                f"Unexpected number of primitive variables: {len(primitive_vars)}"
            )

        # Compute density-weighted statistics
        if is_torch:
            total_mass = torch.sum(rho)
            if total_mass > 0:
                v_rms = torch.sqrt(torch.sum(rho * v_mag**2) / total_mass)
                v_max = torch.max(v_mag)
                v_min = torch.min(v_mag)
            else:
                v_rms = v_max = v_min = torch.tensor(0.0, device=rho.device)
            return {"v_rms": float(v_rms.item()), "v_max": float(v_max.item()), "v_min": float(v_min.item())}
        else:
            total_mass = rho.sum()
            if total_mass > 0:
                v_rms = np.sqrt((rho * v_mag**2).sum() / total_mass)
                v_max = v_mag.max()
                v_min = v_mag.min()
            else:
                v_rms = v_max = v_min = 0.0
            return {"v_rms": float(v_rms), "v_max": float(v_max), "v_min": float(v_min)}

    def compute_density_stats(self, solver, U):
        """Compute density statistics."""
        primitive_vars = solver.equations.primitive(U)
        # Work directly with torch tensors, only convert final results

        # Extract density (first variable)
        rho = primitive_vars[0]

        # Work with torch tensors directly
        try:
            import torch
            is_torch = isinstance(rho, torch.Tensor)
        except ImportError:
            is_torch = False
            torch = None

        if is_torch:
            return {
                "rho_mean": float(torch.mean(rho).item()),
                "rho_max": float(torch.max(rho).item()),
                "rho_min": float(torch.min(rho).item()),
                "rho_std": float(torch.std(rho).item()),
            }
        else:
            return {
                "rho_mean": float(np.mean(rho)),
                "rho_max": float(np.max(rho)),
                "rho_min": float(np.min(rho)),
                "rho_std": float(np.std(rho)),
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
        if hasattr(solver.equations, "conserved_quantities"):
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
                        "mass": float(rho.sum()),
                        "momentum": float((rho * u).sum()),
                        "energy": float(
                            (
                                p / (solver.equations.gamma - 1.0) + 0.5 * rho * u**2
                            ).sum()
                        ),
                    }
                else:  # 2D: rho, ux, uy, p
                    rho, ux, uy, p = primitive_vars
                    self.monitoring_initial_values = {
                        "mass": float(rho.sum()),
                        "momentum_x": float((rho * ux).sum()),
                        "momentum_y": float((rho * uy).sum()),
                        "energy": float(
                            (
                                p / (solver.equations.gamma - 1.0)
                                + 0.5 * rho * (ux**2 + uy**2)
                            ).sum()
                        ),
                    }
            elif len(primitive_vars) == 5:  # 3D: rho, ux, uy, uz, p
                rho, ux, uy, uz, p = primitive_vars
                self.monitoring_initial_values = {
                    "mass": float(rho.sum()),
                    "momentum_x": float((rho * ux).sum()),
                    "momentum_y": float((rho * uy).sum()),
                    "momentum_z": float((rho * uz).sum()),
                    "energy": float(
                        (
                            p / (solver.equations.gamma - 1.0)
                            + 0.5 * rho * (ux**2 + uy**2 + uz**2)
                        ).sum()
                    ),
                }
            else:
                raise ValueError(
                    f"Unexpected number of primitive variables: {len(primitive_vars)}"
                )

        # Reset step counter
        self.monitoring_step_count = 0

    def run(
        self,
        backend: str = "numpy",
        device: Optional[str] = None,
        generate_video: bool = True,
        debug: bool = False,
    ) -> Any:
        """Run the simulation."""
        # Create components
        grid = self.create_grid(backend, device, debug)
        equations = self.create_equations()

        # Create initial conditions (use restart data if available)
        if self.restart_data is not None:
            U0 = self.restart_data["U"]
            print(f"Using restart data for initial conditions (shape: {U0.shape})")
        else:
            U0 = self.create_initial_conditions(grid)

        # Validate restart data compatibility
        self.validate_restart_data(grid, equations)

        # Create solver
        solver_class = self.get_solver_class()
        solver = solver_class(
            grid=grid, equations=equations, scheme=self.scheme, cfl=self.cfl,
            adaptive_config=self.adaptive_config
        )

        # Setup visualization
        frames_dir = os.path.join(self.outdir, "frames")
        if generate_video:
            ensure_outdir(frames_dir)

        # Initialize monitoring
        self.initialize_monitoring(solver, U0)

        # Run simulation
        if hasattr(solver, "run"):
            # New solver interface
            def visualization_callback(t, U):
                self.create_visualization(solver, t, U)

            def monitoring_callback(step_count, dt, U):
                if (
                    self.monitoring_enabled
                    and step_count % self.monitoring_step_interval == 0
                ):
                    self.output_monitoring_info(solver, U, step_count, dt)

            history = solver.run(
                U0,
                t0=self.t0,
                t_end=self.t_end,
                output_interval=self.output_interval,
                checkpoint_interval=self.checkpoint_interval,
                outdir=self.outdir,
                on_output=visualization_callback if generate_video else None,
                on_step=monitoring_callback,
            )
        else:
            # Legacy solver interface
            history = solver.run(
                U0,
                t0=self.t0,
                t_end=self.t_end,
                output_interval=self.output_interval,
                outdir=self.outdir,
            )

        # Final visualization
        self.create_final_visualization(solver)

        # Generate video if requested
        if generate_video and os.path.exists(frames_dir):
            problem_name = self.config.get("problem", "simulation")
            self.generate_video(frames_dir, problem_name)

        return solver, history

    def apply_initial_conditions_smoothing(self, rho, u, p, grid):
        """Apply smoothing to initial conditions if configured.
        
        Args:
            rho: Density array
            u: Velocity array (can be 1D, 2D, or 3D)
            p: Pressure array
            grid: Grid object containing spatial coordinates
            
        Returns:
            Tuple of (rho_smooth, u_smooth, p_smooth)
        """
        if not self.ic_smoothing_config or not self.ic_smoothing_config["enabled"]:
            return rho, u, p
        
        kernel_size = self.ic_smoothing_config["kernel_size"]
        if kernel_size <= 0:
            return rho, u, p
        
        mode = self.ic_smoothing_config["mode"]
        variables = self.ic_smoothing_config["variables"]
        diagnostic = self.ic_smoothing_config["diagnostic_output"]
        
        # Import scipy here to avoid dependency issues
        try:
            from scipy import ndimage
        except ImportError:
            print("Warning: scipy not available for initial conditions smoothing")
            return rho, u, p
        
        # Calculate sigma in grid points
        if hasattr(grid, 'N'):
            # 1D case
            sigma = kernel_size * grid.N / grid.Lx
        elif hasattr(grid, 'Nx'):
            # 2D case
            sigma = kernel_size * grid.Nx / grid.Lx
        elif hasattr(grid, 'Nx') and hasattr(grid, 'Ny'):
            # 3D case - use average
            sigma = kernel_size * (grid.Nx + grid.Ny + grid.Nz) / (grid.Lx + grid.Ly + grid.Lz) / 3
        else:
            print("Warning: Cannot determine grid size for smoothing")
            return rho, u, p
        
        if diagnostic:
            print(f"Applying initial conditions smoothing: kernel_size={kernel_size:.4f}, sigma={sigma:.2f} grid points, mode='{mode}'")
        
        # Apply smoothing to specified variables
        rho_smooth = rho.copy()
        u_smooth = u.copy()
        p_smooth = p.copy()
        
        if "density" in variables:
            rho_smooth = ndimage.gaussian_filter1d(rho, sigma=sigma, mode=mode)
        
        if "velocity" in variables:
            if len(u.shape) == 1:
                # 1D velocity
                u_smooth = ndimage.gaussian_filter1d(u, sigma=sigma, mode=mode)
            elif len(u.shape) == 2:
                # 2D velocity (u, v)
                u_smooth[0] = ndimage.gaussian_filter1d(u[0], sigma=sigma, mode=mode)
                u_smooth[1] = ndimage.gaussian_filter1d(u[1], sigma=sigma, mode=mode)
            elif len(u.shape) == 3:
                # 3D velocity (u, v, w)
                u_smooth[0] = ndimage.gaussian_filter1d(u[0], sigma=sigma, mode=mode)
                u_smooth[1] = ndimage.gaussian_filter1d(u[1], sigma=sigma, mode=mode)
                u_smooth[2] = ndimage.gaussian_filter1d(u[2], sigma=sigma, mode=mode)
        
        if "pressure" in variables:
            p_smooth = ndimage.gaussian_filter1d(p, sigma=sigma, mode=mode)
        
        if diagnostic:
            print(f"Initial conditions smoothing applied to: {variables}")
        
        return rho_smooth, u_smooth, p_smooth

    @abstractmethod
    def get_solver_class(self):
        """Get the appropriate solver class for this problem."""
        pass
