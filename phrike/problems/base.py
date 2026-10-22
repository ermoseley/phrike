"""Base problem class for PHRIKE simulations."""

import os
import shutil
import subprocess
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from phrike.io import (
    ensure_outdir,
    grid_snapshot_identity,
    load_config,
    save_solution_snapshot,
)
from phrike.solver import conserved_diagnostics, nodal_quadrature_weights
from phrike.tracers import FourierTracers1D, FourierTracers2D, FourierTracers3D

# ffmpeg image sequence. Frames are renamed to this pattern before encoding.
_FRAME_SEQUENCE = "frame_%08d.png"
_MOMENTUM_SCALE_NOTE = "normalized by max(sum|rho*u|, mass*c_s)"
_QUALITY_CRF = {"low": 28, "medium": 23, "high": 18}
_QUALITY_BITRATE = {"low": "5M", "medium": "10M", "high": "15M"}
_CRF_CODECS = {"libx264", "libx265", "libopenh264", "libvpx-vp9", "libaom-av1"}
_PRESET_CODECS = {"libx264", "libx265", "libaom-av1"}
_TUNE_CODECS = {"libx264", "libx265"}
_HARDWARE_CODECS = {
    "h264_videotoolbox",
    "hevc_videotoolbox",
    "h264_nvenc",
    "hevc_nvenc",
}

# Keys read by BaseProblem or by the problem, grid, and solver code that
# consumes these sections. Unknown keys inside these sections warn.
# ``initial_conditions`` is problem-specific and is not checked here.
_CONFIG_KEYS = {
    "grid": {
        "N",
        "Nx",
        "Ny",
        "Nz",
        "Lx",
        "Ly",
        "Lz",
        "x0",
        "dealias",
        "precision",
        "basis",
        "basis_x",
        "basis_y",
        "bc",
        "bc_values",
        "bc_config",
        "fft_workers",
    },
    "integration": {
        "t0",
        "t_end",
        "cfl",
        "scheme",
        "output_interval",
        "checkpoint_interval",
        "spectral_filter",
        "spectral_dissipation",
        "modal_filter",
        "sbp_dissipation",
        "adaptive",
    },
    "physics": {"gamma", "mhd"},
    "equations": {"gamma"},
    "io": {"outdir"},
    "monitoring": {
        "enabled",
        "step_interval",
        "output_file",
        "include_conservation",
        "include_timestep",
        "include_time",
        "include_velocity_stats",
        "include_density_stats",
    },
    "video": {
        "enabled",
        "fps",
        "width",
        "height",
        "scale",
        "codec",
        "quality",
        "crf",
        "preset",
        "pix_fmt",
        "bitrate",
        "maxrate",
        "bufsize",
        "tune",
        "profile",
        "level",
        "frame_dpi",
        "projection_type",
        "column_axis",
        "column_thickness",
        "slice_axis",
        "slice_position",
        "colorbar_scale",
        "colorbar_min",
        "colorbar_max",
        "colorbar_fixed",
    },
    "runtime": {"num_threads"},
    "adaptive": {
        "enabled",
        "scheme",
        "rtol",
        "atol",
        "safety_factor",
        "min_dt_factor",
        "max_dt_factor",
        "max_rejections",
        "fallback_scheme",
    },
    "artificial_viscosity": {
        "enabled",
        "mode",
        "nu_constant",
        "nu_max",
        "s_ref",
        "s_min",
        "p",
        "epsilon",
        "variable_weights",
        "sensor_variable",
        "diagnostic_output",
    },
    "initial_conditions_smoothing": {
        "enabled",
        "kernel_size",
        "mode",
        "variables",
        "diagnostic_output",
    },
    "gravity": {"enabled", "type", "gx", "gy", "gz"},
    "tracers": {
        "enabled",
        "num",
        "layout",
        "seed",
        "mass",
        "plot_density",
        "density_grid",
    },
}
_NESTED_CONFIG_KEYS = {
    ("integration", "spectral_dissipation"): {
        "enabled",
        "p",
        "order",
        "rate",
        "e_folding_time_at_cutoff",
        "onset_fraction",
        "alpha",
        "interval",
    },
    ("integration", "modal_filter"): {"enabled", "p", "order", "alpha"},
    ("integration", "sbp_dissipation"): {"enabled", "strength"},
    ("integration", "adaptive"): _CONFIG_KEYS["adaptive"],
    ("physics", "mhd"): {
        "resistivity",
        "viscosity",
        "project_each_substage",
        "divergence_clean",
        "rho_min",
        "p_min",
        "ambipolar",
        "hall",
        "shearing_sheet",
        "resistivity_max",
        "cfl_diff",
    },
    ("physics", "mhd", "ambipolar"): {
        "enabled",
        "eta",
        "alpha",
        "method",
        "preconditioner",
        "midpoint_preconditioner",
        "midpoint_tol",
        "midpoint_maxiter",
        "eta_max",
    },
    ("physics", "mhd", "hall"): {
        "enabled",
        "eta_h",
        "eta_h_form",
        "b0",
        "rho0",
        "eta_max",
        "method",
        "safety",
        "length",
        "length_mode",
        "length_ratio",
        "cg_tol",
        "cg_maxiter",
        "hyper",
        "c_tilde",
        "c_factor",
    },
    ("physics", "mhd", "shearing_sheet"): {
        "enabled",
        "omega",
        "shear",
        "q",
        "vertical_axis",
        "radial_axis",
    },
}
_PROBLEM_MAPPING_KEYS = {"name", "description"}


def _warn_unknown_config_keys(config: Dict[str, Any]) -> None:
    """Warn when a parsed section contains a key the code does not read."""
    if not isinstance(config, dict):
        return
    for section, allowed in _CONFIG_KEYS.items():
        block = config.get(section)
        if not isinstance(block, dict):
            continue
        for key in sorted(set(block) - allowed):
            warnings.warn(
                f"Unknown configuration key '{section}.{key}' is ignored",
                UserWarning,
                stacklevel=3,
            )
    for path, allowed in _NESTED_CONFIG_KEYS.items():
        block: Any = config
        for part in path:
            if not isinstance(block, dict):
                block = None
                break
            block = block.get(part)
        if not isinstance(block, dict):
            continue
        prefix = ".".join(path)
        for key in sorted(set(block) - allowed):
            warnings.warn(
                f"Unknown configuration key '{prefix}.{key}' is ignored",
                UserWarning,
                stacklevel=3,
            )
    problem = config.get("problem")
    if isinstance(problem, dict):
        for key in sorted(set(problem) - _PROBLEM_MAPPING_KEYS):
            warnings.warn(
                f"Unknown configuration key 'problem.{key}' is ignored",
                UserWarning,
                stacklevel=3,
            )


def _pressure_field(primitive_vars: Any) -> Any:
    """Return the pressure array for Euler (1D/2D/3D) and MHD primitives.

    1D Euler primitives are ``(rho, u, p, a)``. 2D Euler primitives are
    ``(rho, ux, uy, p)``. 3D Euler and MHD put pressure at index 4.
    """
    count = len(primitive_vars)
    rho = primitive_vars[0]
    if count in (5, 8):
        return primitive_vars[4]
    if count == 4:
        if getattr(rho, "ndim", 0) == 1:
            return primitive_vars[2]
        return primitive_vars[3]
    return None


def _field_minmax(values: Any) -> tuple:
    if torch is not None and isinstance(values, torch.Tensor):
        return float(torch.max(values).item()), float(torch.min(values).item())
    return float(np.max(values)), float(np.min(values))


def _restart_values_match(current: Any, stored: Any) -> bool:
    if isinstance(current, dict) or isinstance(stored, dict):
        return current == stored
    if isinstance(current, bool) or isinstance(stored, bool):
        return current == stored
    if isinstance(current, (int, float)) and isinstance(stored, (int, float)):
        scale = max(1.0, abs(float(current)), abs(float(stored)))
        return abs(float(current) - float(stored)) <= 1e-12 * scale
    return current == stored


def _codec_rate_args(codec: str, video_config: Dict[str, Any]) -> list:
    """ffmpeg rate-control flags taken from the resolved video config."""
    args = []
    crf = video_config.get("crf")
    bitrate = video_config.get("bitrate")
    if codec == "mpeg4":
        quality = str(video_config.get("quality", "high"))
        quantizer = {"high": "1", "medium": "2", "low": "5"}.get(quality, "2")
        args.extend(["-q:v", quantizer])
    elif codec in _CRF_CODECS and crf is not None:
        args.extend(["-crf", str(int(crf))])
    if codec == "libvpx-vp9" and not bitrate:
        args.extend(["-b:v", "0"])
    if bitrate:
        args.extend(["-b:v", str(bitrate)])
    elif codec in _HARDWARE_CODECS:
        quality = str(video_config.get("quality", "high"))
        args.extend(["-b:v", _QUALITY_BITRATE.get(quality, "10M")])
    preset = video_config.get("preset")
    if preset and codec in _PRESET_CODECS:
        args.extend(["-preset", str(preset)])
    tune = video_config.get("tune")
    if tune and codec in _TUNE_CODECS:
        args.extend(["-tune", str(tune)])
    if video_config.get("maxrate"):
        args.extend(["-maxrate", str(video_config["maxrate"])])
    if video_config.get("bufsize"):
        args.extend(["-bufsize", str(video_config["bufsize"])])
    if video_config.get("profile") and codec not in {"mpeg4", "libvpx-vp9"}:
        args.extend(["-profile:v", str(video_config["profile"])])
    if video_config.get("level") and codec not in {"mpeg4", "libvpx-vp9"}:
        args.extend(["-level", str(video_config["level"])])
    return args

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover - Torch is optional
    torch = None  # type: ignore


class BaseProblem(ABC):
    """Base class for all PHRIKE problems."""

    @staticmethod
    def clear_numba_cache(root: Optional[Path] = None) -> None:
        """Remove this package's Numba caches.

        Numba writes compiled caches under ``__pycache__``. PHRIKE does not
        use ``~/.numba_cache``, and that directory is left untouched.
        Individual removal failures are reported instead of discarded.

        Args:
            root: Package directory to scan. Defaults to the ``phrike`` package.
        """
        package_root = Path(root) if root is not None else Path(__file__).resolve().parent.parent
        failures = []
        if not package_root.exists():
            warnings.warn(
                f"Numba cache directory does not exist: {package_root}",
                UserWarning,
                stacklevel=2,
            )
            return
        for pycache in package_root.rglob("__pycache__"):
            if not pycache.is_dir():
                continue
            try:
                shutil.rmtree(pycache)
            except OSError as exc:
                failures.append(f"{pycache}: {exc}")
        if failures:
            warnings.warn(
                "Could not remove every Numba cache directory: " + "; ".join(failures),
                UserWarning,
                stacklevel=2,
            )

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

    def _problem_label(self) -> str:
        """Registered or configured problem name, never a mapping's repr."""
        raw = self.config.get("problem", "unknown")
        if isinstance(raw, dict):
            nested = raw.get("name")
            if isinstance(nested, str) and nested.strip():
                return nested.strip()
            return "unknown"
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return "unknown"

    def setup_output_directory(self) -> None:
        """Setup output directory from configuration."""
        if "io" in self.config and "outdir" in self.config["io"]:
            self.outdir = os.path.abspath(self.config["io"]["outdir"])
        else:
            self.outdir = os.path.abspath(f"outputs/{self._problem_label()}")

        ensure_outdir(self.outdir)

    def setup_common_parameters(self) -> None:
        """Extract common parameters from configuration."""
        _warn_unknown_config_keys(self.config)
        # Support both "equations" and "physics" for backward compatibility
        if "equations" in self.config and "gamma" in self.config["equations"]:
            self.gamma = float(self.config["equations"]["gamma"])
        elif "physics" in self.config and "gamma" in self.config["physics"]:
            self.gamma = float(self.config["physics"]["gamma"])
        else:
            raise KeyError("Configuration must contain either 'equations.gamma' or 'physics.gamma'")

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
        integration = self.config["integration"]
        if "spectral_filter" in integration:
            raise ValueError(
                "integration.spectral_filter has been removed; use "
                "integration.spectral_dissipation for Fourier grids or "
                "integration.modal_filter for Legendre grids"
            )
        self.filter_config = {
            "spectral_dissipation": integration.get(
                "spectral_dissipation", {"enabled": False}
            ),
            "modal_filter": integration.get(
                "modal_filter", {"enabled": False}
            ),
            "sbp_dissipation": integration.get(
                "sbp_dissipation", {"enabled": False}
            ),
        }
        
        # Grid parameters
        self.precision = str(self.config["grid"].get("precision", "double"))
        
        # Adaptive time-stepping parameters.
        # A nested integration.adaptive block and a top-level adaptive block
        # are both accepted, but not together: the nested value used to hide
        # the top-level one.
        has_nested_adaptive = "adaptive" in self.config["integration"]
        has_top_adaptive = "adaptive" in self.config
        if has_nested_adaptive and has_top_adaptive:
            raise ValueError(
                "Both integration.adaptive and top-level adaptive are set. "
                "Keep only one of them."
            )
        if has_nested_adaptive:
            adaptive_raw = self.config["integration"].get("adaptive")
        else:
            adaptive_raw = self.config.get("adaptive")
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
                "mode": str(artificial_viscosity_raw.get("mode", "sensor")),
                "nu_constant": float(artificial_viscosity_raw.get("nu_constant", 0.0)),
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

        # Gravity parameters
        gravity_raw = self.config.get("gravity", None)
        if gravity_raw:
            self.gravity_config = {
                "enabled": bool(gravity_raw.get("enabled", False)),
                "type": str(gravity_raw.get("type", "linear")),
                "gx": float(gravity_raw.get("gx", 0.0)),
                "gy": float(gravity_raw.get("gy", 0.0)),
                "gz": float(gravity_raw.get("gz", 0.0))
            }
        else:
            self.gravity_config = None

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
                except (TypeError, ValueError):
                    warnings.warn(
                        f"runtime.num_threads={num_threads_cfg!r} is not an integer; "
                        "using the CPU count",
                        UserWarning,
                        stacklevel=2,
                    )
                    threads = os.cpu_count() or 1
        else:
            threads = None

        grid_threads = self.config["grid"].get("fft_workers", None)
        if grid_threads is not None:
            try:
                self.fft_workers = int(grid_threads)
            except (TypeError, ValueError):
                warnings.warn(
                    f"grid.fft_workers={grid_threads!r} is not an integer; "
                    "using the CPU count",
                    UserWarning,
                    stacklevel=2,
                )
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
        raw_interval = self.monitoring_config.get("step_interval", 10)
        if isinstance(raw_interval, bool):
            raise ValueError(
                "monitoring.step_interval must be a positive integer, "
                f"got {raw_interval!r}"
            )
        try:
            interval_value = float(raw_interval)
            self.monitoring_step_interval = int(interval_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "monitoring.step_interval must be a positive integer"
            ) from exc
        if interval_value != self.monitoring_step_interval or self.monitoring_step_interval < 1:
            raise ValueError(
                "monitoring.step_interval must be a positive integer, "
                f"got {raw_interval!r}"
            )
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
        """Raise if a restart snapshot was written for a different problem setup.

        Domain length, basis, boundary condition, dimension, and gamma are
        checked when the snapshot recorded them. Snapshots that predate those
        fields still load; the missing fields produce a warning. A gamma
        mismatch is an error.
        """
        if self.restart_data is None:
            return

        meta = self.restart_data.get("meta") or {}
        if not isinstance(meta, dict):
            warnings.warn(
                "Restart snapshot has no metadata dictionary; "
                "domain and physics settings were not checked",
                UserWarning,
                stacklevel=2,
            )
            return

        current = grid_snapshot_identity(grid, equations)
        compared = ["dimension", "basis", "bc", "gamma", "Lx"]
        if current["dimension"] >= 2:
            compared.append("Ly")
        if current["dimension"] == 3:
            compared.append("Lz")
        if "bc_config" in current or "bc_config" in meta:
            compared.append("bc_config")

        missing = []
        mismatches = []
        for key in compared:
            if key not in meta:
                missing.append(key)
                continue
            current_value = current.get(key)
            if not _restart_values_match(current_value, meta[key]):
                mismatches.append(
                    f"{key}: snapshot={meta[key]!r}, current={current_value!r}"
                )
        if mismatches:
            raise ValueError(
                "Restart snapshot is incompatible with the current problem: "
                + "; ".join(mismatches)
            )
        if missing:
            warnings.warn(
                "Restart snapshot is missing "
                + ", ".join(missing)
                + "; those settings were not checked. Re-save the snapshot "
                "with this version of PHRIKE to record them.",
                UserWarning,
                stacklevel=2,
            )

        dimension = current["dimension"]
        if "N" in meta and "Nx" not in meta:
            if dimension != 1:
                raise ValueError(
                    "Restart snapshot is 1D but the current grid is "
                    f"{dimension}D"
                )
            if meta["N"] != grid.N:
                raise ValueError(
                    f"Grid size mismatch - restart: {meta['N']}, current: {grid.N}"
                )
        elif "Nx" in meta:
            if dimension == 1:
                raise ValueError(
                    "Restart snapshot is multi-dimensional but the current grid is 1D"
                )
            if meta["Nx"] != grid.Nx:
                raise ValueError(
                    f"Grid Nx mismatch - restart: {meta['Nx']}, current: {grid.Nx}"
                )
            if "Ny" in meta and meta["Ny"] != grid.Ny:
                raise ValueError(
                    f"Grid Ny mismatch - restart: {meta['Ny']}, current: {grid.Ny}"
                )
            if "Nz" in meta and meta["Nz"] != grid.Nz:
                raise ValueError(
                    f"Grid Nz mismatch - restart: {meta['Nz']}, current: {grid.Nz}"
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

    def on_output(self, solver, t: float, U) -> None:
        """Hook called at every output time, before the snapshot is written.

        The default does nothing. Problems override it to record diagnostics
        that need every output rather than only the final state.
        """

    @abstractmethod
    def create_visualization(self, solver, t: float, U):
        """Create visualization for current state."""
        pass

    def create_final_visualization(self, solver):
        """Create final visualization plots."""
        pass

    def setup_video_generation(self) -> Dict[str, Any]:
        """Setup video generation parameters from config.

        An explicit ``crf`` wins over the quality preset. When ``crf`` is
        omitted, ``quality`` selects the constant-rate-factor (18/23/28).
        """
        video_config = self.config.get("video", {}) or {}
        quality = str(video_config.get("quality", "high"))
        if video_config.get("crf") is not None:
            crf = int(video_config["crf"])
        else:
            crf = _QUALITY_CRF.get(quality, 23)
        return {
            "fps": int(video_config.get("fps", 30)),
            "width": int(video_config.get("width", 0)),  # 0 = auto
            "height": int(video_config.get("height", 0)),  # 0 = auto
            "scale": float(video_config.get("scale", 1.0)),
            "codec": str(video_config.get("codec", "libx264")),
            "quality": quality,
            "crf": crf,
            "preset": video_config.get("preset", "medium"),
            "pix_fmt": str(video_config.get("pix_fmt", "yuv420p")),
            "bitrate": video_config.get("bitrate") or None,
            "maxrate": video_config.get("maxrate") or None,
            "bufsize": video_config.get("bufsize") or None,
            "tune": video_config.get("tune") or None,
            "profile": video_config.get("profile", "high"),
            "level": video_config.get("level", "4.1"),
        }

    def _prepare_frames_directory(self) -> str:
        """Return a frames directory that contains only this run's images.

        Problem visualizers write ``outdir/frames/frame_*.png``. Removing that
        directory at the start of a video run keeps a rerun from mixing its
        frames with the previous numbering.
        """
        frames_dir = os.path.join(self.outdir, "frames")
        if os.path.isdir(frames_dir):
            existing = [
                name
                for name in os.listdir(frames_dir)
                if name.startswith("frame_") and name.endswith(".png")
            ]
            if existing:
                warnings.warn(
                    f"Removing {len(existing)} existing frame(s) in {frames_dir} "
                    "so this video does not reuse frames from a previous run",
                    UserWarning,
                    stacklevel=2,
                )
            shutil.rmtree(frames_dir)
        ensure_outdir(frames_dir)
        return frames_dir

    def _renumber_frames(self, frames_dir: str) -> int:
        """Rename ``frame_*.png`` to ``frame_%08d.png`` through a staging directory.

        Renaming inside the destination directory overwrites an older numbered
        frame when a time-stamped name sorts first. The staging directory
        keeps every source file.
        """
        frames = sorted(
            name
            for name in os.listdir(frames_dir)
            if name.startswith("frame_") and name.endswith(".png")
        )
        if not frames:
            return 0
        staging = frames_dir.rstrip(os.sep) + ".renumbering"
        if os.path.isdir(staging):
            shutil.rmtree(staging)
        os.makedirs(staging)
        for index, name in enumerate(frames):
            os.replace(
                os.path.join(frames_dir, name),
                os.path.join(staging, f"frame_{index:08d}.png"),
            )
        for name in list(os.listdir(frames_dir)):
            path = os.path.join(frames_dir, name)
            if os.path.isfile(path):
                os.remove(path)
        for name in sorted(os.listdir(staging)):
            os.replace(os.path.join(staging, name), os.path.join(frames_dir, name))
        os.rmdir(staging)
        return len(frames)

    def _ffmpeg_command(
        self,
        frames_dir: str,
        video_path: str,
        video_config: Dict[str, Any],
        codec: str,
    ) -> list:
        command = [
            "ffmpeg",
            "-y",
            "-framerate",
            str(video_config["fps"]),
            "-i",
            os.path.join(frames_dir, _FRAME_SEQUENCE),
            "-c:v",
            codec,
        ]
        command.extend(_codec_rate_args(codec, video_config))
        command.extend(["-pix_fmt", str(video_config["pix_fmt"])])
        scale = video_config.get("scale", 1.0)
        if scale not in (1, 1.0):
            command.extend(["-vf", f"scale=iw*{scale}:ih*{scale}"])
        elif int(video_config.get("width") or 0) > 0 and int(video_config.get("height") or 0) > 0:
            command.extend(
                [
                    "-vf",
                    f"scale={int(video_config['width'])}:{int(video_config['height'])}",
                ]
            )
        command.append(video_path)
        return command

    def generate_video(self, frames_dir: str, video_name: str) -> Optional[str]:
        """Encode ``frames_dir`` to ``{video_name}.mp4``.

        Returns the video path when ffmpeg writes a movie, and None when
        ffmpeg is missing or every codec fails. The frames are left on disk
        either way. The requested codec is tried first, with its crf, preset,
        bitrate, and related flags; the other codecs are fallbacks.
        """
        video_config = self.setup_video_generation()
        video_path = os.path.join(self.outdir, f"{video_name}.mp4")
        n_frames = self._renumber_frames(frames_dir)
        print(f"Found {n_frames} frames")
        if n_frames == 0:
            warnings.warn(
                f"No frames in {frames_dir}; video was not created",
                UserWarning,
                stacklevel=2,
            )
            return None

        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            warnings.warn(
                "ffmpeg was not found, so the movie was not created. "
                f"Frames are in {frames_dir}",
                UserWarning,
                stacklevel=2,
            )
            return None

        requested = str(video_config["codec"])
        fallbacks = [
            "libx264",
            "h264_videotoolbox",
            "h264_nvenc",
            "mpeg4",
            "libvpx-vp9",
        ]
        codecs = [requested]
        for name in fallbacks:
            if name not in codecs:
                codecs.append(name)

        print(f"Creating movie: {video_path}")
        print(
            f"Using {n_frames} frames at {video_config['fps']} fps, "
            f"codec {requested}, crf {video_config['crf']}"
        )

        manual = (
            f"ffmpeg -framerate {video_config['fps']} -i "
            f"{frames_dir}/{_FRAME_SEQUENCE} -c:v {requested} "
            f"-crf {video_config['crf']} -pix_fmt {video_config['pix_fmt']} "
            f"{video_path}"
        )
        for codec in codecs:
            command = self._ffmpeg_command(frames_dir, video_path, video_config, codec)
            print(f"Trying {codec}...")
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=300
                )
            except subprocess.TimeoutExpired:
                print(f"Timeout with {codec}")
                continue
            except OSError as exc:
                warnings.warn(
                    f"Could not run ffmpeg for codec {codec}: {exc}",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            if result.returncode == 0:
                print(f"Movie created successfully using {codec}: {video_path}")
                return video_path
            print(f"Failed with {codec}: {result.stderr}")
            if codec == requested:
                warnings.warn(
                    f"Requested video codec {requested!r} failed; "
                    "trying fallback encoders",
                    UserWarning,
                    stacklevel=2,
                )

        warnings.warn(
            "All video encoders failed. Frames are available for manual "
            f"processing: {manual}",
            UserWarning,
            stacklevel=2,
        )
        print("All encoding options failed. Frames are available for manual processing.")
        print("You can try creating the movie manually with:")
        print(manual)
        return None

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
        """Compute conservation errors relative to initial values.

        Fourier grids use plain grid sums. Chebyshev and Legendre 1D grids use
        quadrature-weighted integrals, matching the solver history.
        """
        if self.monitoring_initial_values is None:
            return {}

        initial_values = self.monitoring_initial_values
        scales = getattr(self, "monitoring_momentum_scales", None) or {}

        # Get current conserved quantities
        if nodal_quadrature_weights(solver.grid, U) is not None:
            current_cons = conserved_diagnostics(solver.grid, solver.equations, U)
        elif getattr(solver.grid, "_mixed_open_y", False):
            current_cons = solver.grid.conserved_quantities(U, solver.equations)
        elif hasattr(solver.equations, "conserved_quantities"):
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
                    
                    # Check if we have quadrature weights for Legendre basis
                    wx = getattr(solver.grid, 'wx', None)
                    if wx is not None:
                        # Use quadrature weights for proper integration
                        if is_torch:
                            import torch
                            wx_tensor = torch.from_numpy(wx).to(dtype=rho.dtype, device=rho.device)
                            current_cons = {
                                "mass": float(torch.sum(rho * wx_tensor).item()),
                                "momentum": float(torch.sum(rho * u * wx_tensor).item()),
                                "energy": float(
                                    torch.sum(
                                        (p / (solver.equations.gamma - 1.0) + 0.5 * rho * u**2) * wx_tensor
                                    ).item()
                                ),
                            }
                        else:
                            current_cons = {
                                "mass": float((rho * wx).sum()),
                                "momentum": float((rho * u * wx).sum()),
                                "energy": float(
                                    ((p / (solver.equations.gamma - 1.0) + 0.5 * rho * u**2) * wx).sum()
                                ),
                            }
                    else:
                        # Fallback to simple sum for Fourier basis
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
                    
                    # Check if we have quadrature weights for Legendre basis
                    wx = getattr(solver.grid, 'wx', None)
                    wy = getattr(solver.grid, 'wy', None)
                    if wx is not None and wy is not None:
                        # Use 2D quadrature weights for proper integration
                        if is_torch:
                            import torch
                            wx_tensor = torch.from_numpy(wx).to(dtype=rho.dtype, device=rho.device)
                            wy_tensor = torch.from_numpy(wy).to(dtype=rho.dtype, device=rho.device)
                            # Create 2D weight matrix
                            w2d = wy_tensor.unsqueeze(-1) * wx_tensor.unsqueeze(-2)
                            current_cons = {
                                "mass": float(torch.sum(rho * w2d).item()),
                                "momentum_x": float(torch.sum(rho * ux * w2d).item()),
                                "momentum_y": float(torch.sum(rho * uy * w2d).item()),
                                "energy": float(
                                    torch.sum(
                                        (p / (solver.equations.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)) * w2d
                                    ).item()
                                ),
                            }
                        else:
                            # Create 2D weight matrix
                            w2d = wy[:, np.newaxis] * wx[np.newaxis, :]
                            current_cons = {
                                "mass": float((rho * w2d).sum()),
                                "momentum_x": float((rho * ux * w2d).sum()),
                                "momentum_y": float((rho * uy * w2d).sum()),
                                "energy": float(
                                    ((p / (solver.equations.gamma - 1.0) + 0.5 * rho * (ux**2 + uy**2)) * w2d).sum()
                                ),
                            }
                    else:
                        # Fallback to simple sum for Fourier basis
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

        # Mass and energy stay relative to their initial values. Momentum is
        # normalized by a physical scale: a signed initial momentum is often
        # only round-off, and dividing by it reports a meaningless ratio.
        errors = {}
        for key, initial_val in initial_values.items():
            if key not in current_cons:
                continue
            change = current_cons[key] - initial_val
            if key.startswith("momentum"):
                scale = scales.get(key, 0.0)
                if not np.isfinite(scale) or scale <= 0.0:
                    warnings.warn(
                        f"No physical scale for {key}; reporting the absolute change",
                        UserWarning,
                        stacklevel=2,
                    )
                    errors[f"{key}_error"] = abs(change)
                else:
                    errors[f"{key}_error"] = abs(change) / scale
            elif initial_val != 0:
                errors[f"{key}_error"] = change / initial_val
            else:
                errors[f"{key}_error"] = change

        return errors

    def _momentum_reference_scales(self, solver, U) -> Dict[str, float]:
        """Scale for each momentum component.

        The scale is ``max(sum(|rho u_i|), mass * c_s)`` using the same
        summation as the conservation diagnostic (quadrature weights on 1D
        Chebyshev/Legendre grids and on the mixed Fourier/SBP grid, otherwise a
        plain sum). ``c_s`` is the rms
        sound speed ``sqrt(gamma p / rho)``.
        """
        primitive_vars = solver.equations.primitive(U)
        rho = primitive_vars[0]
        is_torch = torch is not None and isinstance(rho, torch.Tensor)
        pressure = _pressure_field(primitive_vars)
        if pressure is None:
            raise ValueError(
                "Cannot scale momentum errors for "
                f"{len(primitive_vars)} primitive fields"
            )
        if len(primitive_vars) == 4 and getattr(rho, "ndim", 0) == 1:
            velocities = [primitive_vars[1]]
            names = ["momentum"]
        elif len(primitive_vars) == 4:
            velocities = [primitive_vars[1], primitive_vars[2]]
            names = ["momentum_x", "momentum_y"]
        else:
            velocities = [primitive_vars[1], primitive_vars[2], primitive_vars[3]]
            names = ["momentum_x", "momentum_y", "momentum_z"]

        weight = nodal_quadrature_weights(solver.grid, U)
        if weight is None and getattr(solver.grid, "_mixed_open_y", False):
            try:
                _, weights = solver.grid._sbp_arrays(U)
                weight = solver.grid.dx * np.asarray(weights)[:, None]
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                warnings.warn(
                    f"Momentum scale is unweighted; SBP weights were unavailable ({exc})",
                    UserWarning,
                    stacklevel=2,
                )
                weight = None
        if weight is not None and is_torch and not isinstance(weight, torch.Tensor):
            weight = torch.as_tensor(weight, dtype=rho.dtype, device=rho.device)

        def total(values: Any, absolute: bool = False) -> float:
            data = torch.abs(values) if absolute and is_torch else values
            if absolute and not is_torch:
                data = np.abs(values)
            if weight is not None:
                data = data * weight
            if is_torch:
                return float(torch.sum(data).item())
            return float(np.sum(data))

        mass = abs(total(rho))
        if is_torch:
            safe_rho = torch.clamp(rho, min=torch.finfo(rho.dtype).tiny)
            c2 = solver.equations.gamma * pressure / safe_rho
            finite = torch.isfinite(c2) & (c2 > 0)
            if bool(torch.any(finite).item()):
                c_ref = float(torch.sqrt(torch.mean(c2[finite])).item())
            else:
                c_ref = 0.0
        else:
            rho_np = np.asarray(rho)
            c2 = np.asarray(solver.equations.gamma * pressure / np.maximum(rho_np, 1e-300))
            finite = np.isfinite(c2) & (c2 > 0)
            c_ref = float(np.sqrt(np.mean(c2[finite]))) if np.any(finite) else 0.0
        mass_scale = mass * c_ref

        scales: Dict[str, float] = {}
        for name, velocity in zip(names, velocities):
            scales[name] = max(total(rho * velocity, absolute=True), mass_scale)
        if "momentum" not in scales and "momentum_x" in scales:
            # 2D Euler's conserved_quantities currently reports one "momentum" key.
            scales["momentum"] = scales["momentum_x"]
        return scales

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
        elif len(primitive_vars) == 8:  # MHD: rho, ux, uy, uz, p, Bx, By, Bz
            rho, ux, uy, uz, p, Bx, By, Bz = primitive_vars
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
            stats = {
                "rho_mean": float(torch.mean(rho).item()),
                "rho_max": float(torch.max(rho).item()),
                "rho_min": float(torch.min(rho).item()),
                "rho_std": float(torch.std(rho).item()),
            }
        else:
            stats = {
                "rho_mean": float(np.mean(rho)),
                "rho_max": float(np.max(rho)),
                "rho_min": float(np.min(rho)),
                "rho_std": float(np.std(rho)),
            }
        pressure = _pressure_field(primitive_vars)
        if pressure is None:
            warnings.warn(
                "Pressure statistics need 4, 5, or 8 primitive fields, "
                f"got {len(primitive_vars)}",
                UserWarning,
                stacklevel=2,
            )
            return stats
        try:
            p_max, p_min = _field_minmax(pressure)
        except (RuntimeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"Pressure statistics were not computed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            return stats
        stats["p_max"] = p_max
        stats["p_min"] = p_min
        return stats

    def compute_mhd_stats(self, solver, U):
        """Compute MHD diagnostics for an 8-component state, else empty dict.

        Returns max|div B| (the headline constraint metric), b_rms,
        magnetic_energy, plasma_beta (= <p> / <B^2/2>), and cross_helicity.
        """
        primitive_vars = solver.equations.primitive(U)
        if len(primitive_vars) != 8:
            return {}
        rho, ux, uy, uz, p, Bx, By, Bz = primitive_vars
        try:
            import torch
            is_torch = isinstance(rho, torch.Tensor)
        except ImportError:
            is_torch = False

        try:
            max_div_b = float(solver.max_div_b(U))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"max|div B| was not computed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            max_div_b = float("nan")

        if getattr(solver.grid, "_mixed_open_y", False):
            _, weights = solver.grid._sbp_arrays(U)
            w = solver.grid.dx*weights[:, None]
            volume = solver.grid.Lx*solver.grid.Ly
            b2 = Bx*Bx+By*By+Bz*Bz
            mag_e = float((0.5*b2*w).sum())
            b_rms = (2*mag_e/volume)**0.5
            p_mean = float((p*w).sum())/volume
            bmag_mean = mag_e/volume
            cross_h = float(((ux*Bx+uy*By+uz*Bz)*w).sum())
        elif is_torch:
            b2 = Bx * Bx + By * By + Bz * Bz
            mag_e = float(torch.sum(0.5 * b2).item())
            b_rms = float(torch.sqrt(torch.mean(b2)).item())
            p_mean = float(torch.mean(p).item())
            bmag_mean = float(torch.mean(0.5 * b2).item())
            cross_h = float(torch.sum(ux * Bx + uy * By + uz * Bz).item())
        else:
            b2 = Bx * Bx + By * By + Bz * Bz
            mag_e = float((0.5 * b2).sum())
            b_rms = float(np.sqrt(np.mean(b2)))
            p_mean = float(np.mean(p))
            bmag_mean = float(np.mean(0.5 * b2))
            cross_h = float((ux * Bx + uy * By + uz * Bz).sum())
        beta = p_mean / bmag_mean if bmag_mean > 0 else float("inf")
        return {
            "max|divB|": max_div_b,
            "b_rms": b_rms,
            "magnetic_energy": mag_e,
            "plasma_beta": beta,
            "cross_helicity": cross_h,
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
            # If Legendre weights are available, compute weighted conserved quantities
            wy = getattr(solver.grid, "wy", None)
            wx = getattr(solver.grid, "wx", None)
            primitive_for_weights = solver.equations.primitive(U)
            rho_for_weights = primitive_for_weights[0]
            if (
                wy is not None
                and wx is not None
                and len(primitive_for_weights) == 4
                and getattr(rho_for_weights, "ndim", 0) == 2
            ):
                try:
                    rho, ux, uy, p = primitive_for_weights
                    import numpy as _np
                    if torch is not None and isinstance(rho, torch.Tensor):
                        wyt = torch.from_numpy(_np.asarray(wy)).to(rho.device, rho.dtype)
                        wxt = torch.from_numpy(_np.asarray(wx)).to(rho.device, rho.dtype)
                        w2d = wyt[:, None] * wxt[None, :]
                        mass_w = float(torch.sum(w2d * rho).item())
                        momx_w = float(torch.sum(w2d * rho * ux).item())
                        momy_w = float(torch.sum(w2d * rho * uy).item())
                        energy_w = float(torch.sum(w2d * (p / (solver.equations.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy))).item())
                    else:
                        w2d = _np.outer(wy, wx)
                        mass_w = float((w2d * rho).sum())
                        momx_w = float((w2d * rho * ux).sum())
                        momy_w = float((w2d * rho * uy).sum())
                        energy_w = float((w2d * (p / (solver.equations.gamma - 1.0) + 0.5 * rho * (ux * ux + uy * uy))).sum())
                    info_lines.append(f"mass_w: {mass_w:.6e}")
                    info_lines.append(f"momentum_x_w: {momx_w:.6e}")
                    info_lines.append(f"momentum_y_w: {momy_w:.6e}")
                    info_lines.append(f"energy_w: {energy_w:.6e}")
                except (RuntimeError, TypeError, ValueError) as exc:
                    warnings.warn(
                        f"Weighted conservation statistics were not computed: {exc}",
                        UserWarning,
                        stacklevel=2,
                    )
            errors = self.compute_conservation_errors(solver, U)
            for key, error in errors.items():
                if key.startswith("momentum") and key.endswith("_error"):
                    info_lines.append(
                        f"{key}: {error:.2e} ({_MOMENTUM_SCALE_NOTE})"
                    )
                else:
                    info_lines.append(f"{key}: {error:.2e}")

        if self.monitoring_include_velocity_stats:
            v_stats = self.compute_velocity_stats(solver, U)
            for key, value in v_stats.items():
                info_lines.append(f"{key}: {value:.6f}")

        if self.monitoring_include_density_stats:
            rho_stats = self.compute_density_stats(solver, U)
            for key, value in rho_stats.items():
                info_lines.append(f"{key}: {value:.6f}")

        # MHD diagnostics (only for 8-component states; headline is max|divB|)
        try:
            mhd_stats = self.compute_mhd_stats(solver, U)
        except (RuntimeError, TypeError, ValueError) as exc:
            warnings.warn(
                f"MHD monitoring statistics were not computed: {exc}",
                UserWarning,
                stacklevel=2,
            )
            mhd_stats = {}
        for key, value in mhd_stats.items():
            info_lines.append(f"{key}: {value:.3e}")

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
        if nodal_quadrature_weights(solver.grid, U) is not None:
            self.monitoring_initial_values = conserved_diagnostics(
                solver.grid, solver.equations, U
            )
        elif getattr(solver.grid, "_mixed_open_y", False):
            self.monitoring_initial_values = solver.grid.conserved_quantities(U, solver.equations)
        elif hasattr(solver.equations, "conserved_quantities"):
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

        self.monitoring_momentum_scales = self._momentum_reference_scales(solver, U)
        # Reset step counter
        self.monitoring_step_count = 0

    def _create_tracers_from_config(
        self, grid: Any, U0: Any, equations: Any
    ) -> Optional[Any]:
        """Create tracer particles from config when tracers.enabled and grid is Fourier.

        Returns FourierTracers1D, FourierTracers2D, or FourierTracers3D, or None.
        """
        tcfg = self.config.get("tracers", {})
        if not tcfg.get("enabled", False):
            return None
        is_3d = hasattr(grid, "Nz") and getattr(grid, "Nz", None) is not None
        is_2d = (
            not is_3d
            and hasattr(grid, "Ny")
            and getattr(grid, "Ny", None) is not None
        )
        if is_3d:
            basis = getattr(grid, "basis", "fourier")
            fourier_grid = str(basis).lower() == "fourier"
        elif is_2d:
            fourier_grid = (
                str(getattr(grid, "basis_x", "")).lower() == "fourier"
                and str(getattr(grid, "basis_y", "")).lower() == "fourier"
            )
        else:
            basis_name = getattr(grid, "_basis_name", None)
            if basis_name is None:
                basis_name = getattr(grid, "basis", "")
            fourier_grid = str(basis_name).lower() == "fourier"
        if not fourier_grid:
            raise ValueError(
                "Tracers are only supported on periodic Fourier grids. "
                "Disable tracers.enabled or select a Fourier basis."
            )
        num = int(tcfg.get("num", 100))
        if num <= 0:
            raise ValueError("tracers.num must be positive")
        layout = str(tcfg.get("layout", "random")).lower()
        if layout not in {"random", "uniform"}:
            raise ValueError("tracers.layout must be 'random' or 'uniform'")
        seed = tcfg.get("seed", None)
        rng = np.random.default_rng(seed)

        def tracer_mass(count: int) -> float:
            configured = tcfg.get("mass", 1.0)
            if configured is not None and str(configured).lower() != "auto":
                return float(configured)
            rho = equations.primitive(U0)[0]
            if torch is not None and isinstance(rho, torch.Tensor):
                mean_density = float(torch.mean(rho).item())
            else:
                mean_density = float(np.mean(rho))
            volume = float(grid.Lx)
            if hasattr(grid, "Ly"):
                volume *= float(grid.Ly)
            if hasattr(grid, "Lz"):
                volume *= float(grid.Lz)
            return mean_density * volume / count

        def restart_mass() -> float:
            assert self.restart_data is not None
            values = np.asarray(self.restart_data.get("tracer_mass", 1.0)).reshape(-1)
            if values.size != 1:
                raise ValueError("Restart tracer_mass must be a scalar")
            value = float(values[0])
            if not np.isfinite(value):
                raise ValueError("Restart tracer_mass must be finite")
            return value

        restart_tracer_keys = set()
        if self.restart_data is not None:
            restart_tracer_keys = {
                key for key in self.restart_data if key.startswith("tracer_")
            }
        if restart_tracer_keys and "tracer_x" not in restart_tracer_keys:
            raise ValueError("Tracer restart metadata is missing tracer_x")

        if self.restart_data is not None and "tracer_x" in self.restart_data:
            axes = "xyz" if is_3d else "xy" if is_2d else "x"
            missing = [
                axis for axis in axes if f"tracer_{axis}" not in self.restart_data
            ]
            if missing:
                raise ValueError(
                    "Tracer restart is missing position arrays: " + ", ".join(missing)
                )
            coordinates = {
                axis: np.asarray(
                    self.restart_data[f"tracer_{axis}"], dtype=np.float64
                ).reshape(-1)
                for axis in axes
            }
            count = coordinates["x"].size
            if count == 0:
                raise ValueError("Tracer restart contains no particles")
            if any(values.size != count for values in coordinates.values()):
                raise ValueError("Restart tracer coordinate lengths do not match")
            if any(
                not np.all(np.isfinite(values)) for values in coordinates.values()
            ):
                raise ValueError("Restart tracer coordinates must be finite")

            shape_raw = self.restart_data.get("tracer_lattice_shape")
            lattice_shape = None
            if shape_raw is not None:
                lattice_shape = tuple(
                    int(value) for value in np.asarray(shape_raw).reshape(-1)
                )
                if (
                    len(lattice_shape) != len(axes)
                    or any(value <= 0 for value in lattice_shape)
                    or int(np.prod(lattice_shape)) != count
                ):
                    raise ValueError(
                        "Restart tracer_lattice_shape does not match particle data"
                    )
            if is_3d:
                tracers = FourierTracers3D(
                    coordinates["x"],
                    coordinates["y"],
                    coordinates["z"],
                    mass=restart_mass(),
                    lattice_shape=lattice_shape,
                )
            elif is_2d:
                tracers = FourierTracers2D(
                    coordinates["x"],
                    coordinates["y"],
                    mass=restart_mass(),
                    lattice_shape=lattice_shape,
                )
            else:
                tracers = FourierTracers1D(coordinates["x"], mass=restart_mass())
            unwrapped_axes = [
                axis
                for axis in axes
                if f"tracer_{axis}_unwrapped" in self.restart_data
            ]
            if unwrapped_axes and len(unwrapped_axes) != len(axes):
                raise ValueError(
                    "Tracer restart must provide either all or no unwrapped axes"
                )
            for axis in axes:
                key = f"tracer_{axis}_unwrapped"
                if key not in self.restart_data:
                    continue
                values = np.asarray(self.restart_data[key], dtype=np.float64).reshape(-1)
                if values.size != count or not np.all(np.isfinite(values)):
                    raise ValueError(
                        f"Restart {key} must be finite and match particle count"
                    )
                setattr(tracers, f"{axis}_unwrapped", values.copy())
            return tracers

        # 3D
        if is_3d:
            Lx, Ly, Lz = grid.Lx, grid.Ly, grid.Lz
            if layout == "uniform":
                n = max(1, int(round(num ** (1.0 / 3.0))))
                if n**3 != num:
                    raise ValueError("Uniform 3-D tracers require tracers.num = n^3")
                x = np.linspace(0.0, Lx, n, endpoint=False)
                y = np.linspace(0.0, Ly, n, endpoint=False)
                z = np.linspace(0.0, Lz, n, endpoint=False)
                xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
                x0 = xx.ravel()
                y0 = yy.ravel()
                z0 = zz.ravel()
                lattice_shape = (n, n, n)
            else:
                x0 = rng.uniform(0.0, Lx, size=num).astype(np.float64)
                y0 = rng.uniform(0.0, Ly, size=num).astype(np.float64)
                z0 = rng.uniform(0.0, Lz, size=num).astype(np.float64)
                lattice_shape = None
            return FourierTracers3D(
                x0,
                y0,
                z0,
                mass=tracer_mass(len(x0)),
                lattice_shape=lattice_shape,
            )

        # 2D
        if is_2d:
            Lx, Ly = grid.Lx, grid.Ly
            if layout == "uniform":
                n = max(1, int(round(num ** 0.5)))
                if n**2 != num:
                    raise ValueError("Uniform 2-D tracers require tracers.num = n^2")
                x = np.linspace(0.0, Lx, n, endpoint=False)
                y = np.linspace(0.0, Ly, n, endpoint=False)
                xx, yy = np.meshgrid(x, y, indexing="ij")
                x0 = xx.ravel()
                y0 = yy.ravel()
                lattice_shape = (n, n)
            else:
                x0 = rng.uniform(0.0, Lx, size=num).astype(np.float64)
                y0 = rng.uniform(0.0, Ly, size=num).astype(np.float64)
                lattice_shape = None
            return FourierTracers2D(
                x0,
                y0,
                mass=tracer_mass(len(x0)),
                lattice_shape=lattice_shape,
            )

        # 1D
        Lx = grid.Lx
        if layout == "uniform":
            x0 = np.linspace(0.0, Lx, num, endpoint=False).astype(np.float64)
        else:
            x0 = rng.uniform(0.0, Lx, size=num).astype(np.float64)
        return FourierTracers1D(x0, mass=tracer_mass(len(x0)))

    def run(
        self,
        backend: str = "auto",
        device: Optional[str] = None,
        generate_video: bool = False,
        debug: bool = False,
    ) -> Any:
        """Run the simulation."""
        # Create components
        grid = self.create_grid(backend, device, debug)
        resolved_precision = getattr(grid, "precision", None)
        if resolved_precision:
            self.precision = str(resolved_precision)
        grid._phrike_provenance = {
            "config": self.config,
            "problem": self._problem_label(),
        }
        equations = self.create_equations()

        # Create initial conditions (use restart data if available)
        if self.restart_data is not None:
            U0 = self.restart_data["U"]
            print(f"Using restart data for initial conditions (shape: {U0.shape})")
            
            # Convert restart data to appropriate backend format
            if getattr(grid, "_use_torch", False) and isinstance(U0, np.ndarray):
                try:
                    import torch

                    # Match the grid's resolved backend, device, and precision.
                    U0 = torch.from_numpy(U0).to(
                        dtype=grid.x.dtype, device=grid.x.device
                    )
                    print(
                        "Converted restart data to torch tensor: "
                        f"dtype={grid.x.dtype}, device={grid.x.device}"
                    )
                except ImportError:
                    raise ImportError("PyTorch is required for torch backend but not available")
        else:
            U0 = self.create_initial_conditions(grid)

        # Validate restart data compatibility
        self.validate_restart_data(grid, equations)

        # Create solver
        solver_class = self.get_solver_class()
        solver = solver_class(
            grid=grid, equations=equations, scheme=self.scheme, cfl=self.cfl,
            adaptive_config=self.adaptive_config,
            artificial_viscosity_config=self.artificial_viscosity_config,
            gravity_config=self.gravity_config
        )

        # Setup visualization. Problem visualizers write outdir/frames, so a
        # video run starts by clearing that directory.
        frames_dir = os.path.join(self.outdir, "frames")
        if generate_video:
            frames_dir = self._prepare_frames_directory()

        # Initialize monitoring
        self.initialize_monitoring(solver, U0)

        # Create tracers from config if enabled (Fourier grids only)
        tracers = self._create_tracers_from_config(grid, U0, equations)
        if tracers is not None:
            tracers._ensure_backend(equations.primitive(U0)[1])
            n_tracers = len(tracers.x)
            print(f"Tracers enabled: {n_tracers} particles")

        # Run simulation
        if hasattr(solver, "run"):
            # New solver interface
            def visualization_callback(t, U):
                self.create_visualization(solver, t, U)

            def snapshot_callback(t, U):
                # Save snapshot independently of video generation
                snapshot_path = save_solution_snapshot(
                    self.outdir,
                    t,
                    U=U,
                    grid=solver.grid,
                    equations=solver.equations,
                    tracers=tracers,
                )
                print(f"Saved snapshot at t={t:.3f}: {snapshot_path}")

            def monitoring_callback(step_count, dt, U):
                if (
                    self.monitoring_enabled
                    and step_count % self.monitoring_step_interval == 0
                ):
                    self.output_monitoring_info(solver, U, step_count, dt)

            # Determine which callbacks to use
            on_output_callbacks = []
            if generate_video:
                on_output_callbacks.append(visualization_callback)
            on_output_callbacks.append(snapshot_callback)
            
            def combined_output_callback(t, U):
                self.on_output(solver, t, U)
                for callback in on_output_callbacks:
                    callback(t, U)

            history = solver.run(
                U0,
                t0=self.t0,
                t_end=self.t_end,
                output_interval=self.output_interval,
                checkpoint_interval=self.checkpoint_interval,
                outdir=self.outdir,
                on_output=combined_output_callback,
                on_step=monitoring_callback,
                tracers=tracers,
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

        # Some problem-specific final visualizers also write the final snapshot.
        # Write it once more with tracer state so those helpers cannot discard it.
        if tracers is not None:
            save_solution_snapshot(
                self.outdir,
                solver.t,
                U=solver.U,
                grid=solver.grid,
                equations=solver.equations,
                tracers=tracers,
            )

        # Tracer density plot if enabled (3D only)
        tcfg = self.config.get("tracers", {})
        if (
            tracers is not None
            and tcfg.get("plot_density", False)
            and hasattr(solver.grid, "Nz")
            and getattr(solver.grid, "Nz", None) is not None
        ):
            try:
                from phrike.tracer_density import tracer_density_3d, plot_tracer_density_3d
                domain = (solver.grid.Lx, solver.grid.Ly, solver.grid.Lz)
                layout = str(tcfg.get("layout", "random")).lower()
                grid_shape = tcfg.get("density_grid")
                if isinstance(grid_shape, (list, tuple)) and len(grid_shape) == 3:
                    grid_shape = tuple(int(g) for g in grid_shape)
                else:
                    grid_shape = None
                rho_3d, _ = tracer_density_3d(
                    tracers, domain, grid_shape=grid_shape, layout=layout
                )
                outpath = os.path.join(self.outdir, "tracer_density.png")
                plot_tracer_density_3d(
                    rho_3d, domain, outpath=outpath, log=True,
                    title="Tracer particle density (t={:.3f})".format(solver.t),
                )
                print(f"Saved tracer density plot: {outpath}")
            except (ImportError, OSError, RuntimeError, ValueError) as exc:
                warnings.warn(
                    f"Tracer density plot failed: {exc}",
                    UserWarning,
                    stacklevel=2,
                )

        # Generate video if requested
        if generate_video and os.path.isdir(frames_dir):
            self.generate_video(frames_dir, self._problem_label())

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
            warnings.warn(
                "scipy is not available; initial conditions were not smoothed",
                UserWarning,
                stacklevel=2,
            )
            return rho, u, p
        
        # Calculate sigma in grid points. 3D grids also have Nx, so test Nz first.
        if hasattr(grid, "Nz"):
            sigma = (
                kernel_size
                * (grid.Nx + grid.Ny + grid.Nz)
                / (grid.Lx + grid.Ly + grid.Lz)
                / 3
            )
        elif hasattr(grid, "Nx"):
            sigma = kernel_size * grid.Nx / grid.Lx
        elif hasattr(grid, "N"):
            sigma = kernel_size * grid.N / grid.Lx
        else:
            warnings.warn(
                "Cannot determine the grid size; initial conditions were not smoothed",
                UserWarning,
                stacklevel=2,
            )
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
