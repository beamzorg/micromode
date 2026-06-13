"""Public package exports for the MicroMode Python API."""

from __future__ import annotations

from .beamz import DiscreteMode, ModePlaneSpec, solve_beamz_mode

# Re-export the small public API from package root so users do not need to know
# the internal module split.
from .constants import C_0, EPSILON_0
from .models import BoundarySpec, Grid, Materials, PmlSpec, Spec
from .raster import solve_grid, solve_modes, solve_slice
from .result import Result, overlap
from .sweep import Sweep, track_modes_by_overlap

# Keep __all__ explicit so documentation and static analysis show the intended
# public surface.
__all__ = [
    "C_0",
    "EPSILON_0",
    "BoundarySpec",
    "DiscreteMode",
    "Grid",
    "Materials",
    "ModePlaneSpec",
    "PmlSpec",
    "Result",
    "Spec",
    "Sweep",
    "overlap",
    "solve_beamz_mode",
    "solve_grid",
    "solve_modes",
    "solve_slice",
    "track_modes_by_overlap",
]
