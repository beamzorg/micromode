import numpy as np

from micromode.yee import refine_x_mode_at_fixed_beta


def test_fixed_beta_refinement_returns_normalized_yee_shapes():
    nz, ny = 8, 9
    shapes = {
        "Ex": (nz, ny),
        "Ey": (nz, ny - 1),
        "Ez": (nz - 1, ny),
        "Hx": (nz - 1, ny - 1),
        "Hy": (nz - 1, ny),
        "Hz": (nz, ny - 1),
    }
    z = np.sin(np.pi * (np.arange(nz) + 0.5) / nz)[:, None]
    y = np.sin(np.pi * (np.arange(ny - 1) + 1.0) / ny)[None, :]
    profiles = {name: np.zeros(shape, dtype=np.complex128) for name, shape in shapes.items()}
    profiles["Ey"] = z * y
    profiles["Hz"] = profiles["Ey"] / 250.0
    indices = {name: (slice(None), slice(None)) for name in shapes}
    eps = {name: np.full(shape, 2.25) for name, shape in shapes.items() if name.startswith("E")}
    mu = {name: np.ones(shape) for name, shape in shapes.items() if name.startswith("H")}

    refined, residual, ratio, k_num, initial_ratio = refine_x_mode_at_fixed_beta(
        profiles,
        indices,
        component_permittivity=eps,
        component_permeability=mu,
        omega=2.0 * np.pi * 193.0e12,
        dt=8.0e-17,
        resolution=80.0e-9,
        k_num=1.1e7,
        direction_sign=1.0,
    )

    assert {name: value.shape for name, value in refined.items()} == shapes
    assert residual < 1e-8
    assert abs(ratio - 1.0) < abs(initial_ratio - 1.0)
    assert k_num > 0.0
    assert initial_ratio > 0.0
