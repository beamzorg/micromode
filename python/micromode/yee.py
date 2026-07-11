"""Fixed-beta refinement on BeamZ's x-normal Yee cross section."""

from __future__ import annotations

import numpy as np

_C0 = 299_792_458.0
_MU0 = 1.256_637_062_12e-6


def refine_x_mode_at_fixed_beta(
    profiles,
    indices,
    *,
    component_permittivity,
    component_permeability,
    omega,
    dt,
    resolution,
    k_num,
    direction_sign,
    _correct_beta=True,
):
    """Refine E at the selected beta, then reconstruct H from Faraday's law."""
    from scipy import sparse
    from scipy.sparse import linalg as spla

    out = {name: np.asarray(value, dtype=np.complex128) for name, value in profiles.items()}
    nz, ny = out["Ex"].shape
    d = float(resolution)
    dz = _forward_difference(sparse, nz, d)
    dy = _forward_difference(sparse, ny, d)
    bz, by = -dz.T, -dy.T
    iz = sparse.eye(nz, format="csc")
    iy = sparse.eye(ny, format="csc")
    izm = sparse.eye(nz - 1, format="csc")
    iym = sparse.eye(ny - 1, format="csc")

    dy_ez = sparse.kron(izm, dy, format="csc")
    dz_ey = sparse.kron(dz, iym, format="csc")
    dz_ex = sparse.kron(dz, iy, format="csc")
    dy_ex = sparse.kron(iz, dy, format="csc")
    by_hz = sparse.kron(iz, by, format="csc")
    bz_hy = sparse.kron(bz, iy, format="csc")
    bz_hx = sparse.kron(bz, iym, format="csc")
    by_hx = sparse.kron(izm, by, format="csc")

    sign = 1.0 if float(direction_sign) >= 0.0 else -1.0
    delta = -2j * sign * np.sin(0.5 * float(k_num) * d) / d
    ex_n, ey_n, ez_n = (out[name].size for name in ("Ex", "Ey", "Ez"))
    hx_n, hy_n, hz_n = (out[name].size for name in ("Hx", "Hy", "Hz"))
    z = sparse.csc_matrix
    curl_e = sparse.bmat(
        [
            [z((hx_n, ex_n)), -dz_ey, dy_ez],
            [dz_ex, z((hy_n, ey_n)), -delta * sparse.eye(ez_n)],
            [-dy_ex, delta * sparse.eye(ey_n), z((hz_n, ez_n))],
        ],
        format="csc",
    )
    curl_h = sparse.bmat(
        [
            [z((ex_n, hx_n)), -bz_hy, by_hz],
            [bz_hx, z((ey_n, hy_n)), -delta * sparse.eye(hz_n)],
            [-by_hx, delta * sparse.eye(hy_n), z((ez_n, hz_n))],
        ],
        format="csc",
    )

    mu = np.concatenate(
        [_material(component_permeability[name], indices[name], out[name].shape) for name in ("Hx", "Hy", "Hz")]
    )
    eps = np.concatenate(
        [_material(component_permittivity[name], indices[name], out[name].shape) for name in ("Ex", "Ey", "Ez")]
    )
    operator = curl_h @ sparse.diags(1.0 / mu) @ curl_e
    mass = sparse.diags(eps)
    omega_d = float(omega) if dt is None else 2.0 * np.sin(0.5 * float(omega) * float(dt)) / float(dt)
    target = (omega_d / _C0) ** 2
    seed = np.concatenate([out[name].reshape(-1) for name in ("Ex", "Ey", "Ez")])
    count = min(4, operator.shape[0] - 2)
    values, vectors = spla.eigs(operator, M=mass, k=count, sigma=target, v0=seed, tol=1e-9)
    overlaps = np.abs(np.conjugate(seed) @ vectors)
    selected = int(np.argmax(overlaps))
    electric = np.asarray(vectors[:, selected], dtype=np.complex128)
    phase = np.vdot(seed, electric)
    if abs(phase) > 1e-30:
        electric *= np.exp(-1j * np.angle(phase))
    ex, ey, ez = np.split(electric, (ex_n, ex_n + ey_n))
    out["Ex"], out["Ey"], out["Ez"] = ex.reshape(nz, ny), ey.reshape(nz, ny - 1), ez.reshape(nz - 1, ny)

    magnetic = 1j * (curl_e @ electric) / (_MU0 * omega_d * mu)
    hx, hy, hz = np.split(magnetic, (hx_n, hx_n + hy_n))
    out["Hx"], out["Hy"], out["Hz"] = hx.reshape(nz - 1, ny - 1), hy.reshape(nz - 1, ny), hz.reshape(nz, ny - 1)
    residual = np.linalg.norm(operator @ electric - values[selected] * (mass @ electric))
    residual /= max(np.linalg.norm(values[selected] * (mass @ electric)), np.finfo(float).eps)
    eigenvalue = float(np.real(values[selected]))
    frequency_ratio = float(np.sqrt(max(eigenvalue / target, 0.0)))
    frequency_ratio_initial = frequency_ratio
    if _correct_beta:
        q_axis = 2.0 * np.sin(0.5 * float(k_num) * d) / d
        # One Newton step in q². For the staggered vector operator the local
        # eigenvalue slope is approximately one half, hence the factor two.
        q_corrected = np.sqrt(max(q_axis**2 - 2.0 * (eigenvalue - target), 0.0))
        k_corrected = 2.0 * np.arcsin(np.clip(0.5 * q_corrected * d, -1.0, 1.0)) / d
        if np.isfinite(k_corrected) and k_corrected > 0.0:
            refined, residual, frequency_ratio, _, _ = refine_x_mode_at_fixed_beta(
                out,
                indices,
                component_permittivity=component_permittivity,
                component_permeability=component_permeability,
                omega=omega,
                dt=dt,
                resolution=resolution,
                k_num=k_corrected,
                direction_sign=direction_sign,
                _correct_beta=False,
            )
            return refined, residual, frequency_ratio, float(k_corrected), frequency_ratio_initial
    return out, float(residual), frequency_ratio, float(k_num), frequency_ratio


def _material(values, index, shape):
    arr = np.asarray(values, dtype=np.complex128)
    profile = arr[index] if arr.ndim == 3 else arr
    profile = np.asarray(profile, dtype=np.complex128).squeeze()
    if profile.shape != shape:
        raise ValueError(f"Material shape {profile.shape} does not match {shape}")
    return profile.reshape(-1)


def _forward_difference(sparse, count, spacing):
    rows = np.repeat(np.arange(count - 1), 2)
    cols = np.column_stack((np.arange(count - 1), np.arange(1, count))).reshape(-1)
    data = np.tile(np.asarray([-1.0, 1.0]) / spacing, count - 1)
    return sparse.csc_matrix((data, (rows, cols)), shape=(count - 1, count))
