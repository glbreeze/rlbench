import json
import trimesh
import numpy as np
from pyrep.objects.shape import Shape
from pyrep.objects.dummy import Dummy
from pyrep.const import PrimitiveShape

from pyrep.backend import sim
from pyrep.backend._sim_cffi import lib, ffi
from pyrep.backend.sim import _check_return

from typing import Any, Optional, Sequence, Union, Tuple, List, Dict
Number = Union[int, float]

# ---------- utilities  ---------- 
def _S(d: np.ndarray) -> np.ndarray:
    d = np.asarray(d, dtype=np.float64).reshape(3)
    return (d @ d) * np.eye(3) - np.outer(d, d)

def _get_T(handle: int) -> np.ndarray:
    T = np.asarray(sim.simGetObjectMatrix(handle, -1), dtype=np.float32).reshape(-1)
    assert T.size == 12
    return T

# ------------ sim wrappers  ------------ 

def simGetShapeMassAndInertia(handle: int, transf: Optional[np.ndarray] = None):
    h = int(handle)
    c_mass = ffi.new("float[1]")
    c_I = ffi.new("float[9]")
    c_com = ffi.new("float[3]")

    T = _get_T(handle) if transf is None else np.asarray(transf, np.float32).reshape(-1)
    c_T = ffi.new("float[12]", T.tolist())

    ret = lib.simGetShapeMassAndInertia(h, c_mass, c_I, c_com, c_T)
    _check_return(ret)

    mass = float(c_mass[0])
    I = np.array([float(c_I[i]) for i in range(9)], dtype=np.float64).reshape(3, 3)
    com = np.array([float(c_com[i]) for i in range(3)], dtype=np.float64)
    return mass, I, com, T  # return T used (handy)


def simSetShapeMassAndInertia(handle: int, mass: float, inertia9: np.ndarray, com3: np.ndarray,
                             transf: Optional[np.ndarray] = None):
    h = int(handle)
    I = np.asarray(inertia9, dtype=np.float32).reshape(3, 3)
    com = np.asarray(com3, dtype=np.float32).reshape(3)

    c_I = ffi.new("float[9]", I.reshape(-1).tolist())
    c_com = ffi.new("float[3]", com.tolist())

    T = _get_T(handle) if transf is None else np.asarray(transf, np.float32).reshape(-1)
    c_T = ffi.new("float[12]", T.tolist())

    ret = lib.simSetShapeMassAndInertia(h, float(mass), c_I, c_com, c_T)
    _check_return(ret)


# -------------- reset the Mass, CoM and Frictions -------------- 
def sample_point_inside_shape(shape, pitch=None, margin_ratio=0.05) -> np.ndarray:
    V, F, _ = shape.get_mesh_data()
    mesh = trimesh.Trimesh(vertices=V, faces=F, process=False)

    bmin, bmax = mesh.bounds
    extent = (bmax - bmin)
    if pitch is None:
        pitch = float(np.max(extent) / 80.0)
        pitch = max(pitch, 1e-4)

    vox = mesh.voxelized(pitch).fill()
    pts = vox.points

    margin = margin_ratio * extent
    lo = bmin + margin
    hi = bmax - margin
    mask = np.all((pts >= lo) & (pts <= hi), axis=1)
    pts = pts[mask]

    if len(pts) == 0:
        raise RuntimeError("No interior voxel points after margin filter; adjust pitch/margin.")
    # return pts[np.random.randint(len(pts))]
    d2 = np.sum(pts**2, axis=1)

    # index of farthest point
    idx = np.argmax(d2)

    # the point itself
    return pts[idx]


def add_point_mass_ballast_and_set(shape, ballast_mass: Number, margin_ratio=0.05):
    handle = shape.get_handle()
    mb = float(ballast_mass)
    if mb <= 0:
        raise ValueError(f"ballast_mass must be > 0, got {mb}")

    # -------- Get current mass, com, inertia  -------- 
    m0, I0, com0, T = simGetShapeMassAndInertia(handle)  # T captured here

    #  --------  Sample ballast point in the mesh local frame  -------- 
    p_b = sample_point_inside_shape(shape, margin_ratio=margin_ratio)
    
    #  -------- compute new mass + CoM + inertia (wrt local frame)  -------- 
    m = m0 + mb
    com_new = (m0 * com0 + mb * p_b) / m
    I_new = I0 + m0 * _S(com0 - com_new) + mb * _S(p_b - com_new)
    I_new = 0.5 * (I_new + I_new.T)

    simSetShapeMassAndInertia(handle, m, I_new, com_new, transf=T)

    if hasattr(sim, "simResetDynamicObject"):
        sim.simResetDynamicObject(handle)

    # return effective values (what engine actually uses)
    m_eff, I_eff, com_eff, _ = simGetShapeMassAndInertia(handle, transf=T)
    return m_eff, I_eff, com_eff


def inject_task_physics(task_env, rng=None, spec=None):
    if rng is None:
        rng = np.random

    if spec is None:
        spec = {"mass_frac": (0.05, 0.5), "mu": (0.2, 1.2)}

    task = getattr(task_env, "_task", None)
    if task is None:
        return {"note": "no task handle", "objects": {}}

    groceries = getattr(task, "groceries", None)
    if groceries is None:
        return {"note": "task has no groceries attribute; no injection applied", "objects": {}}

    meta = {"note": "applied to groceries", "objects": {}}

    # for g in groceries:
    g = [g for g in task_env._task.groceries if g.get_name()=='crackers'][0]
    
    name = g.get_name()
    h = g.get_handle()

    # mass props baseline (source of truth)
    m0_prop, _, _, _ = simGetShapeMassAndInertia(h)
    m0 = float(m0_prop)
    mu0 = float(g.get_bullet_friction())

    # ballast
    ballast_mass_override = getattr(task_env, '_ballast_mass_override', None)

    if ballast_mass_override is not None:
        mb = float(ballast_mass_override)
        alpha = mb / m0 if m0 > 0 else 0.0
    else:
        a_lo, a_hi = spec["mass_frac"]
        alpha = float(rng.uniform(a_lo, a_hi))
        alpha = 5.0
        mb = alpha * m0

    new_mass, new_I, new_com = add_point_mass_ballast_and_set(
        g, ballast_mass=mb, margin_ratio=0.05
    )

    # friction
    mu_lo, mu_hi = spec["mu"]
    mu1 = float(rng.uniform(mu_lo, mu_hi))
    g.set_bullet_friction(mu1)

    meta["objects"][name] = {
        "m0": m0,
        "mu0": mu0,
        "alpha": alpha,
        "mb": float(mb),
        "m1": float(new_mass),
        "mu1": mu1,
        "com1_local": np.asarray(new_com, dtype=float).reshape(3).tolist(),
        "inertia1_local": np.asarray(new_I, dtype=float).reshape(-1).tolist(),
        "inertia1_principal": np.linalg.eigvalsh(new_I).tolist(),
        "inertia1_massless_principal": (np.linalg.eigvalsh(new_I) / float(new_mass)).tolist(),
    }

    return meta
