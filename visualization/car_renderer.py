"""
visualization/car_renderer.py
------------------------------
Loads the car geometry (real STL or fallback primitives), colours each
vertex by inferred aerodynamic zone (Cp), applies rake + yaw tilt, and
renders to a base64 PNG data-URI for Dash.

Zone inference from normalised vertex position (works on any STL):
  X_norm ∈ [0,1]  front → rear
  Z_norm ∈ [0,1]  floor → top

  front_wing  X < 0.18
  nose        X < 0.24  AND  Z > 0.28
  underbody   Z < 0.18  AND  0.18 < X < 0.78
  diffuser    X > 0.72  AND  Z < 0.30
  rear_wing   X > 0.80  AND  Z > 0.55
  body        everything else
"""

import base64
import os
import tempfile

import numpy as np
import pyvista as pv

# ── Zone Cp values ────────────────────────────────────────────────────────────
def _zone_cp(zone: str, gh: float, aero_balance: float, yaw: float) -> float:
    if zone == "front_wing":
        return float(np.clip(0.08 + (1.0 - gh) * 0.18 - aero_balance * 0.15, 0, 1))
    if zone == "rear_wing":
        return float(np.clip(0.10 + (1.0 - gh) * 0.15 - (1 - aero_balance) * 0.12, 0, 1))
    if zone == "diffuser":
        return float(np.clip(0.05 + (1.0 - gh) * 0.25, 0, 1))
    if zone == "underbody":
        return float(np.clip(0.12 + (1.0 - gh) * 0.20, 0, 1))
    if zone == "nose":
        return float(np.clip(0.82 + abs(yaw) * 0.008, 0, 1))
    if zone == "body":
        return float(np.clip(0.42 + (1.0 - gh) * 0.06, 0, 1))
    return 0.45


def _assign_cp(mesh: pv.PolyData,
               gh: float, aero_balance: float, yaw: float) -> pv.PolyData:
    """Assign per-point Cp by inferring zones from normalised XZ position."""
    pts = np.array(mesh.points)
    x, z = pts[:, 0], pts[:, 2]

    xn = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    zn = (z - z.min()) / max(z.max() - z.min(), 1e-6)

    cp = np.full(len(pts), _zone_cp("body", gh, aero_balance, yaw), dtype=np.float32)

    cp[xn < 0.18]                          = _zone_cp("front_wing", gh, aero_balance, yaw)
    cp[(xn < 0.24) & (zn > 0.28)]          = _zone_cp("nose",       gh, aero_balance, yaw)
    cp[(zn < 0.18) & (xn > 0.18) & (xn < 0.78)] = _zone_cp("underbody", gh, aero_balance, yaw)
    cp[(xn > 0.72) & (zn < 0.30)]          = _zone_cp("diffuser",   gh, aero_balance, yaw)
    cp[(xn > 0.80) & (zn > 0.55)]          = _zone_cp("rear_wing",  gh, aero_balance, yaw)

    out = mesh.copy()
    out.point_data["Cp"] = cp
    return out


def _normalise(mesh: pv.PolyData, target_length: float = 4.8) -> pv.PolyData:
    """Centre at origin and scale to target car length."""
    mesh = mesh.copy()
    b    = mesh.bounds
    cx   = 0.5 * (b[0] + b[1])
    cy   = 0.5 * (b[2] + b[3])
    cz   = 0.5 * (b[4] + b[5])
    mesh.translate((-cx, -cy, -cz), inplace=True)
    length = b[1] - b[0]
    if length > 1e-6:
        mesh.scale(target_length / length, inplace=True)
    return mesh


# ── STL resolution ────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "models")
)


def _stl_for_class(car_class: str):
    """Exact match first, then any STL in models/ as shared fallback."""
    exact = os.path.join(_MODELS_DIR, f"{car_class}.stl")
    if os.path.isfile(exact):
        return exact, "exact"
    if os.path.isdir(_MODELS_DIR):
        for f in sorted(os.listdir(_MODELS_DIR)):
            if f.lower().endswith(".stl"):
                return os.path.join(_MODELS_DIR, f), "shared"
    return None, None


def _load_stl(path: str) -> pv.PolyData:
    raw = pv.read(path)
    if not isinstance(raw, pv.PolyData):
        raw = raw.extract_surface()
    return _normalise(raw.triangulate())


# ── Main render ───────────────────────────────────────────────────────────────
def render_car(car_class: str,
               front_rh: float, rear_rh: float,
               yaw_angle: float, speed_kmh: float,
               gh_factor: float, aero_balance: float,
               downforce_N: float, drag_N: float,
               width: int = 900, height: int = 320) -> str:

    from physics.aero_model import CAR_SPECS
    spec = CAR_SPECS[car_class]

    # ── Geometry ──────────────────────────────────────────────────────────
    stl_path, stl_mode = _stl_for_class(car_class)
    if stl_path:
        mesh = _load_stl(stl_path)
        tag  = "[STL]"
    else:
        from visualization._primitives import get_parts
        parts = get_parts(car_class)
        mesh  = parts[0].copy()
        for p in parts[1:]:
            mesh = mesh.merge(p)
        tag = "[PRIM]"

    # ── Cp colouring ──────────────────────────────────────────────────────
    mesh = _assign_cp(mesh, gh_factor, aero_balance, yaw_angle)

    # ── Rake tilt ─────────────────────────────────────────────────────────
    rake_m   = (rear_rh - front_rh) / 1000.0
    rake_deg = float(np.degrees(np.arctan2(rake_m, 3.1)))
    mesh.rotate_y(-rake_deg, inplace=True)

    # ── Yaw visual ────────────────────────────────────────────────────────
    mesh.rotate_z(float(yaw_angle) * 0.25, inplace=True)

    # ── Render ────────────────────────────────────────────────────────────
    pl = pv.Plotter(off_screen=True, window_size=(width, height))
    pl.set_background("#06090d")

    pl.add_mesh(mesh, scalars="Cp", cmap="RdBu",
                clim=(0.0, 1.0), smooth_shading=True,
                show_scalar_bar=False)

    # Ground plane
    gp = pv.Plane(center=(0, 0, mesh.bounds[4] - 0.03),
                  direction=(0, 0, 1), i_size=14, j_size=7)
    pl.add_mesh(gp, color="#0c1828", opacity=0.55)

    # Telemetry overlay
    ld = downforce_N / max(drag_N, 1.0)
    pl.add_text(
        f"DF {downforce_N/1000:.2f} kN   DRAG {drag_N/1000:.2f} kN"
        f"   L/D {ld:.2f}   GH {gh_factor*100:.0f}%",
        position=(0.02, 0.04), font_size=8, color="#00ccff",
    )
    pl.add_text(
        f"{spec['name'].upper()}  {tag}",
        position=(0.02, 0.88), font_size=9, color=spec["accent"],
    )

    # Camera
    b       = mesh.bounds
    car_len = b[1] - b[0]
    car_h   = b[5] - b[4]
    pl.camera_position = [
        (-car_len * 1.2, -car_len * 0.85, car_h * 2.2),
        ( car_len * 0.05, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ]
    pl.camera.view_angle = 38

    # Save to base64
    fd, tmp = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        pl.screenshot(tmp)
        pl.close()
        with open(tmp, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    return f"data:image/png;base64,{b64}"