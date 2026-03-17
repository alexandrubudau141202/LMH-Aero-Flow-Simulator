"""
visualization/car_renderer.py
"""

import os
import numpy as np
import plotly.graph_objects as go

# ── Zone Cp ───────────────────────────────────────────────────────────────────
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
    return float(np.clip(0.42 + (1.0 - gh) * 0.06, 0, 1))


def _assign_cp(pts: np.ndarray, gh: float,
               aero_balance: float, yaw: float) -> np.ndarray:
    x, z = pts[:, 0], pts[:, 2]
    xn = (x - x.min()) / max(x.max() - x.min(), 1e-6)
    zn = (z - z.min()) / max(z.max() - z.min(), 1e-6)
    cp = np.full(len(pts), _zone_cp("body", gh, aero_balance, yaw), dtype=np.float32)
    cp[xn < 0.18]                                = _zone_cp("front_wing", gh, aero_balance, yaw)
    cp[(xn < 0.24) & (zn > 0.28)]               = _zone_cp("nose",       gh, aero_balance, yaw)
    cp[(zn < 0.18) & (xn > 0.18) & (xn < 0.78)] = _zone_cp("underbody",  gh, aero_balance, yaw)
    cp[(xn > 0.72) & (zn < 0.30)]               = _zone_cp("diffuser",   gh, aero_balance, yaw)
    cp[(xn > 0.80) & (zn > 0.55)]               = _zone_cp("rear_wing",  gh, aero_balance, yaw)
    return cp


def _fix_orientation(pts: np.ndarray, is_glb: bool = False) -> np.ndarray:
    """
    STL files: already X-forward Z-up from CAD — no rotation applied.
    GLB files: hardcoded transform for glTF Z-forward Y-up convention:
      new_X =  Z  (length)
      new_Y =  X  (lateral)
      new_Z = -Y  (height, negated because glTF Y-up → Z-up flips sign)
    Then translate floor to Z=0.
    """
    pts = pts.copy()

    if is_glb:
        # glTF canonical: Z=forward, Y=up, X=right
        # We want:        X=forward, Y=lateral, Z=up
        x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
        pts = np.stack([z, x, -y], axis=1)

    # Translate so floor sits at Z=0
    pts[:, 2] -= pts[:, 2].min()
    return pts


def _normalise(pts: np.ndarray, target_length: float = 4.8) -> np.ndarray:
    pts = pts.copy()
    pts -= pts.mean(axis=0)
    length = pts[:, 0].max() - pts[:, 0].min()
    if length > 1e-6:
        pts *= target_length / length
    return pts


# ── Model finder ──────────────────────────────────────────────────────────────
_MODELS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
)


def _find_model(car_class: str):
    if not os.path.isdir(_MODELS_DIR):
        return None
    for f in os.listdir(_MODELS_DIR):
        name, ext = os.path.splitext(f)
        if name.lower() == car_class.lower() and ext.lower() in (".stl", ".glb"):
            return os.path.join(_MODELS_DIR, f)
    return None


# ── Loaders ───────────────────────────────────────────────────────────────────
def _read_stl(path: str):
    with open(path, "rb") as fh:
        fh.read(80)
        n_tri = int.from_bytes(fh.read(4), "little")
        data  = fh.read()
    if len(data) == n_tri * 50:
        verts, faces = [], []
        offset = 0
        for i in range(n_tri):
            offset += 12
            v0 = np.frombuffer(data[offset:offset+12], dtype=np.float32); offset += 12
            v1 = np.frombuffer(data[offset:offset+12], dtype=np.float32); offset += 12
            v2 = np.frombuffer(data[offset:offset+12], dtype=np.float32); offset += 12
            offset += 2
            base = len(verts)
            verts.extend([v0, v1, v2])
            faces.append([base, base+1, base+2])
        return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)
    # ASCII fallback
    verts, faces, tri = [], [], []
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith("vertex"):
                p = line.split()
                tri.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith("endloop") and len(tri) == 3:
                base = len(verts)
                verts.extend(tri)
                faces.append([base, base+1, base+2])
                tri = []
    return np.array(verts, dtype=np.float32), np.array(faces, dtype=np.int32)


def _read_glb(path: str):
    import trimesh
    scene = trimesh.load(path, force="scene")
    if isinstance(scene, trimesh.Scene):
        meshes = list(scene.dump(concatenate=False))
        combined = trimesh.util.concatenate(meshes)
    else:
        combined = scene
    return (np.array(combined.vertices, dtype=np.float32),
            np.array(combined.faces,    dtype=np.int32))


def _primitive_mesh(car_class: str):
    from visualization._primitives import get_parts
    parts  = get_parts(car_class)
    merged = parts[0].copy()
    for p in parts[1:]:
        merged = merged.merge(p)
    surf = merged.extract_surface().triangulate()
    return (np.array(surf.points, dtype=np.float32),
            surf.faces.reshape(-1, 4)[:, 1:].astype(np.int32))


# ── Main ──────────────────────────────────────────────────────────────────────
def render_car(car_class: str,
               front_rh: float, rear_rh: float,
               yaw_angle: float, speed_kmh: float,
               gh_factor: float, aero_balance: float,
               downforce_N: float, drag_N: float) -> go.Figure:

    from physics.aero_model import CAR_SPECS
    spec = CAR_SPECS[car_class]

    model_path = _find_model(car_class)
    is_glb     = bool(model_path and model_path.lower().endswith(".glb"))
    tag        = "PRIM"

    if model_path:
        try:
            verts, faces = _read_glb(model_path) if is_glb else _read_stl(model_path)
            tag = "GLB" if is_glb else "STL"
        except Exception as e:
            print(f"[car_renderer] load failed: {e}")
            model_path = None

    if not model_path:
        try:
            verts, faces = _primitive_mesh(car_class)
        except Exception:
            verts = np.array([[0,0,0],[1,0,0],[0,1,0]], dtype=np.float32)
            faces = np.array([[0,1,2]], dtype=np.int32)
            tag   = "ERR"

    # Transform
    verts = _fix_orientation(verts, is_glb=is_glb)
    verts = _normalise(verts)

    # Rake
    rake_rad = np.arctan2((rear_rh - front_rh) / 1000.0, 3.1)
    cy, sy   = np.cos(-rake_rad), np.sin(-rake_rad)
    verts    = verts @ np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]]).T

    # Yaw
    yaw_rad = np.radians(yaw_angle * 0.25)
    cz, sz  = np.cos(yaw_rad), np.sin(yaw_rad)
    verts   = verts @ np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]]).T

    cp = _assign_cp(verts, gh_factor, aero_balance, yaw_angle)

    x, y, z       = verts[:,0], verts[:,1], verts[:,2]
    i_, j_, k_    = faces[:,0], faces[:,1], faces[:,2]
    face_cp        = (cp[i_] + cp[j_] + cp[k_]) / 3.0
    ld             = downforce_N / max(drag_N, 1.0)

    fig = go.Figure(go.Mesh3d(
        x=x, y=y, z=z,
        i=i_, j=j_, k=k_,
        intensity=face_cp,
        colorscale="RdBu", reversescale=True,
        cmin=0.0, cmax=1.0,
        showscale=False, flatshading=False,
        lighting=dict(ambient=0.5, diffuse=0.8, specular=0.3,
                      roughness=0.5, fresnel=0.2),
        lightposition=dict(x=-2, y=-3, z=5),
    ))

    fig.update_layout(
        paper_bgcolor="#06090d",
        plot_bgcolor ="#06090d",
        margin=dict(l=0, r=0, t=0, b=0),
        scene=dict(
            bgcolor="#06090d",
            xaxis=dict(visible=False, showgrid=False),
            yaxis=dict(visible=False, showgrid=False),
            zaxis=dict(visible=False, showgrid=False),
            aspectmode="data",
            camera=dict(
                eye=dict(x=0, y=-4.5, z=0.8),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1),
            ),
            dragmode="orbit",
        ),
        annotations=[dict(
            x=0.01, y=0.04, xref="paper", yref="paper",
            text=(f"<b>DF</b> {downforce_N/1000:.2f} kN  "
                  f"<b>DRAG</b> {drag_N/1000:.2f} kN  "
                  f"<b>L/D</b> {ld:.2f}  "
                  f"<b>GH</b> {gh_factor*100:.0f}%  "
                  f"<span style='color:{spec['accent']}'>"
                  f"{spec['name'].upper()} [{tag}]</span>"),
            showarrow=False,
            font=dict(size=11, color="#00ccff", family="Share Tech Mono"),
            align="left",
        )],
    )
    return fig