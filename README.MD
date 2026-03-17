# Motorsport Aero Simulator

A real-time aerodynamic sensitivity simulator for endurance motorsport car classes built with Dash and Plotly. Load your own 3D car models, tune ride heights, yaw angle and speed, and watch the downforce/drag surfaces update live.

![Motorsport Aero Simulator](screenshot.png)

---

## Car classes

| Class | Series | Notes |
|-------|--------|-------|
| LMH   | WEC    | Le Mans Hypercar |
| LMDh  | WEC / IMSA | Hybrid LMDh |
| LMP2  | ELMS / WEC | ORECA 07 spec |
| LMP3  | ELMS   | Ligier JS P320 |
| GT3   | GTWC Europe | BoP-regulated GT |

---

## Project structure

```
motorsport-aero-sim/
├── app.py                        ← Dash app, layout & callbacks
├── requirements.txt
├── assets/
│   └── style.css                 ← Dark industrial UI theme
├── models/                       ← Drop your STL / GLB files here
│   ├── LMH.stl
│   ├── LMDh.stl
│   ├── LMP2.glb
│   ├── LMP3.glb
│   └── GT3.stl
├── physics/
│   ├── __init__.py
│   └── aero_model.py             ← Parametric aero physics engine
└── visualization/
    ├── __init__.py
    ├── car_renderer.py           ← Plotly Mesh3d car renderer
    └── _primitives.py            ← Fallback box/cylinder geometry
```

---

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
# → http://localhost:8050
```

---

## 3D model files

Place models in the `models/` folder named exactly after the class key. Both STL and GLB formats are supported and can be mixed across classes.

| Filename   | Format |
|------------|--------|
| `LMH.stl`  | STL    |
| `LMDh.stl` | STL    |
| `LMP2.glb` | GLB    |
| `LMP3.glb` | GLB    |
| `GT3.stl`  | STL    |

Filenames are matched **case-insensitively** (`lmp2.glb`, `LMP2.GLB` etc. all work). If no file is found for a class the app falls back to a built-in primitive geometry and shows `[PRIM]` in the 3D view label.

---

## Input controls

| Slider | Range | Effect |
|--------|-------|--------|
| Front Ride Height | class-specific mm | Ground effect + rake angle |
| Rear Ride Height  | class-specific mm | Ground effect + rake angle |
| Yaw Angle | −10° → +10° | Crosswind sensitivity |
| Speed | class-specific km/h | Dynamic pressure scaling |

---

## Live telemetry readouts

- **Downforce** — total and front/rear split (kN)
- **Drag** — total with effective CD
- **L/D ratio** — aerodynamic efficiency
- **Aero balance** — % front downforce
- **Rake** — rear minus front ride height (mm)
- **Dynamic pressure** — ½ρv² (Pa)
- **Ground effect efficiency** — % of peak ground effect

---

## Physics model

The simulator uses an **inviscid parametric aero model**:

```
q    = ½ρv²                              dynamic pressure
F_L  = CL(h_f, h_r, ψ) · q · A_ref     downforce
F_D  = CD(h_f, h_r, ψ) · q · A_ref     drag
CL   = CL_base · GH(h_f, h_r) · YD(ψ)
CD   = (CD_base + k·CL²) · YD_drag(ψ)
```

- **Ground effect** `GH` — 2D Gaussian centred on each class's optimal ride heights
- **Yaw** `YD` — quadratic penalty on downforce, additive on drag
- **Aero balance** — shifts linearly with rake angle delta from optimal

All coefficients are tuned to representative real-world values for each class (LMH peak ~7 kN at 240 km/h, GT3 peak ~1.8 kN at 195 km/h).

---

## Cp colouring

The 3D car view colours each surface zone by pressure coefficient:

| Colour | Zone | Meaning |
|--------|------|---------|
| Blue   | Front wing, underbody, diffuser | Suction / downforce generating |
| White  | Neutral | Freestream pressure |
| Red    | Nose, stagnation points | High pressure / drag |

Colours update live as you move the sliders.

---

## License

MIT