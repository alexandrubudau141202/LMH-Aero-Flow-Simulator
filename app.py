"""
app.py  —  Motorsport Aero Sensitivity Simulator
-------------------------------------------------
Run:  python app.py
      open http://localhost:8050
"""

import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context
import plotly.graph_objects as go

from physics.aero_model import CAR_SPECS, compute_surfaces, get_readouts
from visualization.car_renderer import render_car

# ── App ───────────────────────────────────────────────────────────────────────
app = dash.Dash(__name__, title="Motorsport Aero Simulator", update_title=None)
server = app.server

CAR_KEYS = list(CAR_SPECS.keys())
DEFAULT  = "LMH"

# ── Plotly base styles ────────────────────────────────────────────────────────
_CHART_BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor ="rgba(0,0,0,0)",
    margin=dict(l=0, r=0, t=0, b=0),
    font=dict(family="Share Tech Mono, monospace", size=10, color="#4e6a80"),
)

_SCENE_BASE = dict(
    bgcolor="#06090d",
    xaxis=dict(title="Yaw (°)",        titlefont=dict(size=9), tickfont=dict(size=8),
               gridcolor="#1a2e3e", zerolinecolor="#274460"),
    yaxis=dict(title="Rear RH (mm)",   titlefont=dict(size=9), tickfont=dict(size=8),
               gridcolor="#1a2e3e", zerolinecolor="#274460"),
    zaxis=dict(                         titlefont=dict(size=9), tickfont=dict(size=8),
               gridcolor="#1a2e3e", zerolinecolor="#274460"),
    camera=dict(eye=dict(x=1.55, y=-1.55, z=1.1), up=dict(x=0, y=0, z=1)),
    aspectratio=dict(x=1, y=1, z=0.65),
)


def _surface_fig(yaw_arr, rh_arr, grid, cur_yaw, cur_rh, cur_z,
                 z_label, colorscale, marker_color):
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=yaw_arr, y=rh_arr, z=grid,
        colorscale=colorscale, opacity=0.88, showscale=False,
        contours=dict(z=dict(show=True, color="rgba(255,255,255,0.12)", width=1)),
    ))
    fig.add_trace(go.Scatter3d(
        x=[cur_yaw], y=[cur_rh], z=[cur_z],
        mode="markers+text",
        marker=dict(size=7, color=marker_color, symbol="diamond",
                    line=dict(color="#ffffff", width=1)),
        text=[f"{cur_z/1000:.2f}kN"],
        textfont=dict(size=9, color="#ffffff"),
        textposition="top center",
    ))
    fig.update_layout(
        **_CHART_BASE,
        scene=dict(**_SCENE_BASE,
                   zaxis=dict(**_SCENE_BASE["zaxis"], title=z_label)),
    )
    return fig


# ── Slider helper ─────────────────────────────────────────────────────────────
def _slider(sid, label, unit, lo, hi, default, step=1):
    marks = {int(v): str(int(v)) for v in np.linspace(lo, hi, 5)}
    return html.Div([
        html.Div([html.Span(label, className="slider-name"),
                  html.Span(unit,  className="slider-unit")],
                 className="slider-header"),
        dcc.Slider(id=sid, min=lo, max=hi, value=default, step=step,
                   marks=marks,
                   tooltip={"always_visible": True, "placement": "bottom"},
                   updatemode="drag"),
    ], className="slider-block")


# ── Layout ────────────────────────────────────────────────────────────────────
def build_layout():
    spec  = CAR_SPECS[DEFAULT]
    pills = [
        html.Button(k, id=f"pill-{k}", n_clicks=0,
                    className=f"class-pill active-{k}" if k == DEFAULT else "class-pill")
        for k in CAR_KEYS
    ]

    sidebar = html.Div([
        html.Div("INPUT PARAMETERS", className="section-label"),
        _slider("sl-front-rh", "FRONT RIDE HEIGHT", "mm",
                *spec["rh_front_range"], spec["rh_front_default"]),
        _slider("sl-rear-rh",  "REAR RIDE HEIGHT",  "mm",
                *spec["rh_rear_range"],  spec["rh_rear_default"]),
        _slider("sl-yaw",      "YAW ANGLE",          "°",
                -10, 10, 0, step=0.5),
        _slider("sl-speed",    "SPEED",              "km/h",
                *spec["speed_range"], spec["speed_default"], step=5),

        html.Div("GROUND EFFECT", className="section-label"),
        html.Div([
            html.Div([html.Span("EFFICIENCY"), html.Span(id="gh-pct")],
                     className="gh-bar-label"),
            html.Div(html.Div(id="gh-fill", style={"width": "65%"},
                              className="gh-bar-fill"),
                     className="gh-bar-track"),
        ], className="gh-bar-container"),

        html.Div("LIVE TELEMETRY", className="section-label"),
        html.Div([
            html.Div([html.Div("DOWNFORCE",    className="readout-label"),
                      html.Div(id="ro-df",     className="readout-value"),
                      html.Div(id="ro-df-sub", className="readout-sub")],
                     className="readout-cell"),
            html.Div([html.Div("DRAG",          className="readout-label"),
                      html.Div(id="ro-drag",    className="readout-value accent"),
                      html.Div(id="ro-drag-sub",className="readout-sub")],
                     className="readout-cell"),
            html.Div([html.Div("L / D",         className="readout-label"),
                      html.Div(id="ro-ld",      className="readout-value green"),
                      html.Div("lift-to-drag",  className="readout-sub")],
                     className="readout-cell"),
            html.Div([html.Div("AERO BALANCE",  className="readout-label"),
                      html.Div(id="ro-bal",     className="readout-value yellow"),
                      html.Div("% front",       className="readout-sub")],
                     className="readout-cell"),
            html.Div([html.Div("RAKE",          className="readout-label"),
                      html.Div(id="ro-rake",    className="readout-value"),
                      html.Div("rear − front mm", className="readout-sub")],
                     className="readout-cell"),
            html.Div([html.Div("DYN. PRESSURE", className="readout-label"),
                      html.Div(id="ro-q",       className="readout-value"),
                      html.Div("Pa",            className="readout-sub")],
                     className="readout-cell"),
        ], className="readout-grid"),
    ], className="sidebar")

    main = html.Div([
        html.Div([
            html.Img(id="car-img", className="car-image",
                     src="", style={"background": "#06090d"}),
            html.Div("RdBu  ·  BLUE = SUCTION  ·  RED = STAGNATION",
                     className="car-overlay-label"),
        ], className="car-view-container"),

        html.Div([
            html.Div([
                html.Div([html.Span("▲ DOWNFORCE", className="chart-title cyan"),
                          html.Span(id="df-peak",  className="chart-subtitle")],
                         className="chart-header"),
                dcc.Graph(id="chart-df", config={"displayModeBar": False},
                          style={"flex": "1", "minHeight": "0"}),
            ], className="chart-pane",
               style={"display": "flex", "flexDirection": "column"}),

            html.Div([
                html.Div([html.Span("▶ DRAG",         className="chart-title accent"),
                          html.Span(id="drag-peak",   className="chart-subtitle")],
                         className="chart-header"),
                dcc.Graph(id="chart-drag", config={"displayModeBar": False},
                          style={"flex": "1", "minHeight": "0"}),
            ], className="chart-pane",
               style={"display": "flex", "flexDirection": "column"}),
        ], className="charts-strip"),
    ], className="main-content")

    return html.Div([
        html.Div([
            html.Div([
                html.Span("MOTORSPORT ", className="app-title"),
                html.Span("AERO",        className="app-title",
                          style={"color": "var(--accent)"}),
                html.Span(" SIMULATOR",  className="app-title"),
            ]),
            html.Div(pills, className="class-pills"),
            html.Div(CAR_SPECS[DEFAULT]["series"], id="series-tag",
                     className="series-tag"),
        ], className="app-header"),

        html.Div([sidebar, main], className="app-body"),
        dcc.Store(id="car-class", data=DEFAULT),
    ], className="app-shell")


app.layout = build_layout()


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    [Output("car-class", "data"),
     *[Output(f"pill-{k}", "className") for k in CAR_KEYS],
     Output("series-tag", "children")],
    [Input(f"pill-{k}", "n_clicks") for k in CAR_KEYS],
    State("car-class", "data"),
    prevent_initial_call=True,
)
def pick_class(*args):
    current = args[-1]
    ctx     = callback_context
    if not ctx.triggered:
        selected = current
    else:
        selected = ctx.triggered[0]["prop_id"].split(".")[0].replace("pill-", "")
    pills = [f"class-pill active-{k}" if k == selected else "class-pill"
             for k in CAR_KEYS]
    return [selected, *pills, CAR_SPECS[selected]["series"]]


@app.callback(
    [Output("sl-front-rh", "min"),   Output("sl-front-rh", "max"),
     Output("sl-front-rh", "value"), Output("sl-front-rh", "marks"),
     Output("sl-rear-rh",  "min"),   Output("sl-rear-rh",  "max"),
     Output("sl-rear-rh",  "value"), Output("sl-rear-rh",  "marks"),
     Output("sl-speed",    "min"),   Output("sl-speed",    "max"),
     Output("sl-speed",    "value"), Output("sl-speed",    "marks")],
    Input("car-class", "data"),
)
def update_ranges(car_class):
    s = CAR_SPECS[car_class]
    def m5(lo, hi):
        return {int(v): str(int(v)) for v in np.linspace(lo, hi, 5)}
    fl, fh = s["rh_front_range"]
    rl, rh = s["rh_rear_range"]
    sl, sh = s["speed_range"]
    return (fl, fh, s["rh_front_default"], m5(fl, fh),
            rl, rh, s["rh_rear_default"],  m5(rl, rh),
            sl, sh, s["speed_default"],    m5(sl, sh))


@app.callback(
    [Output("car-img",    "src"),
     Output("chart-df",   "figure"),
     Output("chart-drag", "figure"),
     Output("ro-df",      "children"),   Output("ro-df-sub",   "children"),
     Output("ro-drag",    "children"),   Output("ro-drag-sub", "children"),
     Output("ro-ld",      "children"),   Output("ro-bal",      "children"),
     Output("ro-rake",    "children"),   Output("ro-q",        "children"),
     Output("gh-pct",     "children"),   Output("gh-fill",     "style"),
     Output("df-peak",    "children"),   Output("drag-peak",   "children")],
    [Input("car-class",  "data"),
     Input("sl-front-rh","value"),
     Input("sl-rear-rh", "value"),
     Input("sl-yaw",     "value"),
     Input("sl-speed",   "value")],
)
def update_all(car_class, front_rh, rear_rh, yaw, speed):
    if any(v is None for v in [car_class, front_rh, rear_rh, yaw, speed]):
        raise dash.exceptions.PreventUpdate

    spec = CAR_SPECS[car_class]
    rd   = get_readouts(car_class, front_rh, rear_rh, yaw, speed)
    df   = rd["downforce_N"]
    drag = rd["drag_N"]
    gh   = rd["gh_factor"]
    bal  = rd["aero_balance_pct"]

    yaw_arr, rh_arr, df_grid, drag_grid = compute_surfaces(
        car_class, front_rh, speed)

    df_fig   = _surface_fig(yaw_arr, rh_arr, df_grid,
                            yaw, rear_rh, df,
                            "Downforce (N)", "Blues_r", spec["accent"])
    drag_fig = _surface_fig(yaw_arr, rh_arr, drag_grid,
                            yaw, rear_rh, drag,
                            "Drag (N)", "Reds", "#ff4400")

    car_src = render_car(car_class, front_rh, rear_rh, yaw, speed,
                         gh, bal / 100, df, drag)

    cd_eff = spec["CD_base"] + spec["k_induced"] * (spec["CL_base"] * gh) ** 2

    return (
        car_src,
        df_fig,
        drag_fig,
        f"{df/1000:.2f} kN",
        f"F {rd['df_front_N']/1000:.2f}  R {rd['df_rear_N']/1000:.2f} kN",
        f"{drag/1000:.2f} kN",
        f"CD eff {cd_eff:.3f}",
        f"{rd['ld_ratio']:.2f}",
        f"{bal:.1f}%",
        f"{rd['rake_mm']:+.0f} mm",
        f"{rd['dynamic_q']:.0f}",
        f"{gh*100:.0f}%",
        {"width": f"{gh*100:.0f}%"},
        f"PEAK {df_grid.max()/1000:.2f} kN",
        f"PEAK {drag_grid.max()/1000:.2f} kN",
    )


if __name__ == "__main__":
    app.run(debug=True, port=8050)