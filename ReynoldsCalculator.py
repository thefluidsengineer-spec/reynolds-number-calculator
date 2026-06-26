"""The Fluids Engineer — Reynolds Number Visualizer.

A Streamlit app for estimating Reynolds number for:
1. Internal flow in a circular pipe
2. External flow over a smooth flat plate at zero pressure gradient

All calculations are performed in SI units internally.
"""

from __future__ import annotations

import csv
import io
import math
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Circle, Wedge


# -----------------------------------------------------------------------------
# Site configuration — add these URLs when they are ready.
# -----------------------------------------------------------------------------
YOUTUBE_VIDEO_URL = ""  # Example: "https://www.youtube.com/watch?v=..."
NEWSLETTER_URL = ""  # Example: "https://your-domain.com/newsletter"


# -----------------------------------------------------------------------------
# Constants and reference values
# -----------------------------------------------------------------------------
PIPE_LAMINAR_LIMIT = 2300.0
PIPE_TURBULENT_LIMIT = 4000.0
DEFAULT_FLAT_PLATE_CRITICAL_RE = 5.0e5

LAMINAR_COLOR = "#2CA58D"
TRANSITION_COLOR = "#F4A261"
TURBULENT_COLOR = "#E76F51"
DARK_COLOR = "#0B132B"
ACCENT_COLOR = "#2D6CDF"
GRID_COLOR = "#D6DCE8"

FT_TO_M = 0.3048
IN_TO_M = 0.0254
US_GPM_TO_M3_S = 0.003785411784 / 60.0
LBM_FT3_TO_KG_M3 = 16.01846337396
LBM_FT_S_TO_PA_S = 1.48816394357
CST_TO_M2_S = 1.0e-6


@dataclass(frozen=True)
class FluidPreset:
    density_kg_m3: float
    dynamic_viscosity_pa_s: float
    note: str


FLUID_PRESETS = {
    "Water at 20 °C (approx.)": FluidPreset(
        density_kg_m3=998.2,
        dynamic_viscosity_pa_s=1.002e-3,
        note="Approximate properties near 20 °C and 1 atm.",
    ),
    "Air at 20 °C, 1 atm (approx.)": FluidPreset(
        density_kg_m3=1.204,
        dynamic_viscosity_pa_s=1.825e-5,
        note="Approximate dry-air properties near 20 °C and 1 atm.",
    ),
}


# -----------------------------------------------------------------------------
# Pure calculation helpers
# -----------------------------------------------------------------------------
def reynolds_from_dynamic(
    density_kg_m3: float,
    velocity_m_s: float,
    characteristic_length_m: float,
    dynamic_viscosity_pa_s: float,
) -> float:
    """Return Re = rho * V * L / mu."""
    return (
        density_kg_m3
        * velocity_m_s
        * characteristic_length_m
        / dynamic_viscosity_pa_s
    )


def reynolds_from_kinematic(
    velocity_m_s: float,
    characteristic_length_m: float,
    kinematic_viscosity_m2_s: float,
) -> float:
    """Return Re = V * L / nu."""
    return velocity_m_s * characteristic_length_m / kinematic_viscosity_m2_s


def pipe_velocity_from_flow_rate(flow_rate_m3_s: float, diameter_m: float) -> float:
    """Return mean velocity in a circular pipe from Q/A."""
    area_m2 = math.pi * diameter_m**2 / 4.0
    return flow_rate_m3_s / area_m2


def classify_pipe_flow(reynolds_number: float) -> tuple[str, str, str]:
    """Return regime, display color, and explanation for circular-pipe flow."""
    if reynolds_number < PIPE_LAMINAR_LIMIT:
        return (
            "Laminar",
            LAMINAR_COLOR,
            "Viscous effects dominate and orderly layers are expected.",
        )
    if reynolds_number < PIPE_TURBULENT_LIMIT:
        return (
            "Transitional",
            TRANSITION_COLOR,
            "The flow may switch between laminar and turbulent behavior.",
        )
    return (
        "Turbulent",
        TURBULENT_COLOR,
        "Inertial effects dominate and strong mixing is expected.",
    )


def classify_flat_plate_flow(
    reynolds_number_x: float, critical_reynolds_number: float
) -> tuple[str, str, str]:
    """Classify a local flat-plate boundary layer using an assumed critical Re_x."""
    if reynolds_number_x < critical_reynolds_number:
        return (
            "Likely laminar",
            LAMINAR_COLOR,
            "The selected location is upstream of the assumed transition point.",
        )
    return (
        "Transition/turbulence likely",
        TURBULENT_COLOR,
        "The selected location is at or beyond the assumed transition point.",
    )


def laminar_flat_plate_thickness(
    x_m: np.ndarray | float, reynolds_number_x: np.ndarray | float
) -> np.ndarray | float:
    """Approximate 99% laminar boundary-layer thickness: delta ≈ 5x/sqrt(Re_x)."""
    return 5.0 * np.asarray(x_m) / np.sqrt(np.asarray(reynolds_number_x))


def turbulent_flat_plate_thickness(
    x_m: np.ndarray | float, reynolds_number_x: np.ndarray | float
) -> np.ndarray | float:
    """Approximate smooth-plate turbulent thickness: delta ≈ 0.37x/Re_x^(1/5).

    This is a fully turbulent, zero-pressure-gradient reference correlation and does
    not model the transition region itself.
    """
    return 0.37 * np.asarray(x_m) / np.asarray(reynolds_number_x) ** 0.2


def format_number(value: float) -> str:
    """Readable engineering-style formatting."""
    absolute = abs(value)
    if absolute == 0:
        return "0"
    if absolute >= 1.0e6 or absolute < 1.0e-3:
        return f"{value:.3e}"
    if absolute >= 1000:
        return f"{value:,.0f}"
    if absolute >= 10:
        return f"{value:,.2f}"
    return f"{value:,.4f}"


def safe_query_value(
    name: str, default: str, cast: Callable[[str], object] = str
) -> object:
    """Read a query parameter without allowing malformed links to crash the app."""
    try:
        raw_value = st.query_params.get(name, default)
        return cast(raw_value)
    except (TypeError, ValueError):
        return cast(default)


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------
def _log_fraction(value: float, minimum: float, maximum: float) -> float:
    clipped = float(np.clip(value, minimum, maximum))
    return (
        math.log10(clipped) - math.log10(minimum)
    ) / (math.log10(maximum) - math.log10(minimum))


def draw_regime_gauge(
    value: float,
    minimum: float,
    maximum: float,
    boundaries: list[float],
    labels: list[str],
    colors: list[str],
) -> plt.Figure:
    """Draw a logarithmic semicircular Reynolds-number regime gauge."""
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    segment_edges = [minimum, *boundaries, maximum]
    for start, end, color in zip(segment_edges[:-1], segment_edges[1:], colors):
        start_fraction = _log_fraction(start, minimum, maximum)
        end_fraction = _log_fraction(end, minimum, maximum)
        start_angle = 180.0 * (1.0 - start_fraction)
        end_angle = 180.0 * (1.0 - end_fraction)
        ax.add_patch(
            Wedge(
                (0, 0),
                1.0,
                end_angle,
                start_angle,
                width=0.27,
                facecolor=color,
                edgecolor="white",
                linewidth=2,
            )
        )

    for start, end, label in zip(segment_edges[:-1], segment_edges[1:], labels):
        midpoint = 10 ** ((math.log10(start) + math.log10(end)) / 2.0)
        fraction = _log_fraction(midpoint, minimum, maximum)
        angle = math.pi * (1.0 - fraction)
        ax.text(
            0.84 * math.cos(angle),
            0.84 * math.sin(angle),
            label,
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
            color=DARK_COLOR,
        )

    value_fraction = _log_fraction(value, minimum, maximum)
    needle_angle = math.pi * (1.0 - value_fraction)
    ax.plot(
        [0, 0.72 * math.cos(needle_angle)],
        [0, 0.72 * math.sin(needle_angle)],
        color=DARK_COLOR,
        linewidth=4,
        solid_capstyle="round",
        zorder=5,
    )
    ax.add_patch(Circle((0, 0), 0.065, facecolor=DARK_COLOR, zorder=6))

    ax.text(
        0,
        -0.18,
        f"Re = {format_number(value)}",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=DARK_COLOR,
    )
    ax.text(
        -1.02,
        -0.03,
        format_number(minimum),
        ha="center",
        va="center",
        fontsize=8,
        color="#52606D",
    )
    ax.text(
        1.02,
        -0.03,
        format_number(maximum),
        ha="center",
        va="center",
        fontsize=8,
        color="#52606D",
    )
    ax.text(
        0,
        -0.35,
        "Logarithmic scale",
        ha="center",
        va="center",
        fontsize=8,
        color="#7B8794",
    )

    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-0.44, 1.08)
    ax.set_aspect("equal")
    ax.axis("off")
    plt.tight_layout(pad=0.2)
    return fig


def draw_pipe_velocity_profile(reynolds_number: float) -> plt.Figure:
    """Draw exact laminar or approximate turbulent normalized pipe profiles."""
    radial_coordinate = np.linspace(-1.0, 1.0, 400)
    laminar_profile = 1.0 - radial_coordinate**2
    turbulent_profile = np.maximum(0.0, 1.0 - np.abs(radial_coordinate)) ** (1.0 / 7.0)

    if reynolds_number < PIPE_LAMINAR_LIMIT:
        profile = laminar_profile
        label = "Fully developed laminar profile"
    elif reynolds_number < PIPE_TURBULENT_LIMIT:
        blend = (reynolds_number - PIPE_LAMINAR_LIMIT) / (
            PIPE_TURBULENT_LIMIT - PIPE_LAMINAR_LIMIT
        )
        profile = (1.0 - blend) * laminar_profile + blend * turbulent_profile
        label = "Illustrative transitional blend"
    else:
        profile = turbulent_profile
        label = "Approximate 1/7-power turbulent profile"

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    ax.plot(profile, radial_coordinate, linewidth=3, color=ACCENT_COLOR)
    ax.fill_betweenx(
        radial_coordinate, 0, profile, alpha=0.12, color=ACCENT_COLOR
    )
    ax.axhline(0, linewidth=0.8, color=GRID_COLOR)
    ax.set_xlim(0, 1.08)
    ax.set_ylim(-1.03, 1.03)
    ax.set_xlabel("Local velocity / centerline velocity")
    ax.set_ylabel("Normalized radius, r/R")
    ax.set_title(label)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def draw_pipe_flow_pattern(reynolds_number: float) -> plt.Figure:
    """Draw a conceptual flow-pattern illustration; this is explicitly not CFD."""
    rng = np.random.default_rng(12)
    x = np.linspace(0.0, 10.0, 500)
    fig, ax = plt.subplots(figsize=(7.0, 2.3))

    for index, baseline in enumerate(np.linspace(-0.82, 0.82, 11)):
        if reynolds_number < PIPE_LAMINAR_LIMIT:
            y = np.full_like(x, baseline)
        elif reynolds_number < PIPE_TURBULENT_LIMIT:
            amplitude = 0.015 + 0.012 * (index % 3)
            y = baseline + amplitude * np.sin(2.0 * x + index * 0.6)
        else:
            noise = rng.normal(0.0, 1.0, size=x.size)
            kernel = np.ones(25) / 25.0
            smooth_noise = np.convolve(noise, kernel, mode="same")
            y = baseline + 0.18 * smooth_noise + 0.025 * np.sin(5 * x + index)
        ax.plot(x, y, linewidth=1.2)

    ax.plot([0, 10], [1, 1], color=DARK_COLOR, linewidth=3)
    ax.plot([0, 10], [-1, -1], color=DARK_COLOR, linewidth=3)
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.12, 1.12)
    ax.set_title("Conceptual flow pattern — not a CFD result")
    ax.axis("off")
    fig.tight_layout()
    return fig


def draw_flat_plate_boundary_layer(
    velocity_m_s: float,
    kinematic_viscosity_m2_s: float,
    selected_x_m: float,
    critical_reynolds_number: float,
) -> plt.Figure:
    """Plot laminar and fully turbulent reference thickness correlations."""
    transition_x_m = critical_reynolds_number * kinematic_viscosity_m2_s / velocity_m_s
    plot_end_m = max(selected_x_m * 1.35, min(transition_x_m * 1.15, selected_x_m * 5.0))
    plot_end_m = max(plot_end_m, selected_x_m + 1.0e-6)
    x_values = np.linspace(max(plot_end_m / 1000.0, 1.0e-8), plot_end_m, 700)
    re_x_values = velocity_m_s * x_values / kinematic_viscosity_m2_s

    delta_laminar_mm = 1000.0 * laminar_flat_plate_thickness(x_values, re_x_values)
    delta_turbulent_mm = 1000.0 * turbulent_flat_plate_thickness(x_values, re_x_values)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(
        x_values,
        delta_laminar_mm,
        linewidth=2.5,
        color=LAMINAR_COLOR,
        label="Laminar correlation: 5x/√Reₓ",
    )
    ax.plot(
        x_values,
        delta_turbulent_mm,
        linewidth=2.5,
        linestyle="--",
        color=TURBULENT_COLOR,
        label="Fully turbulent reference: 0.37x/Reₓ⁰·²",
    )
    ax.axvline(
        selected_x_m,
        color=ACCENT_COLOR,
        linewidth=2,
        label="Selected location",
    )
    if transition_x_m <= plot_end_m:
        ax.axvline(
            transition_x_m,
            color=TRANSITION_COLOR,
            linewidth=2,
            linestyle=":",
            label="Assumed transition location",
        )

    ax.set_xlabel("Distance from leading edge, x (m)")
    ax.set_ylabel("Approximate 99% boundary-layer thickness (mm)")
    ax.set_title("Smooth flat plate, zero pressure gradient")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def result_csv(data: dict[str, str]) -> str:
    """Create a two-column CSV string for download."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["Quantity", "Value"])
    for key, value in data.items():
        writer.writerow([key, value])
    return buffer.getvalue()


# -----------------------------------------------------------------------------
# Streamlit page setup and styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Reynolds Number Calculator | The Fluids Engineer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
        [data-testid="stSidebar"] {background: #F5F7FB;}
        .tfe-kicker {color: #2D6CDF; font-weight: 800; letter-spacing: .08em;
                     text-transform: uppercase; font-size: .78rem; margin-bottom: .35rem;}
        .tfe-subtitle {font-size: 1.08rem; color: #52606D; max-width: 780px;
                       margin-top: -.45rem; margin-bottom: 1.4rem;}
        .regime-card {padding: 1rem 1.15rem; border-radius: 14px; border: 1px solid #E1E7EF;
                      background: white; box-shadow: 0 3px 14px rgba(13, 38, 76, .06);}
        .regime-label {font-size: .78rem; color: #66788A; font-weight: 700;
                       text-transform: uppercase; letter-spacing: .05em;}
        .regime-value {font-size: 1.45rem; font-weight: 800; margin: .15rem 0;}
        .regime-note {color: #52606D; font-size: .92rem;}
        .cta-card {padding: 1.25rem 1.35rem; border-radius: 16px; background: #EEF4FF;
                   border: 1px solid #CFE0FF;}
        div[data-testid="stMetric"] {background: white; border: 1px solid #E1E7EF;
                                     padding: .8rem 1rem; border-radius: 12px;}
        .small-note {color: #66788A; font-size: .84rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="tfe-kicker">The Fluids Engineer</div>', unsafe_allow_html=True)
st.title("Reynolds Number Calculator & Flow Visualizer")
st.markdown(
    '<div class="tfe-subtitle">TURN ON LIGHT MODE Estimate whether a pipe flow or flat-plate boundary layer is laminar, transitional, or turbulent—and see what the result means physically.</div>',
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# URL-backed defaults
# -----------------------------------------------------------------------------
geometry_query = safe_query_value("geometry", "pipe")
units_query = safe_query_value("units", "SI")
fluid_query = safe_query_value("fluid", "Water at 20 °C (approx.)")

geometry_options = ["Circular pipe", "Flat plate boundary layer"]
geometry_default = 0 if geometry_query == "pipe" else 1
unit_options = ["SI", "U.S. customary"]
unit_default = 0 if units_query == "SI" else 1
fluid_options = [*FLUID_PRESETS.keys(), "Custom fluid"]
fluid_default = fluid_options.index(fluid_query) if fluid_query in fluid_options else 0


# -----------------------------------------------------------------------------
# Sidebar controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Inputs")
    geometry = st.selectbox("Flow geometry", geometry_options, index=geometry_default)
    unit_system = st.radio("Unit system", unit_options, index=unit_default, horizontal=True)
    fluid_choice = st.selectbox("Fluid", fluid_options, index=fluid_default)

    is_si = unit_system == "SI"

    if fluid_choice in FLUID_PRESETS:
        preset = FLUID_PRESETS[fluid_choice]
        density_kg_m3 = preset.density_kg_m3
        dynamic_viscosity_pa_s = preset.dynamic_viscosity_pa_s
        kinematic_viscosity_m2_s = dynamic_viscosity_pa_s / density_kg_m3

        if is_si:
            st.caption(
                f"ρ = {density_kg_m3:.4g} kg/m³ · μ = {dynamic_viscosity_pa_s:.4g} Pa·s"
            )
        else:
            density_us = density_kg_m3 / LBM_FT3_TO_KG_M3
            viscosity_us = dynamic_viscosity_pa_s / LBM_FT_S_TO_PA_S
            st.caption(f"ρ = {density_us:.4g} lbm/ft³ · μ = {viscosity_us:.4g} lbm/(ft·s)")
        st.caption(preset.note)
        viscosity_basis = "Dynamic viscosity, μ"
        density_input = density_kg_m3 if is_si else density_kg_m3 / LBM_FT3_TO_KG_M3
        viscosity_input = (
            dynamic_viscosity_pa_s
            if is_si
            else dynamic_viscosity_pa_s / LBM_FT_S_TO_PA_S
        )
    else:
        st.subheader("Fluid properties")
        viscosity_basis_query = safe_query_value("viscosity_basis", "dynamic")
        viscosity_basis = st.radio(
            "Viscosity input",
            ["Dynamic viscosity, μ", "Kinematic viscosity, ν"],
            index=0 if viscosity_basis_query == "dynamic" else 1,
            horizontal=False,
        )

        if is_si:
            density_input = st.number_input(
                "Density, ρ (kg/m³)",
                min_value=1.0e-6,
                value=float(safe_query_value("rho", "1000", float)),
                format="%.6g",
            )
            density_kg_m3 = density_input
        else:
            density_input = st.number_input(
                "Density, ρ (lbm/ft³)",
                min_value=1.0e-6,
                value=float(safe_query_value("rho", "62.3", float)),
                format="%.6g",
            )
            density_kg_m3 = density_input * LBM_FT3_TO_KG_M3

        if viscosity_basis == "Dynamic viscosity, μ":
            if is_si:
                viscosity_input = st.number_input(
                    "Dynamic viscosity, μ (Pa·s)",
                    min_value=1.0e-12,
                    value=float(safe_query_value("viscosity", "0.001", float)),
                    format="%.8g",
                )
                dynamic_viscosity_pa_s = viscosity_input
            else:
                viscosity_input = st.number_input(
                    "Dynamic viscosity, μ (lbm/(ft·s))",
                    min_value=1.0e-12,
                    value=float(safe_query_value("viscosity", "0.000672", float)),
                    format="%.8g",
                )
                dynamic_viscosity_pa_s = viscosity_input * LBM_FT_S_TO_PA_S
            kinematic_viscosity_m2_s = dynamic_viscosity_pa_s / density_kg_m3
        else:
            if is_si:
                viscosity_input = st.number_input(
                    "Kinematic viscosity, ν (m²/s)",
                    min_value=1.0e-12,
                    value=float(safe_query_value("viscosity", "0.000001", float)),
                    format="%.8g",
                )
                kinematic_viscosity_m2_s = viscosity_input
            else:
                viscosity_input = st.number_input(
                    "Kinematic viscosity, ν (cSt)",
                    min_value=1.0e-8,
                    value=float(safe_query_value("viscosity", "1.0", float)),
                    format="%.8g",
                )
                kinematic_viscosity_m2_s = viscosity_input * CST_TO_M2_S
            dynamic_viscosity_pa_s = kinematic_viscosity_m2_s * density_kg_m3

    st.divider()
    st.subheader("Flow conditions")

    if geometry == "Circular pipe":
        input_mode_options = ["Mean velocity", "Volumetric flow rate"]
        mode_query = safe_query_value("mode", "velocity")
        mode_default = 0 if mode_query == "velocity" else 1
        input_mode = st.radio("Specify flow using", input_mode_options, index=mode_default)

        if is_si:
            diameter_input = st.number_input(
                "Inside diameter (m)",
                min_value=1.0e-8,
                value=float(safe_query_value("diameter", "0.05", float)),
                format="%.6g",
            )
            diameter_m = diameter_input
        else:
            diameter_input = st.number_input(
                "Inside diameter (in)",
                min_value=1.0e-6,
                value=float(safe_query_value("diameter", "2.0", float)),
                format="%.6g",
            )
            diameter_m = diameter_input * IN_TO_M

        if input_mode == "Mean velocity":
            if is_si:
                velocity_input = st.number_input(
                    "Mean velocity (m/s)",
                    min_value=1.0e-10,
                    value=float(safe_query_value("velocity", "1.0", float)),
                    format="%.6g",
                )
                velocity_m_s = velocity_input
            else:
                velocity_input = st.number_input(
                    "Mean velocity (ft/s)",
                    min_value=1.0e-10,
                    value=float(safe_query_value("velocity", "3.0", float)),
                    format="%.6g",
                )
                velocity_m_s = velocity_input * FT_TO_M
            flow_rate_m3_s = velocity_m_s * math.pi * diameter_m**2 / 4.0
            flow_rate_input = (
                flow_rate_m3_s if is_si else flow_rate_m3_s / US_GPM_TO_M3_S
            )
        else:
            if is_si:
                flow_rate_input = st.number_input(
                    "Volumetric flow rate (m³/s)",
                    min_value=1.0e-12,
                    value=float(safe_query_value("flow_rate", "0.002", float)),
                    format="%.8g",
                )
                flow_rate_m3_s = flow_rate_input
            else:
                flow_rate_input = st.number_input(
                    "Volumetric flow rate (US gpm)",
                    min_value=1.0e-10,
                    value=float(safe_query_value("flow_rate", "10.0", float)),
                    format="%.6g",
                )
                flow_rate_m3_s = flow_rate_input * US_GPM_TO_M3_S
            velocity_m_s = pipe_velocity_from_flow_rate(flow_rate_m3_s, diameter_m)
            velocity_input = velocity_m_s if is_si else velocity_m_s / FT_TO_M

        characteristic_length_m = diameter_m
        reynolds_number = reynolds_from_dynamic(
            density_kg_m3,
            velocity_m_s,
            characteristic_length_m,
            dynamic_viscosity_pa_s,
        )
        regime, regime_color, regime_note = classify_pipe_flow(reynolds_number)
        critical_reynolds_number = None

    else:
        if is_si:
            velocity_input = st.number_input(
                "Free-stream velocity (m/s)",
                min_value=1.0e-10,
                value=float(safe_query_value("velocity", "10.0", float)),
                format="%.6g",
            )
            x_input = st.number_input(
                "Distance from leading edge, x (m)",
                min_value=1.0e-8,
                value=float(safe_query_value("x", "1.0", float)),
                format="%.6g",
            )
            velocity_m_s = velocity_input
            x_location_m = x_input
        else:
            velocity_input = st.number_input(
                "Free-stream velocity (ft/s)",
                min_value=1.0e-10,
                value=float(safe_query_value("velocity", "30.0", float)),
                format="%.6g",
            )
            x_input = st.number_input(
                "Distance from leading edge, x (ft)",
                min_value=1.0e-8,
                value=float(safe_query_value("x", "3.0", float)),
                format="%.6g",
            )
            velocity_m_s = velocity_input * FT_TO_M
            x_location_m = x_input * FT_TO_M

        with st.expander("Advanced assumption"):
            critical_reynolds_number = st.number_input(
                "Assumed critical local Reynolds number, Reₓ,crit",
                min_value=1.0e3,
                value=float(safe_query_value("critical_re", "500000", float)),
                format="%.6g",
                help=(
                    "Transition is not fixed. Surface roughness, pressure gradient, "
                    "vibration, and free-stream turbulence can move it substantially."
                ),
            )

        characteristic_length_m = x_location_m
        reynolds_number = reynolds_from_dynamic(
            density_kg_m3,
            velocity_m_s,
            characteristic_length_m,
            dynamic_viscosity_pa_s,
        )
        regime, regime_color, regime_note = classify_flat_plate_flow(
            reynolds_number, critical_reynolds_number
        )
        flow_rate_m3_s = None
        flow_rate_input = None
        input_mode = None


# -----------------------------------------------------------------------------
# Results
# -----------------------------------------------------------------------------
st.header("Result")
metric_columns = st.columns(4)
metric_columns[0].metric("Reynolds number", format_number(reynolds_number))
metric_columns[1].metric(
    "Velocity",
    f"{format_number(velocity_m_s)} m/s"
    if is_si
    else f"{format_number(velocity_m_s / FT_TO_M)} ft/s",
)
metric_columns[2].metric(
    "Characteristic length",
    f"{format_number(characteristic_length_m)} m"
    if is_si
    else (
        f"{format_number(characteristic_length_m / IN_TO_M)} in"
        if geometry == "Circular pipe"
        else f"{format_number(characteristic_length_m / FT_TO_M)} ft"
    ),
)
metric_columns[3].metric(
    "Kinematic viscosity",
    f"{format_number(kinematic_viscosity_m2_s)} m²/s"
    if is_si
    else f"{format_number(kinematic_viscosity_m2_s / CST_TO_M2_S)} cSt",
)

st.markdown(
    f"""
    <div class="regime-card">
        <div class="regime-label">Estimated flow regime</div>
        <div class="regime-value" style="color:{regime_color};">{regime}</div>
        <div class="regime-note">{regime_note}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if geometry == "Circular pipe":
    gauge_maximum = max(1.0e5, reynolds_number * 1.5, PIPE_TURBULENT_LIMIT * 5.0)
    gauge = draw_regime_gauge(
        reynolds_number,
        minimum=1.0,
        maximum=gauge_maximum,
        boundaries=[PIPE_LAMINAR_LIMIT, PIPE_TURBULENT_LIMIT],
        labels=["Laminar", "Transition", "Turbulent"],
        colors=[LAMINAR_COLOR, TRANSITION_COLOR, TURBULENT_COLOR],
    )
else:
    gauge_maximum = max(
        critical_reynolds_number * 10.0,
        reynolds_number * 1.5,
        1.0e6,
    )
    gauge = draw_regime_gauge(
        reynolds_number,
        minimum=100.0,
        maximum=gauge_maximum,
        boundaries=[critical_reynolds_number],
        labels=["Likely laminar", "Transition likely"],
        colors=[LAMINAR_COLOR, TURBULENT_COLOR],
    )

st.pyplot(gauge, width="stretch")
plt.close(gauge)


# -----------------------------------------------------------------------------
# Interpretation and visualizations
# -----------------------------------------------------------------------------
if geometry == "Circular pipe":
    detail_left, detail_right = st.columns([1, 1])

    with detail_left:
        st.subheader("Velocity profile")
        profile_figure = draw_pipe_velocity_profile(reynolds_number)
        st.pyplot(profile_figure, width="stretch")
        plt.close(profile_figure)
        st.caption(
            "The laminar parabola is exact for steady, fully developed flow in a "
            "straight circular pipe. The turbulent profile uses an approximate "
            "1/7-power-law shape; the transitional curve is illustrative."
        )

    with detail_right:
        st.subheader("Flow pattern")
        pattern_figure = draw_pipe_flow_pattern(reynolds_number)
        st.pyplot(pattern_figure, width="stretch")
        plt.close(pattern_figure)
        st.caption(
            "This picture communicates the qualitative difference between flow "
            "regimes. It is not a numerical simulation or a prediction of eddy size."
        )

    st.subheader("Calculation")
    st.latex(r"Re_D = \frac{\rho V D}{\mu} = \frac{V D}{\nu}")

    flow_rate_display = (
        f"{format_number(flow_rate_m3_s)} m³/s"
        if is_si
        else f"{format_number(flow_rate_m3_s / US_GPM_TO_M3_S)} US gpm"
    )
    diameter_display = (
        f"{format_number(diameter_m)} m"
        if is_si
        else f"{format_number(diameter_m / IN_TO_M)} in"
    )

    result_data = {
        "Geometry": "Circular pipe",
        "Reynolds number": format_number(reynolds_number),
        "Estimated regime": regime,
        "Mean velocity": (
            f"{format_number(velocity_m_s)} m/s"
            if is_si
            else f"{format_number(velocity_m_s / FT_TO_M)} ft/s"
        ),
        "Inside diameter": diameter_display,
        "Volumetric flow rate": flow_rate_display,
        "Density": f"{density_kg_m3:.8g} kg/m^3",
        "Dynamic viscosity": f"{dynamic_viscosity_pa_s:.8g} Pa*s",
        "Kinematic viscosity": f"{kinematic_viscosity_m2_s:.8g} m^2/s",
    }

    with st.expander("Assumptions and limitations"):
        st.markdown(
            """
            - The characteristic length is the **inside pipe diameter**.
            - The displayed thresholds (2,300 and 4,000) are conventional guides, not universal guarantees.
            - Entrance effects, disturbances, fittings, wall roughness, pulsation, and non-Newtonian behavior can alter transition.
            - The velocity-profile panel assumes a straight circular pipe and fully developed flow.
            - For a noncircular duct, use the hydraulic diameter, \\(D_h=4A/P_w\\), in a dedicated duct calculation.
            """
        )

else:
    transition_x_m = (
        critical_reynolds_number * kinematic_viscosity_m2_s / velocity_m_s
    )
    laminar_delta_m = float(
        laminar_flat_plate_thickness(x_location_m, reynolds_number)
    )
    turbulent_delta_m = float(
        turbulent_flat_plate_thickness(x_location_m, reynolds_number)
    )
    selected_delta_m = (
        laminar_delta_m
        if reynolds_number < critical_reynolds_number
        else turbulent_delta_m
    )

    flat_metrics = st.columns(3)
    flat_metrics[0].metric(
        "Estimated transition location",
        f"{format_number(transition_x_m)} m"
        if is_si
        else f"{format_number(transition_x_m / FT_TO_M)} ft",
    )
    flat_metrics[1].metric(
        "Selected thickness estimate",
        f"{format_number(selected_delta_m * 1000.0)} mm"
        if is_si
        else f"{format_number(selected_delta_m / IN_TO_M)} in",
    )
    flat_metrics[2].metric(
        "Local Reₓ / critical Reₓ",
        f"{reynolds_number / critical_reynolds_number:.3f}",
    )

    st.subheader("Boundary-layer development")
    boundary_layer_figure = draw_flat_plate_boundary_layer(
        velocity_m_s,
        kinematic_viscosity_m2_s,
        x_location_m,
        critical_reynolds_number,
    )
    st.pyplot(boundary_layer_figure, width="stretch")
    plt.close(boundary_layer_figure)

    st.caption(
        "The two curves are reference correlations, not a single continuous transition "
        "model. The turbulent correlation assumes a turbulent layer from the leading edge, "
        "so use it as an estimate rather than a precise post-transition prediction."
    )

    equation_left, equation_right = st.columns(2)
    with equation_left:
        st.markdown("**Local Reynolds number**")
        st.latex(r"Re_x = \frac{\rho V_\infty x}{\mu} = \frac{V_\infty x}{\nu}")
    with equation_right:
        st.markdown("**Reference thickness correlations**")
        st.latex(r"\delta_{lam} \approx \frac{5x}{\sqrt{Re_x}}")
        st.latex(r"\delta_{turb} \approx \frac{0.37x}{Re_x^{1/5}}")

    result_data = {
        "Geometry": "Flat plate boundary layer",
        "Local Reynolds number Re_x": format_number(reynolds_number),
        "Estimated regime": regime,
        "Free-stream velocity": (
            f"{format_number(velocity_m_s)} m/s"
            if is_si
            else f"{format_number(velocity_m_s / FT_TO_M)} ft/s"
        ),
        "Distance from leading edge": (
            f"{format_number(x_location_m)} m"
            if is_si
            else f"{format_number(x_location_m / FT_TO_M)} ft"
        ),
        "Assumed critical Re_x": format_number(critical_reynolds_number),
        "Estimated transition location": f"{transition_x_m:.8g} m",
        "Laminar thickness reference": f"{laminar_delta_m:.8g} m",
        "Fully turbulent thickness reference": f"{turbulent_delta_m:.8g} m",
        "Density": f"{density_kg_m3:.8g} kg/m^3",
        "Dynamic viscosity": f"{dynamic_viscosity_pa_s:.8g} Pa*s",
        "Kinematic viscosity": f"{kinematic_viscosity_m2_s:.8g} m^2/s",
    }

    with st.expander("Assumptions and limitations"):
        st.markdown(
            """
            - The model represents an incompressible external flow over a **smooth flat plate** with approximately zero pressure gradient.
            - The local characteristic length is the distance \\(x\\) from the leading edge.
            - The default \\(Re_{x,crit}=5\\times10^5\\) is an engineering assumption, not a universal transition point.
            - Surface roughness, pressure gradient, acoustic/vibration disturbances, and free-stream turbulence can move transition substantially.
            - The thickness equations are approximate 99% boundary-layer correlations and do not resolve the transition region.
            - Compressibility, heat transfer, curvature, separation, and three-dimensional effects are outside this calculator's scope.
            """
        )


# -----------------------------------------------------------------------------
# Export and share controls
# -----------------------------------------------------------------------------
st.subheader("Save or share this result")
button_left, button_middle, button_right = st.columns([1, 1, 1.5])

with button_left:
    st.download_button(
        "Download result as CSV",
        data=result_csv(result_data),
        file_name="reynolds_number_result.csv",
        mime="text/csv",
        width="stretch",
    )

with button_middle:
    if st.button("Update shareable URL", width="stretch"):
        st.query_params.clear()
        st.query_params["geometry"] = "pipe" if geometry == "Circular pipe" else "plate"
        st.query_params["units"] = unit_system
        st.query_params["fluid"] = fluid_choice
        st.query_params["velocity"] = f"{velocity_input:.10g}"

        if fluid_choice == "Custom fluid":
            st.query_params["rho"] = f"{density_input:.10g}"
            st.query_params["viscosity"] = f"{viscosity_input:.10g}"
            st.query_params["viscosity_basis"] = (
                "dynamic" if viscosity_basis == "Dynamic viscosity, μ" else "kinematic"
            )

        if geometry == "Circular pipe":
            st.query_params["mode"] = (
                "velocity" if input_mode == "Mean velocity" else "flow_rate"
            )
            st.query_params["diameter"] = f"{diameter_input:.10g}"
            st.query_params["flow_rate"] = f"{flow_rate_input:.10g}"
        else:
            st.query_params["x"] = f"{x_input:.10g}"
            st.query_params["critical_re"] = f"{critical_reynolds_number:.10g}"
        st.success("The browser URL now contains these inputs. Copy it from the address bar.")

with button_right:
    st.caption(
        "The browser URL stores the selected geometry, units, fluid properties, and "
        "primary flow inputs so another visitor can reproduce the calculation."
    )


# -----------------------------------------------------------------------------
# Educational and business CTA sections
# -----------------------------------------------------------------------------
if YOUTUBE_VIDEO_URL:
    st.divider()
    st.subheader("See the experiment")
    st.video(YOUTUBE_VIDEO_URL)

st.divider()
st.markdown(
    """
    <div class="cta-card">
        <strong>Go beyond the number.</strong><br>
        Reynolds number predicts the relative importance of inertia and viscosity, but
        geometry, roughness, disturbances, and boundary conditions determine what happens
        in a real system. The Fluids Engineer builds experiments and interactive tools to
        connect the equation to the physical flow.
    </div>
    """,
    unsafe_allow_html=True,
)

if NEWSLETTER_URL:
    st.link_button("Get the Pipe Flow Quick Reference", NEWSLETTER_URL, width="stretch")

st.caption(
    "Educational engineering estimate only. Verify design decisions using applicable "
    "codes, validated correlations, test data, and qualified engineering review."
)
