import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge

st.set_page_config(page_title="Fluid Flow Visualizer", layout="wide")

# =========================
# Gauge Function
# =========================
def draw_gauge(value, max_value, flow_type):

    fig, ax = plt.subplots(figsize=(4, 2.5))

    # Colored regions
    if flow_type == "Pipe Flow":

        # Laminar
        ax.add_patch(
            Wedge((0, 0), 1, 180, 128, color="green", alpha=0.6)
        )

        # Transitional
        ax.add_patch(
            Wedge((0, 0), 1, 128, 90, color="orange", alpha=0.6)
        )

        # Turbulent
        ax.add_patch(
            Wedge((0, 0), 1, 90, 0, color="red", alpha=0.6)
        )

        ax.text(-0.85, 0.1, "Laminar", fontsize=8)
        ax.text(-0.15, 1.05, "Transition", fontsize=8)
        ax.text(0.60, 0.1, "Turbulent", fontsize=8)

    else:

        laminar_fraction = 5e5 / max_value
        laminar_angle = 180 * laminar_fraction

        ax.add_patch(
            Wedge(
                (0, 0),
                1,
                180,
                180 - laminar_angle,
                color="green",
                alpha=0.6
            )
        )

        ax.add_patch(
            Wedge(
                (0, 0),
                1,
                180 - laminar_angle,
                0,
                color="red",
                alpha=0.6
            )
        )

        ax.text(-0.85, 0.1, "Laminar", fontsize=8)
        ax.text(0.60, 0.1, "Turbulent", fontsize=8)

    # Needle
    fraction = min(value, max_value) / max_value
    angle = np.pi * (1 - fraction)

    ax.plot(
        [0, 0.85 * np.cos(angle)],
        [0, 0.85 * np.sin(angle)],
        linewidth=4
    )

    ax.plot(0, 0, marker='o', markersize=8)

    ax.text(
        0,
        -0.25,
        f"Re = {value:,.0f}",
        ha='center',
        fontsize=11,
        fontweight='bold'
    )

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.4, 1.2)
    ax.axis("off")

    return fig


# =========================
# Title
# =========================
st.title("The Fluids Engineer - Flow Visualizer")

flow_type = st.selectbox(
    "Flow Geometry",
    ["Pipe Flow", "Flat Plate"]
)

# =========================
# Sidebar Inputs
# =========================
st.sidebar.header("Fluid Properties")

velocity = st.sidebar.number_input(
    "Velocity (m/s)",
    min_value=0.01,
    value=1.0
)

density = st.sidebar.number_input(
    "Density (kg/m³)",
    min_value=0.01,
    value=1000.0
)

viscosity = st.sidebar.number_input(
    "Dynamic Viscosity (Pa·s)",
    min_value=0.000001,
    value=0.001,
    format="%.6f"
)

# =========================
# Reynolds Number
# =========================
if flow_type == "Pipe Flow":

    diameter = st.sidebar.number_input(
        "Pipe Diameter (m)",
        min_value=0.001,
        value=0.05
    )

    Re = density * velocity * diameter / viscosity

    if Re < 2300:
        regime = "Laminar"
        color = "green"
    elif Re < 4000:
        regime = "Transitional"
        color = "orange"
    else:
        regime = "Turbulent"
        color = "red"

else:

    x_location = st.sidebar.number_input(
        "Distance from Leading Edge (m)",
        min_value=0.001,
        value=1.0
    )

    Re = density * velocity * x_location / viscosity

    if Re < 5e5:
        regime = "Laminar"
        color = "green"
    else:
        regime = "Turbulent"
        color = "red"

# =========================
# Results Layout
# =========================
st.header("Results")

left_col, right_col = st.columns([1, 1])

# =========================
# Left Side
# =========================
with left_col:

    st.metric(
        "Reynolds Number",
        f"{Re:,.0f}"
    )

    st.markdown(
        f"### Flow Regime: :{color}[{regime}]"
    )

    if flow_type == "Pipe Flow":

        st.subheader("Flow Regime Gauge")

        fig_gauge = draw_gauge(
            Re,
            8000,
            "Pipe Flow"
        )

    else:

        st.subheader("Boundary Layer Regime Gauge")

        fig_gauge = draw_gauge(
            Re,
            5e6,
            "Flat Plate"
        )

    st.pyplot(fig_gauge)

# =========================
# Right Side
# =========================
with right_col:

    if flow_type == "Pipe Flow":

        st.subheader("Velocity Profile")

        y = np.linspace(-1, 1, 200)

        if Re < 2300:
            u = 1 - y**2

        elif Re < 4000:
            u = np.maximum(
                0,
                1 - y**2
            )**0.5

        else:
            u = np.maximum(
                0,
                1 - np.abs(y)
            )**0.15

        fig, ax = plt.subplots(
            figsize=(4, 3)
        )

        ax.plot(u, y)

        ax.set_xlabel(
            "Normalized Velocity"
        )

        ax.set_ylabel(
            "Pipe Radius"
        )

        ax.set_title(
            "Pipe Velocity Profile"
        )

        st.pyplot(fig)

        st.subheader(
            "Flow Visualization"
        )

        fig2, ax2 = plt.subplots(
            figsize=(5, 1.5)
        )

        ax2.set_xlim(0, 10)
        ax2.set_ylim(-1, 1)

        for yline in np.linspace(
            -0.9,
            0.9,
            12
        ):

            x = np.linspace(
                0,
                10,
                400
            )

            if Re < 2300:

                yy = (
                    np.ones_like(x)
                    * yline
                )

            elif Re < 4000:

                yy = (
                    yline
                    + 0.03*np.sin(4*x)
                )

            else:

                yy = (
                    yline
                    + 0.08*np.sin(10*x)
                )

            ax2.plot(x, yy)

        ax2.set_title(
            "Qualitative Flow Pattern"
        )

        ax2.axis("off")

        st.pyplot(fig2)

    else:

        st.subheader(
            "Boundary Layer Development"
        )

        x = np.linspace(
            0.01,
            5,
            500
        )

        delta = 5 * np.sqrt(
            viscosity * x /
            (density * velocity)
        )

        fig, ax = plt.subplots(
            figsize=(4, 3)
        )

        ax.plot(x, delta)

        ax.fill_between(
            x,
            0,
            delta,
            alpha=0.3
        )

        ax.set_xlabel(
            "Distance Along Plate (m)"
        )

        ax.set_ylabel(
            "Boundary Layer Thickness"
        )

        ax.set_title(
            "Boundary Layer Growth"
        )

        st.pyplot(fig)

        transition_re = 5e5

        transition_x = (
            transition_re * viscosity
        ) / (
            density * velocity
        )

        st.subheader(
            "Laminar-Turbulent Transition"
        )

        st.write(
            f"Estimated transition occurs near x = {transition_x:.3f} m"
        )

        fig2, ax2 = plt.subplots(
            figsize=(5, 1.5)
        )

        ax2.set_xlim(0, 5)
        ax2.set_ylim(0, 1)

        ax2.plot(
            [0, 5],
            [0.2, 0.2],
            linewidth=4
        )

        ax2.axvline(
            transition_x,
            linestyle="--"
        )

        ax2.text(
            transition_x,
            0.6,
            "Transition"
        )

        ax2.set_title(
            "Laminar → Turbulent"
        )

        ax2.axis("off")

        st.pyplot(fig2)
