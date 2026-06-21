import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Fluid Flow Visualizer", layout="wide")

st.title("The Fluids Engineer - Flow Visualizer")

flow_type = st.selectbox(
"Flow Geometry",
["Pipe Flow", "Flat Plate"]
)

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

if flow_type == "Pipe Flow":

    diameter = st.sidebar.number_input(
        "Pipe Diameter (m)",
        min_value=0.001,
        value=0.05
    )

    Re = density * velocity * diameter / viscosity

else:

    x_location = st.sidebar.number_input(
        "Distance from Leading Edge (m)",
        min_value=0.001,
        value=1.0
    )

    Re = density * velocity * x_location / viscosity

st.header("Results")

col1, col2 = st.columns(2)

with col1:
    st.metric("Reynolds Number", f"{Re:,.0f}")

with col2:
    if Re < 2300:
        regime = "Laminar"
        color = "green"
    elif Re < 4000:
        regime = "Transitional"
        color = "orange"
    else:
        regime = "Turbulent"
        color = "red"

    st.markdown(
        f"### Flow Regime: :{color}[{regime}]"
    )

st.subheader("Flow Regime Gauge")

gauge_max = 10000

normalized = min(Re, gauge_max)

st.progress(normalized / gauge_max)

st.caption(
"Laminar < 2300 | Transitional 2300-4000 | Turbulent > 4000"
)

if flow_type == "Pipe Flow":

    st.subheader("Velocity Profile")

    y = np.linspace(-1, 1, 200)

    if Re < 2300:
        u = 1 - y**2
    elif Re < 4000:
        u = (1 - y**2)**0.5
    else:
        u = (1 - np.abs(y))**0.15

    fig, ax = plt.subplots()

    ax.plot(u, y)

    ax.set_xlabel("Normalized Velocity")
    ax.set_ylabel("Pipe Radius")
    ax.set_title("Pipe Velocity Profile")

    st.pyplot(fig)

    st.subheader("Flow Visualization")

    fig2, ax2 = plt.subplots(figsize=(8, 2))

    ax2.set_xlim(0, 10)
    ax2.set_ylim(-1, 1)

    for yline in np.linspace(-0.9, 0.9, 12):

        x = np.linspace(0, 10, 400)

        if Re < 2300:
            yy = np.ones_like(x) * yline
        elif Re < 4000:
            yy = yline + 0.03 * np.sin(4 * x)
        else:
            yy = yline + 0.08 * np.sin(10 * x)

        ax2.plot(x, yy)

    ax2.set_title("Qualitative Flow Pattern")
    ax2.axis("off")

    st.pyplot(fig2)

else:

    st.subheader("Boundary Layer Development")

    x = np.linspace(0.01, 5, 500)

    delta = 5 * np.sqrt(
        viscosity * x / (density * velocity)
    )

    fig, ax = plt.subplots()

    ax.plot(x, delta)

    ax.fill_between(x, 0, delta, alpha=0.3)

    ax.set_xlabel("Distance Along Plate (m)")
    ax.set_ylabel("Boundary Layer Thickness")
    ax.set_title("Boundary Layer Growth")

    st.pyplot(fig)

    transition_re = 5e5

    transition_x = (
        transition_re * viscosity
    ) / (density * velocity)

    st.subheader("Transition Location")

    st.write(
        f"Estimated transition begins near x = {transition_x:.3f} m"
    )

    fig2, ax2 = plt.subplots(figsize=(8, 2))

    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 1)

    ax2.plot([0, 5], [0.2, 0.2], linewidth=4)

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
        "Laminar → Transitional → Turbulent"
    )

    ax2.axis("off")

    st.pyplot(fig2)
