import streamlit as st

st.title("Reynolds Number Calculator")

velocity = st.number_input("Velocity (m/s)", min_value=0.0)
diameter = st.number_input("Diameter (m)", min_value=0.0)
density = st.number_input("Density (kg/m³)", value=1000.0)
viscosity = st.number_input("Dynamic Viscosity (Pa·s)", value=0.001)

if viscosity > 0:
    Re = density * velocity * diameter / viscosity

    st.metric("Reynolds Number", f"{Re:,.0f}")

    if Re < 2300:
        st.success("Laminar")
    elif Re < 4000:
        st.warning("Transitional")
    else:
        st.error("Turbulent")