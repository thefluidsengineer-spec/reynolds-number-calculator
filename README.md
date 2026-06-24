# The Fluids Engineer — Reynolds Number Calculator

A Streamlit app for internal circular-pipe flow and external flat-plate boundary layers.

## Run locally

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

macOS/Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Before publishing

At the top of `app.py`, set:

- `YOUTUBE_VIDEO_URL`
- `NEWSLETTER_URL`

The calculator performs all math in SI units internally. U.S. customary inputs are converted before calculation.

## Scope

- Circular-pipe Reynolds number using inside diameter
- Velocity or volumetric-flow-rate input
- Dynamic or kinematic viscosity for custom fluids
- Flat-plate local Reynolds number and approximate boundary-layer thickness correlations
- CSV export and shareable query parameters

This is an educational engineering estimate, not a substitute for code-compliant design or validated analysis.
