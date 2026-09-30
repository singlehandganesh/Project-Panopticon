# Project Panopticon — VS Code Web App

A separate Streamlit localhost interface for the original Project Panopticon notebook.

## Preserved original files
- `Project_Panopticon.ipynb` — unchanged
- `video_telemetry.csv` — unchanged
- `system_events.csv` — unchanged

## Added files
- `app.py` — Streamlit web dashboard
- `requirements.txt` — dependencies
- `run_app.bat` — Windows launcher

## Run in VS Code
Open this folder in VS Code, then in Terminal:

```bat
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the localhost URL printed by Streamlit, normally `http://localhost:8501`.

The app performs the same core workflow as the notebook: 1-second backward `merge_asof`, missing-value repair, 10-second rolling features, 300-tree class-balanced Random Forest, 50% comparison, 90% strict threshold, threshold metrics, PR curves, feature importance, and classification report.

You can use the bundled CSVs or upload replacement CSVs from the sidebar.

### Windows note
The earlier local environment was blocked by Windows Application Control when loading a scikit-learn `.pyd` file. Streamlit does not bypass that policy. If Windows still blocks the compiled dependency, the Python environment must be allowed by Windows or run in an environment where the dependency is permitted.
