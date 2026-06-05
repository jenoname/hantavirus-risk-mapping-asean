from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd

from config import GRID_RESOLUTION_DEGREES, MAP_DIR, MODEL_DIR, RAW_DIR, ensure_directories
from scripts.common import load_dataset
from scripts.patch_model import predict_patch_model


def color_for(prob: float) -> str:
    if prob < 0.3:
        return "#2ca25f"
    if prob <= 0.6:
        return "#fee08b"
    return "#d73027"


def main() -> None:
    ensure_directories()
    df = load_dataset()
    rf = joblib.load(MODEL_DIR / "model_rf.pkl")
    cnn = joblib.load(MODEL_DIR / "model_cnn.pkl")
    weights = json.loads((MODEL_DIR / "ensemble_weights.json").read_text(encoding="utf-8"))
    p_a = rf["model"].predict_proba(df[rf["features"]])[:, 1]
    p_b = predict_patch_model(cnn, df)
    df = df.copy()
    df["risk_probability"] = weights["w_model_a_rf"] * p_a + weights["w_model_b_cnn"] * p_b
    df["risk_tier"] = df["risk_probability"].map(lambda p: "low" if p < 0.3 else "medium" if p <= 0.6 else "high")

    half = GRID_RESOLUTION_DEGREES / 2
    out = MAP_DIR / "risk_map.html"
    try:
        import folium

        m = folium.Map(location=[-2.5, 113.0], zoom_start=5, tiles="CartoDB positron")
        for row in df.itertuples(index=False):
            bounds = [[row.lat - half, row.lon - half], [row.lat + half, row.lon + half]]
            tooltip = (
                f"Cell: {row.cell_id}<br>"
                f"Probability: {row.risk_probability:.3f}<br>"
                f"Tier: {row.risk_tier}<br>"
                f"Presence count: {row.presence_count:.0f}<br>"
                f"Precip: {row.chirps_precip_mm:.1f} mm<br>"
                f"LST: {row.modis_lst_day_c:.1f} C"
            )
            folium.Rectangle(
                bounds=bounds,
                color=color_for(row.risk_probability),
                weight=0.5,
                fill=True,
                fill_color=color_for(row.risk_probability),
                fill_opacity=0.45,
                tooltip=tooltip,
            ).add_to(m)
        gbif_path = RAW_DIR / "gbif_rodent_occurrences_2025_05_to_2026_05.csv"
        if gbif_path.exists():
            gbif = pd.read_csv(gbif_path)
            gbif = gbif.dropna(subset=["decimalLongitude", "decimalLatitude"])
            presence_layer = folium.FeatureGroup(name="GBIF presence points", show=True)
            for row in gbif.itertuples(index=False):
                tooltip = (
                    f"GBIF: {getattr(row, 'gbifID', '')}<br>"
                    f"Species: {getattr(row, 'species', '')}<br>"
                    f"Date: {getattr(row, 'eventDate', '')}<br>"
                    f"Country: {getattr(row, 'country', '')}"
                )
                folium.CircleMarker(
                    location=[row.decimalLatitude, row.decimalLongitude],
                    radius=2.5,
                    color="#111827",
                    weight=0.5,
                    fill=True,
                    fill_color="#111827",
                    fill_opacity=0.75,
                    tooltip=tooltip,
                ).add_to(presence_layer)
            presence_layer.add_to(m)
            folium.LayerControl(collapsed=True).add_to(m)
        m.save(out)
    except ModuleNotFoundError:
        rows = []
        for row in df.itertuples(index=False):
            rows.append(
                f"<tr style='background:{color_for(row.risk_probability)}33'>"
                f"<td>{row.cell_id}</td><td>{row.lon:.2f}</td><td>{row.lat:.2f}</td>"
                f"<td>{row.risk_probability:.3f}</td><td>{row.risk_tier}</td>"
                f"<td>{row.presence_count:.0f}</td><td>{row.chirps_precip_mm:.1f}</td>"
                f"<td>{row.modis_lst_day_c:.1f}</td></tr>"
            )
        html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Hantavirus Risk Map Table</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }
    table { border-collapse: collapse; width: 100%; font-size: 12px; }
    th, td { border: 1px solid #d7dde2; padding: 6px 8px; text-align: right; }
    th:first-child, td:first-child, th:nth-child(5), td:nth-child(5) { text-align: left; }
    th { position: sticky; top: 0; background: #f5f7f9; }
  </style>
</head>
<body>
  <h1>Hantavirus Risk Grid, May 2025-May 2026</h1>
  <p>Folium is not installed, so this fallback file lists the same risk-tier output as an HTML table. Install folium to generate the interactive choropleth map.</p>
  <table>
    <thead><tr><th>cell_id</th><th>lon</th><th>lat</th><th>probability</th><th>tier</th><th>presence</th><th>precip_mm</th><th>lst_c</th></tr></thead>
    <tbody>
""" + "\n".join(rows) + """
    </tbody>
  </table>
</body>
</html>"""
        out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
