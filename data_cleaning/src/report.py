import logging
import time
from pathlib import Path

import polars as pl


def analyze_step_drop(
    df_before: pl.DataFrame, df_after: pl.DataFrame, step_name: str
) -> dict:
    """
    Compares the DataFrame before and after a module to extract metrics
    and the Top 5 most frequent dropped records dynamically based on available columns.
    """
    count_before = len(df_before)
    count_after = len(df_after)
    dropped_count = count_before - count_after

    top_dropped = []
    if dropped_count > 0:
        # Dynamically detect which name/identifier column exists
        name_col = None
        for col in ["name_normalized", "name", "business_name", "title"]:
            if col in df_before.columns:
                name_col = col
                break

        # Build join keys based on existing common columns
        join_keys = []
        if name_col and name_col in df_after.columns:
            join_keys.append(name_col)
        for c in ["latitude", "longitude", "id", "place_id"]:
            if c in df_before.columns and c in df_after.columns:
                join_keys.append(c)

        if join_keys:
            # Use anti-join to isolate the exact dropped rows
            dropped_df = df_before.join(df_after, on=join_keys, how="anti")
        else:
            # If no clear join keys exist, safely skip the difference report
            dropped_df = pl.DataFrame()

        if len(dropped_df) > 0 and name_col and name_col in dropped_df.columns:
            top_counts = (
                dropped_df.group_by(name_col)
                .agg(pl.len().alias("freq"))
                .sort("freq", descending=True)
                .head(5)
            )
            top_dropped = [(row[0], row[1]) for row in top_counts.iter_rows()]
        elif len(dropped_df) > 0:
            top_dropped = [("Records without a detected name column", len(dropped_df))]

    return {
        "step_name": step_name,
        "dropped_count": dropped_count,
        "remaining_count": count_after,
        "top_dropped": top_dropped,
    }


def generate_html_report(metrics_list: list[dict], data_dir: Path) -> Path:
    """
    Generates an interactive HTML file with a bar chart (Chart.js)
    showing filtered records per step and the Top 5 discarded items.
    """
    reports_dir = data_dir.parent / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    html_file = reports_dir / f"pipeline_report_{timestamp}.html"

    steps_labels = [m["step_name"] for m in metrics_list]
    dropped_data = [m["dropped_count"] for m in metrics_list]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Pipeline Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; margin-top: 20px; }}
        .step-card {{ background: #fdfdfd; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
        .step-title {{ font-weight: bold; color: #2980b9; font-size: 1.1em; }}
        ul {{ margin: 5px 0; padding-left: 20px; list-style-type: none; }}
        li {{ font-size: 0.95em; color: #444; margin-bottom: 4px; }}
        .count-badge {{ background-color: #e74c3c; color: white; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.85em; margin-right: 8px; display: inline-block; min-width: 20px; text-align: center; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Pipeline Execution Report</h1>
        <p>Generated at: <strong>{time.strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
        
        <div class="chart-container">
            <canvas id="reportChart"></canvas>
        </div>

        <h2>Step Details (Top 5 Filtered Records)</h2>
"""

    for m in metrics_list:
        html_content += f"""
        <div class="step-card">
            <div class="step-title">{m["step_name"]}</div>
            <p>Filtered records in this step: <strong>{m["dropped_count"]:,}</strong> | Remaining: {m["remaining_count"]:,}</p>
"""
        if m["top_dropped"]:
            html_content += "<ul>"
            for name, freq in m["top_dropped"]:
                # Replaced "({freq} times)" with a clean numerical badge and format
                html_content += f"<li><span class='count-badge'>{freq}</span> <code>{name}</code></li>"
            html_content += "</ul>"
        else:
            html_content += "<p style='color: #888; font-size: 0.9em;'>No prominent exclusions recorded.</p>"
        html_content += "</div>"

    html_content += f"""
    </div>
    <script>
        const ctx = document.getElementById('reportChart').getContext('2d');
        const reportChart = new Chart(ctx, {{
            type: 'bar',
            data: {{
                labels: {steps_labels},
                datasets: [{{
                    label: 'Filtered Records',
                    data: {dropped_data},
                    backgroundColor: 'rgba(52, 152, 219, 0.7)',
                    borderColor: 'rgba(41, 128, 185, 1)',
                    borderWidth: 1,
                    borderRadius: 4
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                scales: {{
                    y: {{ beginAtZero: true }}
                }}
            }}
        }});
    </script>
</body>
</html>
"""

    with open(html_file, "w", encoding="utf-8") as f:
        f.write(html_content)

    logging.info(f"Interactive HTML report generated at: {html_file}")
    return html_file
