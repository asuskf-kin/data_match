import time
from pathlib import Path

import polars as pl


def analyze_issue_impact(df: pl.DataFrame, issue_name: str, total_count: int) -> dict:
    """Analyzes a specific diagnostic flag to extract metrics and the

    Top 5 most frequent records or affected areas (including municipalities and bottlers).
    """
    issue_df = df.filter(pl.col(issue_name) == True)
    flagged_count = len(issue_df)
    remaining_count = total_count - flagged_count

    top_dropped = []
    top_bottlers = []  # Nuevo contenedor para el Top 5 de estados con embotelladoras

    if flagged_count > 0:
        # 1. Prioritize geographic grouping if analyzing geo issues
        if "geo" in issue_name.lower() and "municipality" in issue_df.columns:
            top_counts = (
                issue_df.group_by(["municipality", "state"])
                .agg(pl.len().alias("freq"))
                .sort("freq", descending=True)
                .head(5)
            )
            top_dropped = [
                (f"Municipio: {row[0]} | Estado: {row[1]}", row[2])
                for row in top_counts.iter_rows()
            ]

            # Capturar el Top 5 de Estados con más embotelladoras únicas
            if "bottler" in issue_df.columns and "state" in issue_df.columns:
                bottler_counts = (
                    df.filter(
                        pl.col("state").is_not_null() & pl.col("bottler").is_not_null()
                    )
                    .unique(["state", "bottler"])
                    .group_by("state")
                    .agg(pl.len().alias("unique_bottlers"))
                    .sort("unique_bottlers", descending=True)
                    .head(5)
                )
                top_bottlers = [(row[0], row[1]) for row in bottler_counts.iter_rows()]
        else:
            # 2. Default name/identifier column check
            name_col = None
            for col in ["name", "business_name", "title", "dataplor_id"]:
                if col in issue_df.columns:
                    name_col = col
                    break

            if name_col:
                top_counts = (
                    issue_df.group_by(name_col)
                    .agg(pl.len().alias("freq"))
                    .sort("freq", descending=True)
                    .head(5)
                )
                top_dropped = [(row[0], row[1]) for row in top_counts.iter_rows()]
            else:
                top_dropped = [
                    ("Records without a detected name column", flagged_count)
                ]

    return {
        "step_name": issue_name.replace("_", " ").title(),
        "dropped_count": flagged_count,
        "remaining_count": remaining_count,
        "top_dropped": top_dropped,
        "top_bottlers": top_bottlers,
        "is_geo": "geo" in issue_name.lower(),
    }


def generate_html_report(metrics_list: list[dict], reports_dir: Path) -> Path:
    """
    Generates an interactive HTML file highlighting critical issues (like geo-outliers and municipalities)
    alongside a Chart.js graphical representation.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    html_file = reports_dir / f"diagnostic_report_{timestamp}.html"

    steps_labels = [m["step_name"] for m in metrics_list]
    dropped_data = [m["dropped_count"] for m in metrics_list]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Diagnostic Pipeline Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; }}
        .chart-container {{ position: relative; height: 400px; width: 100%; margin-top: 20px; }}
        .step-card {{ background: #fdfdfd; border-left: 4px solid #3498db; padding: 15px; margin: 15px 0; border-radius: 4px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }}
        .step-card.geo-highlight {{ border-left: 4px solid #e74c3c; background: #fff5f5; }}
        .step-title {{ font-weight: bold; color: #2980b9; font-size: 1.1em; }}
        .geo-highlight .step-title {{ color: #c0392b; }}
        ul {{ margin: 5px 0; padding-left: 20px; list-style-type: none; }}
        li {{ font-size: 0.95em; color: #444; margin-bottom: 4px; }}
        .count-badge {{ background-color: #e74c3c; color: white; padding: 2px 8px; border-radius: 12px; font-weight: bold; font-size: 0.85em; margin-right: 8px; display: inline-block; min-width: 20px; text-align: center; }}
        .highlight-tag {{ background-color: #f39c12; color: white; font-size: 0.75em; padding: 2px 6px; border-radius: 4px; vertical-align: middle; margin-left: 8px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Data Health Diagnostic Report (Municipality & Geo Insights)</h1>
        <p>Generated at: <strong>{time.strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
        
        <div class="chart-container">
            <canvas id="reportChart"></canvas>
        </div>

        <h2>Issue Details & Impact Breakdown</h2>
"""

    for m in metrics_list:
        card_class = "step-card geo-highlight" if m["is_geo"] else "step-card"
        tag = (
            "<span class='highlight-tag'>Geo / Municipality Impact</span>"
            if m["is_geo"]
            else ""
        )

        html_content += f"""
        <div class="{card_class}">
            <div class="step-title">{m["step_name"]} {tag}</div>
            <p>Records flagged: <strong>{m["dropped_count"]:,}</strong> | Unaffected: {m["remaining_count"]:,}</p>
"""
        if m["top_dropped"]:
            html_content += "<ul>"
            for name, freq in m["top_dropped"]:
                html_content += f"<li><span class='count-badge'>{freq}</span> <code>{name}</code></li>"
            html_content += "</ul>"
        else:
            html_content += "<p style='color: #888; font-size: 0.9em;'>No records flagged for this issue.</p>"

        # Renderizar la sección adicional de Top 5 Embotelladoras si existe
        if m.get("top_bottlers"):
            html_content += """
            <div style="margin-top: 12px; border-top: 1px dashed #dcdcdc; padding-top: 8px;">
                <strong style="font-size: 0.9em; color: #c0392b;">🏆 Top 5 States with Most Bottlers:</strong>
                <ul style="margin-top: 4px;">
            """
            for state_name, b_count in m["top_bottlers"]:
                html_content += f"<li><span class='count-badge' style='background-color: #f39c12;'>{b_count}</span> <code>State: {state_name}</code></li>"
            html_content += "</ul></div>"

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
                    label: 'Flagged Records',
                    data: {dropped_data},
                    backgroundColor: 'rgba(231, 76, 60, 0.7)',
                    borderColor: 'rgba(192, 57, 43, 1)',
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

    print(f" -> Interactive HTML report generated at: {html_file}")
    return html_file


def generate_and_save_reports(
    df: pl.DataFrame,
    severe_issues: list,
    minor_issues: list,
    reports_dir: str,
):
    """
    Exports the final Parquet dataset, CSV summary, and the Interactive HTML report.
    """
    print("Exporting reports...")
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total_records = len(df)

    export_path = out_dir / "health_flags.parquet"
    df.write_parquet(export_path)

    all_issues = severe_issues + minor_issues
    metrics_list = []

    for issue in all_issues:
        metrics = analyze_issue_impact(df, issue, total_records)
        metrics_list.append(metrics)

    html_path = generate_html_report(metrics_list, out_dir)

    print("✅ Process completed.")
    print(f" -> Dataset saved at: {export_path}")

    if "is_usable" in df.columns:
        usable_count = df.select(pl.col("is_usable").sum()).item()
        print(
            f"\nSummary: {usable_count:,} usable records ({(usable_count / total_records * 100):.1f}%)"
        )
