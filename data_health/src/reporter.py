# src/reporter.py
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
    top_bottlers = []

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


def generate_html_report(
    metrics_list: list[dict],
    reports_dir: Path,
    usable_count: int,
    total_records: int,
    spatial_summary: dict,
) -> Path:
    """
    Generates an interactive HTML file highlighting critical issues and
    including a prominent Executive Summary banner at the top.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    html_file = reports_dir / f"diagnostic_report_{timestamp}.html"

    steps_labels = [m["step_name"] for m in metrics_list]
    dropped_data = [m["dropped_count"] for m in metrics_list]
    usable_pct = (usable_count / total_records * 100) if total_records > 0 else 0

    # Format the spatial distribution lines
    dist_html = ""
    for row in spatial_summary.get("distribution", []):
        dist_html += f"<div style='padding: 3px 0; border-bottom: 1px solid rgba(255,255,255,0.1);'>➔ State: {row['state']} | Bottler: {row['bottler']} | Municipality: {row['municipality']} | <strong style='color: #a29bfe;'>Points: {row['count']:,}</strong></div>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Diagnostic Pipeline Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; background-color: #f8f9fa; color: #333; margin: 0; padding: 20px; }}
        .container {{ max-width: 1000px; margin: auto; background: white; padding: 30px; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); }}
        h1 {{ color: #2c3e50; border-bottom: 2px solid #eaeaea; padding-bottom: 10px; margin-top: 0; }}
        
        /* EXECUTIVE BANNER STYLES */
        .exec-banner {{ background: linear-gradient(135deg, #1e272e 0%, #2f3640 100%); color: #f5f6fa; padding: 25px; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 6px 15px rgba(0,0,0,0.15); border-left: 6px solid #4cd137; }}
        .exec-banner h2 {{ margin-top: 0; color: #fbc531; border-bottom: 1px solid rgba(255,255,255,0.1); padding-bottom: 10px; font-size: 1.6em; }}
        .exec-metrics {{ display: flex; justify-content: space-between; font-size: 1.3em; margin-bottom: 20px; background: rgba(0,0,0,0.2); padding: 15px; border-radius: 8px; }}
        .exec-metrics span {{ color: #4cd137; font-weight: bold; font-size: 1.2em; }}
        .spatial-dist {{ background: rgba(0,0,0,0.3); padding: 15px; border-radius: 6px; max-height: 220px; overflow-y: auto; font-family: 'Courier New', Courier, monospace; font-size: 0.9em; margin-top: 10px; }}
        /* SCROLLBAR */
        .spatial-dist::-webkit-scrollbar {{ width: 8px; }}
        .spatial-dist::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.1); border-radius: 4px; }}
        .spatial-dist::-webkit-scrollbar-thumb {{ background: rgba(255,255,255,0.3); border-radius: 4px; }}
        
        .chart-container {{ position: relative; height: 400px; width: 100%; margin-top: 20px; margin-bottom: 40px; }}
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
        <h1>📊 Data Health Diagnostic Report</h1>
        <p style="color: #7f8c8d; font-size: 0.9em;">Generated at: <strong>{time.strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
        
        <!-- NEW DISTINCTIVE EXECUTIVE BANNER -->
        <div class="exec-banner">
            <h2>Executive Summary</h2>
            <div class="exec-metrics">
                <div>Total Records Processed: <br><strong>{total_records:,}</strong></div>
                <div style="text-align: right;">Total Usable Records: <br><span>{usable_count:,}</span> ({usable_pct:.1f}%)</div>
            </div>
            
            <h3 style="color: #dcdde1; margin-bottom: 10px; font-size: 1.2em;">📍 Spatial Evaluation Summary</h3>
            <ul style="list-style: none; padding: 0; margin: 0 0 15px 0; color: #f5f6fa;">
                <li>Total Evaluated Points: <strong>{spatial_summary.get("total", 0):,}</strong></li>
                <li>❌ Spatial Outliers (Outside area): <strong style="color: #e84118;">{spatial_summary.get("outliers", 0):,}</strong></li>
                <li>✅ Retained Points (Inside area): <strong style="color: #4cd137;">{spatial_summary.get("retained", 0):,}</strong></li>
            </ul>

            <h4 style="color: #dcdde1; margin-bottom: 8px;">Distribution of Retained Points:</h4>
            <div class="spatial-dist">
                {dist_html if dist_html else "<div style='color: #7f8fa6;'>No spatial distribution data available.</div>"}
            </div>
        </div>

        <h2>Issue Details & Impact Breakdown</h2>
        <div class="chart-container">
            <canvas id="reportChart"></canvas>
        </div>
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

    # 1. Calculate usable count
    usable_count = 0
    if "is_usable" in df.columns:
        usable_count = df.select(pl.col("is_usable").sum()).item()

    # 2. Extract Spatial Evaluation Summary for the HTML Banner
    spatial_summary = {
        "total": total_records,
        "outliers": 0,
        "retained": total_records,
        "distribution": [],
    }
    if "geo_outlier" in df.columns and "state" in df.columns:
        outliers_count = df.select(pl.col("geo_outlier").sum()).item()
        retained_df = df.filter(~pl.col("geo_outlier") & pl.col("state").is_not_null())
        retained_count = len(retained_df)

        # Group to mimic the distribution printed in terminal
        distribution = (
            retained_df.group_by(["state", "bottler", "municipality"])
            .agg(pl.len().alias("count"))
            .sort("count", descending=True)
            .to_dicts()
        )

        spatial_summary = {
            "total": total_records,
            "outliers": outliers_count,
            "retained": retained_count,
            "distribution": distribution,
        }

    # 3. Export Parquet
    export_path = out_dir / "health_flags.parquet"
    df.write_parquet(export_path)

    # 4. Generate Issue Impact Metrics
    all_issues = severe_issues + minor_issues
    metrics_list = []
    for issue in all_issues:
        metrics = analyze_issue_impact(df, issue, total_records)
        metrics_list.append(metrics)

    # 5. Build the HTML report with the new banner
    html_path = generate_html_report(
        metrics_list=metrics_list,
        reports_dir=out_dir,
        usable_count=usable_count,
        total_records=total_records,
        spatial_summary=spatial_summary,
    )

    print("✅ Process completed.")
    print(f" -> Dataset saved at: {export_path}")
    print(
        f"\nSummary: {usable_count:,} usable records ({(usable_count / total_records * 100):.1f}%)"
    )
