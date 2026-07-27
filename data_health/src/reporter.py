import time
from pathlib import Path

import polars as pl


def analyze_issue_impact(df: pl.DataFrame, issue_name: str, total_count: int) -> dict:
    """Analyzes a specific diagnostic flag to extract metrics and the
    Top 5 most frequent records or affected areas.
    """
    issue_df = df.filter(pl.col(issue_name) == True)
    flagged_count = len(issue_df)
    remaining_count = total_count - flagged_count

    top_dropped = []

    if flagged_count > 0:
        # 1. Prioritize geographic grouping if analyzing geo issues
        if "geo" in issue_name.lower():
            geo_cols = []
            if "state" in issue_df.columns:
                geo_cols.append("state")
            if "municipality" in issue_df.columns:
                geo_cols.append("municipality")
            
            # If we have geographic columns to group by
            if geo_cols:
                top_counts = (
                    issue_df.group_by(geo_cols)
                    .agg(pl.len().alias("freq"))
                    .sort("freq", descending=True)
                    .head(5)
                )
                
                # Format dynamically based on existing columns
                for row in top_counts.iter_rows():
                    freq = row[-1]  # Last element is the count
                    labels = [f"{col.title()}: {val}" for col, val in zip(geo_cols, row[:-1])]
                    top_dropped.append((" | ".join(labels), freq))
            else:
                 top_dropped = [("Geographic outlier (no detailed columns enabled)", flagged_count)]

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
    Generates an interactive HTML file highlighting critical issues with
    a modern dashboard style.
    """
    reports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    html_file = reports_dir / f"diagnostic_report_{timestamp}.html"

    steps_labels = [m["step_name"] for m in metrics_list]
    dropped_data = [m["dropped_count"] for m in metrics_list]
    usable_pct = (usable_count / total_records * 100) if total_records > 0 else 0

    # Determinar color de salud
    health_color = "#2ecc71" if usable_pct >= 80 else "#f1c40f" if usable_pct >= 50 else "#e74c3c"

    # Format the spatial distribution lines dynamically
    dist_html = ""
    for row in spatial_summary.get("distribution", []):
        parts = []
        if "state" in row:
            parts.append(f"<span style='color:#a29bfe'>State:</span> {row['state']}")
        if "bottler" in row:
            parts.append(f"<span style='color:#a29bfe'>Bottler:</span> {row['bottler']}")
        if "municipality" in row:
            parts.append(f"<span style='color:#a29bfe'>Mun:</span> {row['municipality']}")
            
        row_str = " | ".join(parts)
        dist_html += f"<div style='padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 0.95em;'>➔ {row_str} <span style='float:right; font-weight:bold; color: #4cd137;'>{row['count']:,} pts</span></div>"

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Data Health Diagnostic Report</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
        
        body {{ font-family: 'Inter', sans-serif; background-color: #f4f7f6; color: #2c3e50; margin: 0; padding: 20px 0; }}
        .container {{ max-width: 1100px; margin: auto; padding: 0 20px; }}
        
        /* HEADER */
        .header {{ display: flex; justify-content: space-between; align-items: flex-end; margin-bottom: 25px; }}
        .header h1 {{ margin: 0; font-size: 2.2em; font-weight: 800; color: #1a252f; letter-spacing: -0.5px; }}
        .header p {{ margin: 0; color: #7f8c8d; font-size: 0.9em; }}
        
        /* EXECUTIVE BANNER STYLES */
        .exec-banner {{ background: #1e272e; color: #f5f6fa; padding: 30px; border-radius: 16px; margin-bottom: 35px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); position: relative; overflow: hidden; }}
        .exec-banner::before {{ content: ''; position: absolute; top: 0; left: 0; width: 6px; height: 100%; background: {health_color}; }}
        
        .score-container {{ margin-bottom: 25px; }}
        .score-title {{ font-size: 0.9em; text-transform: uppercase; letter-spacing: 1px; color: #a4b0be; margin-bottom: 8px; display: block; }}
        .progress-bg {{ background: rgba(255,255,255,0.1); width: 100%; height: 12px; border-radius: 10px; overflow: hidden; }}
        .progress-bar {{ background: {health_color}; height: 100%; width: {usable_pct}%; transition: width 1s ease-in-out; }}
        
        .grid-metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 25px; }}
        .metric-card {{ background: rgba(255,255,255,0.05); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.05); }}
        .metric-value {{ font-size: 2em; font-weight: 800; margin: 5px 0; }}
        .metric-label {{ font-size: 0.85em; color: #ced6e0; text-transform: uppercase; letter-spacing: 0.5px; }}
        
        .spatial-dist {{ background: rgba(0,0,0,0.2); padding: 15px 20px; border-radius: 10px; max-height: 200px; overflow-y: auto; font-family: 'Courier New', Courier, monospace; }}
        
        /* SCROLLBAR */
        ::-webkit-scrollbar {{ width: 8px; }}
        ::-webkit-scrollbar-track {{ background: rgba(0,0,0,0.05); border-radius: 4px; }}
        ::-webkit-scrollbar-thumb {{ background: rgba(0,0,0,0.2); border-radius: 4px; }}
        
        /* CHARTS & CARDS */
        .content-section {{ background: white; padding: 30px; border-radius: 16px; box-shadow: 0 4px 15px rgba(0,0,0,0.03); margin-bottom: 30px; }}
        .section-title {{ font-size: 1.4em; font-weight: 800; margin-top: 0; margin-bottom: 20px; border-bottom: 2px solid #f1f2f6; padding-bottom: 10px; }}
        
        .chart-container {{ position: relative; height: 350px; width: 100%; margin-bottom: 30px; }}
        
        .issues-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 20px; }}
        .step-card {{ background: #ffffff; border: 1px solid #e1e8ed; padding: 20px; border-radius: 12px; transition: transform 0.2s, box-shadow 0.2s; }}
        .step-card:hover {{ transform: translateY(-3px); box-shadow: 0 8px 20px rgba(0,0,0,0.06); }}
        .step-card.geo-highlight {{ border-top: 4px solid #e74c3c; background: #fffcfc; }}
        .step-card:not(.geo-highlight) {{ border-top: 4px solid #3498db; }}
        
        .step-title {{ font-weight: 800; color: #2c3e50; font-size: 1.1em; margin-bottom: 10px; display: flex; justify-content: space-between; align-items: center; }}
        .step-stats {{ font-size: 0.9em; color: #7f8c8d; margin-bottom: 15px; display: flex; gap: 15px; }}
        .step-stats span {{ display: flex; align-items: center; }}
        .step-stats strong {{ color: #2c3e50; margin-left: 5px; }}
        
        ul.detail-list {{ margin: 0; padding: 0; list-style: none; }}
        ul.detail-list li {{ font-size: 0.9em; color: #34495e; margin-bottom: 8px; display: flex; align-items: center; background: #f8f9fa; padding: 6px 10px; border-radius: 6px; }}
        .count-badge {{ background-color: #e74c3c; color: white; padding: 2px 8px; border-radius: 20px; font-weight: 600; font-size: 0.85em; margin-right: 12px; min-width: 25px; text-align: center; }}
        .highlight-tag {{ background-color: #f39c12; color: white; font-size: 0.7em; padding: 3px 8px; border-radius: 20px; font-weight: 600; text-transform: uppercase; }}
        .empty-state {{ color: #bdc3c7; font-size: 0.9em; font-style: italic; }}
    </style>
</head>
<body>
    <div class="container">
        
        <div class="header">
            <div>
                <h1>Data Health Diagnostic</h1>
                <p>Pipeline Execution Report</p>
            </div>
            <p>Generated: <strong>{time.strftime("%Y-%m-%d %H:%M:%S")}</strong></p>
        </div>
        
        <div class="exec-banner">
            <div class="score-container">
                <span class="score-title">Data Usability Score: {usable_pct:.1f}%</span>
                <div class="progress-bg">
                    <div class="progress-bar"></div>
                </div>
            </div>
            
            <div class="grid-metrics">
                <div class="metric-card">
                    <div class="metric-label">Total Processed</div>
                    <div class="metric-value">{total_records:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label" style="color: {health_color};">Usable Records</div>
                    <div class="metric-value" style="color: {health_color};">{usable_count:,}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label" style="color: #e84118;">Spatial Outliers</div>
                    <div class="metric-value" style="color: #e84118;">{spatial_summary.get("outliers", 0):,}</div>
                </div>
            </div>
            
            <h4 style="color: #dcdde1; margin: 0 0 10px 0; font-size: 0.95em; text-transform: uppercase; letter-spacing: 1px;">Geographic Distribution (Valid Points)</h4>
            <div class="spatial-dist">
                {dist_html if dist_html else "<div style='color: #7f8fa6;'>No spatial distribution data available.</div>"}
            </div>
        </div>

        <div class="content-section">
            <h2 class="section-title">Impact Overview</h2>
            <div class="chart-container">
                <canvas id="reportChart"></canvas>
            </div>
        </div>

        <div class="content-section">
            <h2 class="section-title">Detailed Diagnostic Flags</h2>
            <div class="issues-grid">
"""

    for m in metrics_list:
        card_class = "step-card geo-highlight" if m["is_geo"] else "step-card"
        tag = (
            "<span class='highlight-tag'>Spatial Flag</span>"
            if m["is_geo"]
            else ""
        )

        html_content += f"""
                <div class="{card_class}">
                    <div class="step-title">{m["step_name"]} {tag}</div>
                    
                    <div class="step-stats">
                        <span>Flagged: <strong style="color:#e74c3c;">{m["dropped_count"]:,}</strong></span>
                        <span>Clean: <strong>{m["remaining_count"]:,}</strong></span>
                    </div>
"""
        if m["top_dropped"]:
            html_content += "<ul class='detail-list'>"
            for name, freq in m["top_dropped"]:
                # Truncate very long names for UI cleanliness
                display_name = name if len(str(name)) < 45 else str(name)[:42] + "..."
                html_content += f"<li><span class='count-badge'>{freq}</span> <code>{display_name}</code></li>"
            html_content += "</ul>"
        else:
            html_content += "<div class='empty-state'>No issues detected for this rule. ✨</div>"

        html_content += "</div>"

    html_content += f"""
            </div>
        </div>
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
                    backgroundColor: 'rgba(52, 152, 219, 0.8)',
                    hoverBackgroundColor: 'rgba(41, 128, 185, 1)',
                    borderRadius: 6,
                    borderSkipped: false
                }}]
            }},
            options: {{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {{
                    legend: {{ display: false }}
                }},
                scales: {{
                    y: {{ 
                        beginAtZero: true,
                        grid: {{ color: 'rgba(0,0,0,0.04)', drawBorder: false }}
                    }},
                    x: {{
                        grid: {{ display: false, drawBorder: false }}
                    }}
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

    # 2. Extract Spatial Evaluation Summary for the HTML Banner dynamically
    spatial_summary = {
        "total": total_records,
        "outliers": 0,
        "retained": total_records,
        "distribution": [],
    }
    
    if "geo_outlier" in df.columns:
        outliers_count = df.select(pl.col("geo_outlier").sum()).item()
        retained_df = df.filter(~pl.col("geo_outlier"))
        retained_count = len(retained_df)

        # Build dynamic group list based on existing columns
        group_cols = []
        if "state" in df.columns:
            group_cols.append("state")
        if "bottler" in df.columns:
            group_cols.append("bottler")
        if "municipality" in df.columns:
            group_cols.append("municipality")

        distribution = []
        if group_cols:
            distribution = (
                retained_df.group_by(group_cols)
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