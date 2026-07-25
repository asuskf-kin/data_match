from pathlib import Path

import yaml

BASE_DIR = Path(__file__).resolve().parent.parent


def load_config():
    """Loads the YAML configuration file and resolves paths."""
    config_path = BASE_DIR / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths to absolute paths
    config["paths"]["raw_file"] = BASE_DIR / config["paths"]["raw_file"]
    config["paths"]["reports_dir"] = BASE_DIR / config["paths"]["reports_dir"]
    config["paths"]["geojson_boundaries"] = (
        BASE_DIR / config["paths"]["geojson_boundaries"]
    )

    # Convert collections if needed
    if "bad_hours" in config.get("constants", {}):
        config["constants"]["bad_hours"] = set(config["constants"]["bad_hours"])

    # Ensure default values for parameters if not defined in YAML
    if "parameters" not in config:
        config["parameters"] = {}

    if "batch_size" not in config["parameters"]:
        config["parameters"]["batch_size"] = 50000  # Default security value

    if "top_bottlers_limit" not in config["parameters"]:
        config["parameters"]["top_bottlers_limit"] = 5  # Default value for Top limit

    if "analyze_municipality" not in config["parameters"]:
        config["parameters"]["analyze_state"] = False  # Default value for Top limit

    return config
