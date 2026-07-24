from pathlib import Path

import yaml

# Main project base directory (points to data_health/)
BASE_DIR = Path(__file__).resolve().parent.parent


def load_config():
    """
    Loads the YAML configuration file and resolves paths.

    Returns:
        dict: The complete configuration dictionary.
    """
    # UPDATED: Now looking for config.yaml directly in the root directory
    config_path = BASE_DIR / "config.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Resolve relative paths to absolute paths
    config["paths"]["raw_file"] = BASE_DIR / config["paths"]["raw_file"]
    config["paths"]["reports_dir"] = BASE_DIR / config["paths"]["reports_dir"]
    config["paths"]["geojson_boundaries"] = (
        BASE_DIR / config["paths"]["geojson_boundaries"]
    )

    # Convert bad_hours list to a set for O(1) lookup performance
    config["constants"]["bad_hours"] = set(config["constants"]["bad_hours"])

    return config
