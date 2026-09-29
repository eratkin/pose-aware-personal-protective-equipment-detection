"""
Main entry point for the Pose-Aware PPE Compliance Detection project.

Run with:

    python main.py
"""

import sys
from pathlib import Path


# ------------------------------------------------------------
# PROJECT PATHS
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
INPUT_DIR = PROJECT_ROOT / "input"
CONFIG_FILE = PROJECT_ROOT / "configs" / "config.yaml"


# ------------------------------------------------------------
# MAKE src IMPORTABLE
# ------------------------------------------------------------

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from infer import run_pipeline


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def main():

    print()
    print("=" * 70)
    print("POSE-AWARE PPE COMPLIANCE DETECTION")
    print("=" * 70)
    print()

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Input folder: {INPUT_DIR}")
    print(f"Config file:  {CONFIG_FILE}")
    print()

    # Create input folder if necessary.
    if not INPUT_DIR.exists():

        INPUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print("The input folder was created:")
        print(f"  {INPUT_DIR}")
        print()
        print("Add images to it and run:")
        print("  python main.py")

        return

    # Make sure config exists.
    if not CONFIG_FILE.exists():

        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_FILE}"
        )

    # Run the complete pipeline.
    run_pipeline(
        input_path=INPUT_DIR,
        config_path=CONFIG_FILE,
    )


if __name__ == "__main__":
    main()