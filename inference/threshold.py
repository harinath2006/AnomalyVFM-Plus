import json
from pathlib import Path


THRESHOLD_PATH = Path(
    "results/evaluation/threshold.json"
)


def load_threshold():

    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Threshold file not found: {THRESHOLD_PATH}"
        )

    with open(
        THRESHOLD_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return float(
        data["threshold"]
    )