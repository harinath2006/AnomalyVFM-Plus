import json
from pathlib import Path
from datetime import datetime


def save_evaluation_result(
    output_path,
    model_name,
    dataset_name,
    num_images,
    image_auroc,
    pixel_auroc
):
    """
    Save evaluation results as a JSON file.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "model": model_name,
        "dataset": dataset_name,
        "num_images": num_images,
        "image_auroc": float(image_auroc),
        "pixel_auroc": float(pixel_auroc),
        "timestamp": datetime.now().isoformat()
    }

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            results,
            file,
            indent=4
        )

    print(
        "Evaluation results saved:",
        output_path
    )


if __name__ == "__main__":

    print(
        "=== Result Logger Test ==="
    )

    save_evaluation_result(
        "results/evaluation/test_result.json",
        "AnomalyVFM+",
        "MVTec Bottle",
        5,
        0.90,
        0.95
    )

    print(
        "Result logger test: SUCCESS"
    )