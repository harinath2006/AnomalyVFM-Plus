from pathlib import Path
import json
import sys
from collections import Counter

import torch
from PIL import Image
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    confusion_matrix,
)

# Ensure imports work when running this script directly.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.brainmri import BrainMRITestDataset
from models.anomaly_vfm_plus import AnomalyVFMPlus


# ============================================================
# CONFIGURATION
# ============================================================

CHECKPOINT_PATH = (
    PROJECT_ROOT / "experiments" / "plus" / "checkpoint_medical.pt"
)

DATASET_PATH = (
    PROJECT_ROOT / "data" / "raw" / "medical" / "brain_mri"
)

RESULTS_DIR = PROJECT_ROOT / "results" / "evaluation"

RESULTS_PATH = RESULTS_DIR / "brainmri_results.json"

DEVICE = torch.device("cpu")

MAX_IMAGES = 200

# Descriptive threshold only. This is NOT medically validated.
CLASSIFICATION_THRESHOLD = 0.5


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():
    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    print("\nLoading AnomalyVFM+ model...")

    model = AnomalyVFMPlus()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Checkpoint does not contain 'model_state_dict'."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    print("Checkpoint loaded successfully.")

    return model


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset(dataset):
    print("\n========== DATASET VALIDATION ==========")

    if len(dataset) == 0:
        raise ValueError(
            "Brain MRI dataset is empty. Check the dataset path."
        )

    labels = []
    categories = []

    for index in range(len(dataset)):
        sample = dataset[index]

        label = int(sample["label"])
        category = str(sample["category"])

        if label not in (0, 1):
            raise ValueError(
                f"Unexpected label {label} at dataset index {index}."
            )

        labels.append(label)
        categories.append(category)

    label_counts = Counter(labels)
    category_counts = Counter(categories)

    print("Dataset path:", DATASET_PATH)
    print("Total images:", len(dataset))

    print(
        "Class distribution:",
        {
            "normal (0)": label_counts.get(0, 0),
            "abnormal (1)": label_counts.get(1, 0),
        },
    )

    print(
        "Category distribution:",
        dict(category_counts),
    )

    if len(label_counts) < 2:
        print(
            "\nWARNING: Dataset contains only one class. "
            "Image-level AUROC cannot be calculated."
        )

    if not category_counts:
        raise ValueError("No categories were found in the dataset.")

    print("Dataset validation completed.")

    return label_counts, category_counts


# ============================================================
# EVALUATION
# ============================================================

def evaluate():

    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(
            f"Brain MRI dataset not found: {DATASET_PATH}"
        )

    print("Loading Brain MRI dataset...")

    dataset = BrainMRITestDataset(
        path=DATASET_PATH,
        max_images=MAX_IMAGES,
        image_size=224,
    )

    if len(dataset) == 0:
        raise ValueError("Brain MRI dataset contains no images.")

    # Validate labels and category distribution before inference.
    label_counts, category_counts = validate_dataset(dataset)

    model = load_model()

    labels = []
    scores = []
    category_records = []

    print("\n========== RUNNING BRAIN MRI EVALUATION ==========")
    print("Device:", DEVICE)
    print("Images:", len(dataset))

    with torch.inference_mode():

        for index in range(len(dataset)):

            sample = dataset[index]

            image_path = DATASET_PATH / sample["path"]

            # The model expects PIL images.
            # Its own image processor performs preprocessing.
            with Image.open(image_path) as source:
                pil_image = source.convert("RGB")

            _, image_logit = model.forward_with_score(
                [pil_image]
            )

            score = torch.sigmoid(
                image_logit
            ).reshape(-1)[0].item()

            label = int(sample["label"])
            category = str(sample["category"])

            labels.append(label)
            scores.append(float(score))

            category_records.append({
                "path": str(sample["path"]),
                "category": category,
                "label": label,
                "score": float(score),
            })

            if (
                (index + 1) % 20 == 0
                or index + 1 == len(dataset)
            ):
                print(
                    f"Processed {index + 1}/{len(dataset)} images"
                )

    # ========================================================
    # IMAGE-LEVEL AUROC
    # ========================================================

    image_auroc = None

    if len(set(labels)) >= 2:
        image_auroc = float(
            roc_auc_score(labels, scores)
        )
    else:
        print(
            "\nImage-level AUROC unavailable: "
            "the evaluated dataset contains only one image class."
        )

    # ========================================================
    # BINARY CLASSIFICATION METRICS
    # ========================================================

    predictions = [
        int(score >= CLASSIFICATION_THRESHOLD)
        for score in scores
    ]

    accuracy = float(
        accuracy_score(labels, predictions)
    )

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    # ========================================================
    # CATEGORY-WISE ONE-VS-REST AUROC
    # ========================================================

    category_auroc = {}

    for category in sorted(set(
        record["category"]
        for record in category_records
    )):

        binary_labels = [
            1 if record["category"] == category else 0
            for record in category_records
        ]

        category_scores = [
            record["score"]
            for record in category_records
        ]

        if len(set(binary_labels)) >= 2:
            category_auroc[category] = float(
                roc_auc_score(
                    binary_labels,
                    category_scores,
                )
            )
        else:
            category_auroc[category] = None

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {
        "dataset": "Brain MRI",

        "checkpoint": str(
            CHECKPOINT_PATH.relative_to(PROJECT_ROOT)
        ),

        "evaluation_type": (
            "cross-domain image-level anomaly evaluation"
        ),

        "device": str(DEVICE),

        "num_images": len(labels),

        "normal_count": int(
            sum(label == 0 for label in labels)
        ),

        "abnormal_count": int(
            sum(label == 1 for label in labels)
        ),

        "class_distribution": {
            "normal": int(label_counts.get(0, 0)),
            "abnormal": int(label_counts.get(1, 0)),
        },

        "category_distribution": {
            category: int(count)
            for category, count in category_counts.items()
        },

        "image_auroc": image_auroc,

        "classification_threshold": CLASSIFICATION_THRESHOLD,

        "accuracy_at_threshold": accuracy,

        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },

        "one_vs_rest_category_auroc": category_auroc,

        "warning": (
            "The checkpoint was trained on MVTec Bottle. "
            "These results measure cross-domain generalization, "
            "not medical diagnostic performance. "
            "The classification threshold is not clinically validated. "
            "Category-wise AUROC measures each category against all "
            "other evaluated categories; it does not establish "
            "tumor-type diagnostic capability."
        ),

        "samples": category_records,
    }

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    # ========================================================
    # PRINT FINAL SUMMARY
    # ========================================================

    print("\n========== BRAIN MRI RESULTS ==========")

    print(f"Images evaluated : {len(labels)}")
    print(f"Normal images    : {sum(label == 0 for label in labels)}")
    print(f"Abnormal images  : {sum(label == 1 for label in labels)}")
    print(f"Image AUROC      : {image_auroc}")
    print(f"Accuracy @ 0.5   : {accuracy:.4f}")

    print(
        f"Confusion matrix : "
        f"TN={tn}, FP={fp}, FN={fn}, TP={tp}"
    )

    print("\nOne-vs-rest category AUROC:")

    for category, value in category_auroc.items():
        if value is None:
            print(f"  {category}: unavailable")
        else:
            print(f"  {category}: {value:.4f}")

    print("\nResults saved to:")
    print(RESULTS_PATH)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    evaluate()