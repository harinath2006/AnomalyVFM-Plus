import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from configs.loader import load_config
from data.mvtec import MVTecNormalDataset
from models.foundation import create_foundation_model
from models.decoder import AnomalyDecoder
from models.adapters.inject_lora import inject_lora
from synthetic.masks import create_rectangle_mask
from synthetic.generate import apply_synthetic_anomaly
from training.losses import anomaly_loss
from training.checkpoint import save_checkpoint


def train_lora(max_steps=None):

    print("=== AnomalyVFM+ LoRA Training ===")

    # --------------------------------------------------
    # 1. Configuration
    # --------------------------------------------------

    base_config = load_config(
        "configs/baseline.yaml"
    )

    experiment_config = load_config(
        "configs/experiments.yaml"
    )

    experiment = experiment_config["experiments"]["lora"]

    if max_steps is None:
        max_steps = experiment["max_steps"]

    batch_size = experiment["batch_size"]
    learning_rate = experiment["learning_rate"]
    seed = experiment_config["training"]["seed"]

    torch.manual_seed(seed)

    print("Max steps:", max_steps)
    print("Batch size:", batch_size)
    print("Learning rate:", learning_rate)
    print("Seed:", seed)

    # --------------------------------------------------
    # 2. Dataset
    # --------------------------------------------------

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = MVTecNormalDataset(
        root=experiment_config["dataset"]["root"],
        category=experiment_config["dataset"]["category"],
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    print("Dataset size:", len(dataset))

    # --------------------------------------------------
    # 3. Foundation model
    # --------------------------------------------------

    foundation = create_foundation_model(
        base_config["model"]["name"]
    )

    print("Loading DINOv2...")
    foundation.load()

    # --------------------------------------------------
    # 4. Inject LoRA
    # --------------------------------------------------

    print("Injecting LoRA...")

    lora_count = inject_lora(
        foundation.model,
        target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ),
        rank=4,
        alpha=8
    )

    print("LoRA layers:", lora_count)

    # --------------------------------------------------
    # 5. Decoder
    # --------------------------------------------------

    decoder_config = base_config["decoder"]

    decoder = AnomalyDecoder(
        in_channels=decoder_config["input_channels"],
        hidden_channels=tuple(
            decoder_config["hidden_channels"]
        )
    )

    decoder.train()

    # --------------------------------------------------
    # 6. Trainable parameters
    # --------------------------------------------------

    trainable_lora = [
        parameter
        for parameter in foundation.model.parameters()
        if parameter.requires_grad
    ]

    trainable_parameters = (
        trainable_lora
        + list(decoder.parameters())
    )

    trainable_count = sum(
        parameter.numel()
        for parameter in trainable_parameters
    )

    print(
        "Trainable parameters:",
        trainable_count
    )

    # --------------------------------------------------
    # 7. Optimizer
    # --------------------------------------------------

    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=learning_rate
    )

    # --------------------------------------------------
    # 8. Training
    # --------------------------------------------------

    foundation.model.train()
    decoder.train()

    for step, images in enumerate(loader):

        if step >= max_steps:
            break

        print()
        print(
            f"----- Step {step + 1}/{max_steps} -----"
        )

        batch_size_current = images.shape[0]

        # --------------------------------------------------
        # Synthetic anomaly
        # --------------------------------------------------

        masks = create_rectangle_mask(
            batch_size=batch_size_current,
            height=224,
            width=224,
            top=80,
            left=60,
            rectangle_height=40,
            rectangle_width=70
        )

        anomalous_images = apply_synthetic_anomaly(
            images,
            masks
        )

        # --------------------------------------------------
        # Convert to PIL
        # --------------------------------------------------

        to_pil = transforms.ToPILImage()

        pil_images = [
            to_pil(image)
            for image in anomalous_images
        ]

        # --------------------------------------------------
        # DINOv2 + LoRA
        # --------------------------------------------------

        features = foundation.extract_features(
            pil_images,
            trainable=True
        )

        print(
            "Feature shape:",
            features.shape
        )

        # --------------------------------------------------
        # Decoder
        # --------------------------------------------------

        logits = decoder(features)

        logits = torch.nn.functional.interpolate(
            logits,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        # --------------------------------------------------
        # Loss
        # --------------------------------------------------

        loss = anomaly_loss(
            logits,
            masks
        )

        print(
            "Loss:",
            loss.item()
        )

        # --------------------------------------------------
        # Backpropagation
        # --------------------------------------------------

        optimizer.zero_grad()

        loss.backward()

        # --------------------------------------------------
        # Verify LoRA gradient
        # --------------------------------------------------

        lora_gradient_found = False

        for name, parameter in foundation.model.named_parameters():

            if (
                "lora_B" in name
                and parameter.grad is not None
            ):

                print(
                    "LoRA gradient:",
                    name,
                    "->",
                    parameter.grad.abs().mean().item()
                )

                lora_gradient_found = True
                break

        print(
            "LoRA gradient received:",
            lora_gradient_found
        )

        # --------------------------------------------------
        # Optimizer
        # --------------------------------------------------

        optimizer.step()

        # --------------------------------------------------
        # Checkpoint
        # --------------------------------------------------

        save_checkpoint(
            path="experiments/lora/checkpoint.pt",
            foundation_model=foundation.model,
            decoder=decoder,
            optimizer=optimizer,
            step=step + 1,
            config={
                "experiment": "lora",
                "base_config": base_config,
                "experiment_config": experiment_config,
                "lora_rank": 4,
                "lora_alpha": 8
            }
        )

    print()
    print("=== LoRA training: SUCCESS ===")


if __name__ == "__main__":
    train_lora()