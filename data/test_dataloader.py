from torch.utils.data import DataLoader
from torchvision import transforms

from data.mvtec import MVTecNormalDataset


def main():
    print("=== MVTec DataLoader Test ===")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = MVTecNormalDataset(
        root="data/raw/mvtec",
        category="bottle",
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True
    )

    images = next(iter(loader))

    print("Dataset size:", len(dataset))
    print("Batch shape:", images.shape)
    print("Batch minimum:", images.min().item())
    print("Batch maximum:", images.max().item())
    print("DataLoader test: SUCCESS")


if __name__ == "__main__":
    main()