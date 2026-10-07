from pathlib import Path
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset


class MVTecNormalDataset(Dataset):
    """
    Dataset containing only normal MVTec training images.

    Expected path:
        data/raw/mvtec/<category>/train/good/
    """

    def __init__(self, root, category, transform=None):
        self.root = Path(root)
        self.category = category
        self.transform = transform

        self.image_dir = (
            self.root
            / category
            / "train"
            / "good"
        )

        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.image_dir}"
            )

        self.images = sorted(
            [
                path
                for path in self.image_dir.iterdir()
                if path.suffix.lower()
                in [".png", ".jpg", ".jpeg"]
            ]
        )

        if not self.images:
            raise RuntimeError(
                f"No images found in: {self.image_dir}"
            )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image_path = self.images[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


if __name__ == "__main__":
    print("=== MVTec Dataset Loader Test ===")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = MVTecNormalDataset(
        root="data/raw/mvtec",
        category="bottle",
        transform=transform
    )

    print("Category:", dataset.category)
    print("Number of images:", len(dataset))
    print("First image:", dataset.images[0])
    image = dataset[0]
    print("Image shape:", image.shape)
    print("Image range:",image.min().item(),"to",image.max().item())
    print("Dataset loader test: SUCCESS")