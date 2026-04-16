import os
import cv2
from pathlib import Path
from tqdm import tqdm
import albumentations as A


# ----------------------------
# Configuration
# ----------------------------
INPUT_DIR = os.environ.get(
    "INPUT_DIR",
    "/homes/j244s673/documents/wsu/phd/tornado_data/tornado"
)
OUTPUT_DIR = os.environ.get(
    "OUTPUT_DIR",
    "/homes/j244s673/documents/wsu/phd/tornado_data/augmented_tornadoes"
)
AUGS_PER_IMAGE = 100
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ----------------------------
# Augmentation pipeline
# ----------------------------
transform = A.Compose([
    A.RandomResizedCrop(
        size=(256, 256),
        scale=(0.8, 1.0),
        ratio=(0.9, 1.1),
        p=0.5
    ),
    A.HorizontalFlip(p=0.5),
    A.Rotate(limit=10, border_mode=cv2.BORDER_REFLECT_101, p=0.4),
    A.RandomBrightnessContrast(
        brightness_limit=0.2,
        contrast_limit=0.2,
        p=0.5
    ),
    A.HueSaturationValue(
        hue_shift_limit=10,
        sat_shift_limit=15,
        val_shift_limit=10,
        p=0.3
    ),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MotionBlur(blur_limit=5, p=1.0),
    ], p=0.3),
    A.GaussNoise(p=0.3),
    A.OneOf([
        A.RandomFog(fog_coef_range=(0.1, 0.3), alpha_coef=0.08, p=1.0),
        A.RandomRain(
            slant_range=(-10, 10),
            drop_length=10,
            drop_width=1,
            drop_color=(200, 200, 200),
            blur_value=3,
            brightness_coefficient=0.9,
            p=1.0
        ),
    ], p=0.2),
    A.CoarseDropout(
        num_holes_range=(1, 5),
        hole_height_range=(10, 30),
        hole_width_range=(10, 30),
        fill=0,
        p=0.2
    ),
    A.Resize(256, 256)
])


# ----------------------------
# Utility functions
# ----------------------------
def is_image_file(file_path: Path) -> bool:
    return file_path.suffix.lower() in IMAGE_EXTENSIONS


def load_image(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def save_image(image, save_path: Path):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(save_path), image_bgr)


# ----------------------------
# Main augmentation loop
# ----------------------------
def augment_images(input_dir, output_dir, augs_per_image):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input directory : {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Augs per image  : {augs_per_image}")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    image_files = sorted(
        [p for p in input_dir.iterdir() if p.is_file() and is_image_file(p)]
    )

    if not image_files:
        print("No images found in input directory.")
        return

    print(f"Found {len(image_files)} input images.")

    for image_path in tqdm(image_files, desc="Augmenting images"):
        try:
            image = load_image(image_path)

            # Save resized original once
            original_save_path = output_dir / image_path.name
            if not original_save_path.exists():
                resized_original = cv2.resize(image, (256, 256))
                save_image(resized_original, original_save_path)

            # Create augmented versions
            for i in range(augs_per_image):
                augmented = transform(image=image)["image"]
                save_name = f"{image_path.stem}_aug_{i+1}{image_path.suffix}"
                save_path = output_dir / save_name
                save_image(augmented, save_path)

        except Exception as e:
            print(f"Skipping {image_path}: {e}")


if __name__ == "__main__":
    augment_images(INPUT_DIR, OUTPUT_DIR, AUGS_PER_IMAGE)
    print(f"Done. Augmented images saved to: {OUTPUT_DIR}")