import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT = Path("/homes/j244s673/documents/wsu/phd/Tornado-Detection-with-Explainability-Analysis/dataset_updated")
ANNOT_OUT = DATASET_ROOT / "annotations_clean"

CLASS_INFO = {
    "tornado": 1,
    "non_tornado": 0,
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ANNOT_OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_rows(folder: Path, class_name: str, label: int, split_name: str, source_folder: str):
    rows = []

    if not folder.exists():
        print(f"[WARNING] Missing folder: {folder}")
        return rows

    image_files = sorted([p for p in folder.iterdir() if is_image_file(p)])
    print(f"{split_name:>5} | {class_name:12s} | {len(image_files):5d} images | {folder}")

    for img_path in image_files:
        rows.append({
            "filepath": str(img_path.resolve()),
            "binary_label": label,
            "class_name": class_name,
            "split": split_name,
            "source_folder": source_folder,
            "basename": img_path.name,
        })

    return rows


def audit_df(df: pd.DataFrame, name: str):
    print("\n" + "=" * 80)
    print(f"AUDIT: {name}")
    print("=" * 80)

    print("Rows:", len(df))
    if df.empty:
        print("[WARNING] Empty split")
        return

    print("\nLabel counts:")
    print(df["binary_label"].value_counts().sort_index())

    print("\nClass counts:")
    print(df["class_name"].value_counts())

    dup_fp = df[df.duplicated("filepath", keep=False)].sort_values("filepath")
    print("\nDuplicate full filepaths:", len(dup_fp))
    if len(dup_fp) > 0:
        dup_fp.to_csv(ANNOT_OUT / f"{name}_duplicate_filepaths.csv", index=False)

    conflict_base = (
        df.groupby("basename")["binary_label"]
          .nunique()
          .reset_index()
    )
    conflict_base = conflict_base[conflict_base["binary_label"] > 1]
    print("Conflicting basenames:", len(conflict_base))
    if len(conflict_base) > 0:
        bad = df[df["basename"].isin(conflict_base["basename"])].sort_values(["basename", "class_name", "filepath"])
        bad.to_csv(ANNOT_OUT / f"{name}_conflicting_basenames.csv", index=False)

    missing_files = df[~df["filepath"].map(lambda x: Path(x).exists())]
    print("Missing files on disk:", len(missing_files))
    if len(missing_files) > 0:
        missing_files.to_csv(ANNOT_OUT / f"{name}_missing_files.csv", index=False)


# =========================================================
# MAIN
# =========================================================
def main():
    train_rows = []
    val_rows = []
    test_rows = []

    for class_name, label in CLASS_INFO.items():
        class_dir = DATASET_ROOT / class_name

        # TRAIN comes from augment_train
        train_rows.extend(
            collect_rows(
                folder=class_dir / "augment_train",
                class_name=class_name,
                label=label,
                split_name="train",
                source_folder="augment_train"
            )
        )

        # VAL
        val_rows.extend(
            collect_rows(
                folder=class_dir / "val",
                class_name=class_name,
                label=label,
                split_name="val",
                source_folder="val"
            )
        )

        # TEST
        test_rows.extend(
            collect_rows(
                folder=class_dir / "test",
                class_name=class_name,
                label=label,
                split_name="test",
                source_folder="test"
            )
        )

    train_df = pd.DataFrame(train_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    val_df = pd.DataFrame(val_rows).sample(frac=1, random_state=42).reset_index(drop=True)
    test_df = pd.DataFrame(test_rows).sample(frac=1, random_state=42).reset_index(drop=True)

    audit_df(train_df, "train")
    audit_df(val_df, "val")
    audit_df(test_df, "test")

    train_csv = ANNOT_OUT / "train.csv"
    val_csv = ANNOT_OUT / "val.csv"
    test_csv = ANNOT_OUT / "test.csv"

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    print("\n" + "=" * 80)
    print("CLEAN ANNOTATIONS WRITTEN")
    print("=" * 80)
    print("Train CSV:", train_csv)
    print("Val CSV  :", val_csv)
    print("Test CSV :", test_csv)

    print("\nQuick summary:")
    print("Train:", len(train_df))
    print("Val  :", len(val_df))
    print("Test :", len(test_df))


if __name__ == "__main__":
    main()