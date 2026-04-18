import pandas as pd
from pathlib import Path

# =========================================================
# CONFIG
# =========================================================
DATASET_ROOT = Path("/homes/j244s673/documents/wsu/phd/Tornado-Detection-with-Explainability-Analysis/dataset_updated")
ANNOT_OUT = DATASET_ROOT / "annotations_clean"

CLASS_INFO = {
    "tornado": 1,
    "nontornado": 0,
}

SPLITS = ["train", "augment_train", "val", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

ANNOT_OUT.mkdir(parents=True, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def collect_split_rows(split_name: str):
    rows = []
    for class_name, label in CLASS_INFO.items():
        folder = DATASET_ROOT / class_name / split_name

        if not folder.exists():
            print(f"[WARNING] Missing folder: {folder}")
            continue

        image_files = sorted([p for p in folder.iterdir() if is_image_file(p)])

        print(f"{split_name:>12} | {class_name:10s} | {len(image_files):5d} images | {folder}")

        for img_path in image_files:
            rows.append({
                "filepath": str(img_path.resolve()),
                "binary_label": label,
                "class_name": class_name,
                "split": "train" if split_name == "augment_train" else split_name,
                "source_folder": split_name,
                "basename": img_path.name,
            })

    return rows


def audit_df(df: pd.DataFrame, name: str):
    print("\n" + "=" * 80)
    print(f"AUDIT: {name}")
    print("=" * 80)
    print("Rows:", len(df))

    if df.empty:
        print("[WARNING] DataFrame is empty")
        return

    print("\nLabel counts:")
    print(df["binary_label"].value_counts().sort_index())

    print("\nClass counts:")
    print(df["class_name"].value_counts())

    print("\nPath prefixes:")
    print(df["filepath"].str.extract(r"(^/homes/[^/]+)")[0].value_counts())

    dup_fp = df[df.duplicated("filepath", keep=False)].sort_values("filepath")
    print("\nDuplicate full filepaths:", len(dup_fp))
    if len(dup_fp) > 0:
        dup_fp.to_csv(ANNOT_OUT / f"{name}_duplicate_filepaths.csv", index=False)
        print(f"Saved duplicate filepath report -> {ANNOT_OUT / f'{name}_duplicate_filepaths.csv'}")

    conflict_fp = (
        df.groupby("filepath")["binary_label"]
          .nunique()
          .reset_index()
    )
    conflict_fp = conflict_fp[conflict_fp["binary_label"] > 1]
    print("Conflicting full filepaths:", len(conflict_fp))
    if len(conflict_fp) > 0:
        bad = df[df["filepath"].isin(conflict_fp["filepath"])]
        bad.to_csv(ANNOT_OUT / f"{name}_conflicting_filepaths.csv", index=False)
        print(f"Saved conflicting filepath report -> {ANNOT_OUT / f'{name}_conflicting_filepaths.csv'}")

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
        print(f"Saved conflicting basename report -> {ANNOT_OUT / f'{name}_conflicting_basenames.csv'}")

    missing_files = df[~df["filepath"].map(lambda x: Path(x).exists())]
    print("Missing files on disk:", len(missing_files))
    if len(missing_files) > 0:
        missing_files.to_csv(ANNOT_OUT / f"{name}_missing_files.csv", index=False)
        print(f"Saved missing file report -> {ANNOT_OUT / f'{name}_missing_files.csv'}")


# =========================================================
# BUILD CLEAN ANNOTATIONS
# =========================================================
def main():
    all_rows = []
    for split in SPLITS:
        all_rows.extend(collect_split_rows(split))

    all_df = pd.DataFrame(all_rows)

    if all_df.empty:
        raise RuntimeError("No images found. Check DATASET_ROOT and folder structure.")

    # Keep train.csv = train + augment_train
    train_df = all_df[all_df["source_folder"].isin(["train", "augment_train"])].copy()
    val_df   = all_df[all_df["source_folder"] == "val"].copy()
    test_df  = all_df[all_df["source_folder"] == "test"].copy()

    # Shuffle for reproducibility
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
    val_df   = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
    test_df  = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Audit each split
    audit_df(train_df, "train")
    audit_df(val_df, "val")
    audit_df(test_df, "test")

    # Save CSVs
    train_csv = ANNOT_OUT / "train.csv"
    val_csv   = ANNOT_OUT / "val.csv"
    test_csv  = ANNOT_OUT / "test.csv"

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