import os
import pandas as pd


# ---------------- COMBINED OUTPUTS ----------------
def generate_csv(data, base_dir):
    path = os.path.join(base_dir, "results.csv")
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def generate_txt(data, base_dir):
    path = os.path.join(base_dir, "results.txt")
    with open(path, "w") as f:
        for row in data:
            f.write(str(row) + "\n")
    return path


# ---------------- SEPARATE OUTPUTS ----------------
def generate_separate_txt(data, base_dir):
    """
    Generates one TXT per image.
    Returns: { image_name : file_path }
    """
    grouped = {}
    for row in data:
        grouped.setdefault(row["image"], []).append(row)

    result = {}
    for img, rows in grouped.items():
        name = os.path.splitext(img)[0]
        file_path = os.path.join(base_dir, f"{name}_results.txt")

        with open(file_path, "w") as f:
            for r in rows:
                f.write(str(r) + "\n")

        result[img] = file_path

    return result


def generate_separate_csv(data, base_dir):
    """
    Generates one CSV per image.
    Returns: { image_name : file_path }
    """
    if not data:
        return {}

    df = pd.DataFrame(data)
    result = {}

    for img in df["image"].unique():
        name = os.path.splitext(img)[0]
        img_df = df[df["image"] == img]

        path = os.path.join(base_dir, f"{name}_results.csv")
        img_df.to_csv(path, index=False)

        result[img] = path

    return result
