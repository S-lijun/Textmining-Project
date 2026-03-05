import os
import requests
import py7zr
import shutil
import json
from pathlib import Path

alfred_dataset = "https://ai2-vision-alfred.s3-us-west-2.amazonaws.com/json_2.1.0.7z"
local_filepath = Path("./data/alfred_dataset.7z")
output_dir = Path("./data/alfred")
os.makedirs(output_dir, exist_ok=True)

# Download the dataset
print("Downloading the ALFRED dataset...")
response = requests.get(alfred_dataset)
with open(local_filepath, "wb") as f:
    f.write(response.content)

# Extract the dataset
print("Extracting the ALFRED dataset...")
with py7zr.SevenZipFile(local_filepath, mode="r") as z:
    z.extractall(path="./data")

# restructure the dataset
print("Restructuring the ALFRED dataset...")
dataset_path = Path("./data/json_2.1.0")
for split in ["train", "valid_seen", "valid_unseen", "tests_seen", "tests_unseen"]:
    split_path = dataset_path / split
    json_files = list(split_path.rglob("*.json"))
    extracted_data = []
    for json_file in json_files:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

            task_type = data.get("task_type", "")
            annotations = data.get("turk_annotations", {}).get("anns", [])
            for ann in annotations:
                task_desc = ann.get("task_desc")
                high_descs = ann.get("high_descs")
                if task_desc:
                    extracted_data.append(
                        {
                            "input": task_desc,
                            "label": task_type,
                            "path": str(json_file),
                            "high_descs": high_descs,
                        }
                    )
    # Save the extracted data for the split
    output_file = output_dir / f"{split}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(extracted_data, f, indent=4, ensure_ascii=False)

shutil.rmtree(dataset_path)  # Clean up the extracted folder

print("Dataset downloaded and extracted successfully.")
