import os
import shutil

def create_filtered_dataset(input_dir, output_dir, allowed_types):
    """
    allowed_types = ["original", "turbid"] etc.
    """

    classes = os.listdir(input_dir)

    for cls in classes:
        if cls == "metadata.json": continue
        input_class = os.path.join(input_dir, cls)
        output_class = os.path.join(output_dir, cls)

        os.makedirs(output_class, exist_ok=True)

        for file in os.listdir(input_class):
            if not file.lower().endswith(('.png','.jpg','.jpeg')):
                continue

            # Check if file contains any allowed type
            if any(t in file for t in allowed_types):
                src = os.path.join(input_class, file)
                dst = os.path.join(output_class, file)
                shutil.copy(src, dst)

    print(f"Dataset created at: {output_dir}")

create_filtered_dataset(
    "dataset/expanded_5x",
    "dataset/original_turbid",
    ["original", "turbid"]
)
create_filtered_dataset(
    "dataset/expanded_5x",
    "dataset/original_deep",
    ["original", "deep"]
)
create_filtered_dataset(
    "dataset/expanded_5x",
    "dataset/original_lowlight",
    ["original", "lowlight"]
)
create_filtered_dataset(
    "dataset/expanded_5x",
    "dataset/original_robot",
    ["original", "robot"]
)