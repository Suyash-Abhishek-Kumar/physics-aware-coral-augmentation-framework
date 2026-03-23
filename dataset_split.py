import os
import random
import shutil

def split_dataset(input_dir, output_dir, split=0.8):
    classes = os.listdir(input_dir)

    for cls in classes:
        if cls == "metadata.json": continue
        images = os.listdir(os.path.join(input_dir, cls))
        random.shuffle(images)

        split_idx = int(len(images) * split)

        train_imgs = images[:split_idx]
        test_imgs = images[split_idx:]

        for img in train_imgs:
            src = os.path.join(input_dir, cls, img)
            dst = os.path.join(output_dir, "train", cls, img)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

        for img in test_imgs:
            src = os.path.join(input_dir, cls, img)
            dst = os.path.join(output_dir, "test", cls, img)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy(src, dst)

segment = ["deep", "lowlight", "robot", "turbid"]
for i in segment:
    split_dataset("dataset\\original_{}".format(i), "dataset\\split_original_{}".format(i))