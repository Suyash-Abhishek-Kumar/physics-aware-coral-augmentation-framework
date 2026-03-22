import cv2
import numpy as np
import os
import json
import random

# -----------------------------
# LOAD IMAGE
# -----------------------------
def load_image(path, size=(256,256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img


# -----------------------------
# SAVE IMAGE
# -----------------------------
def save_image(img, path):
    img = (img*255).astype(np.uint8)
    cv2.imwrite(path, img)


# -----------------------------
# AUGMENTATIONS
# -----------------------------
def color_attenuation(img, depth=10):
    beta_r = 0.15
    beta_g = 0.07
    beta_b = 0.03

    r = img[:,:,2] * np.exp(-beta_r * depth)
    g = img[:,:,1] * np.exp(-beta_g * depth)
    b = img[:,:,0] * np.exp(-beta_b * depth)

    result = np.stack([b,g,r], axis=2)
    return np.clip(result, 0, 1)


def turbidity(img, strength=0.6):
    haze = np.ones_like(img) * 0.7
    result = img * (1-strength) + haze * strength
    result = cv2.GaussianBlur(result, (9,9), 0)
    return np.clip(result, 0, 1)


def low_light(img, factor=0.4):
    result = img * factor
    noise = np.random.normal(0, 0.03, img.shape)
    result += noise
    return np.clip(result, 0, 1)


def marine_snow(img, density=0.01):
    h,w,_ = img.shape
    num_particles = int(h*w*density)

    for _ in range(num_particles):
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)
        img[y,x] = [1,1,1]

    return img


def motion_blur(img, size=15):
    kernel = np.zeros((size,size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel /= size

    result = cv2.filter2D(img, -1, kernel)
    return result


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def expand_dataset(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    metadata = []

    classes = os.listdir(input_dir)

    for cls in classes:
        input_class = os.path.join(input_dir, cls)
        output_class = os.path.join(output_dir, cls)

        os.makedirs(output_class, exist_ok=True)

        images = [f for f in os.listdir(input_class) if f.lower().endswith(('.jpg','.png','.jpeg'))]

        for img_name in images:
            img_path = os.path.join(input_class, img_name)
            img = load_image(img_path)

            base_name = os.path.splitext(img_name)[0]

            # ---------- ORIGINAL ----------
            save_name = f"{base_name}_original.png"
            save_path = os.path.join(output_class, save_name)
            save_image(img, save_path)

            metadata.append({
                "file": save_name,
                "class": cls,
                "type": "original"
            })

            # ---------- 1. DEEP WATER ----------
            depth = random.uniform(10, 25)
            aug = color_attenuation(img, depth)
            aug = turbidity(aug, 0.3)

            save_name = f"{base_name}_deep.png"
            save_image(aug, os.path.join(output_class, save_name))

            metadata.append({
                "file": save_name,
                "class": cls,
                "type": "deep_water",
                "depth": depth
            })

            # ---------- 2. TURBID ----------
            strength = random.uniform(0.5, 0.8)
            aug = turbidity(img, strength)
            aug = marine_snow(aug, 0.01)

            save_name = f"{base_name}_turbid.png"
            save_image(aug, os.path.join(output_class, save_name))

            metadata.append({
                "file": save_name,
                "class": cls,
                "type": "turbid_water",
                "strength": strength
            })

            # ---------- 3. LOW LIGHT ----------
            factor = random.uniform(0.3, 0.6)
            aug = low_light(img, factor)

            save_name = f"{base_name}_lowlight.png"
            save_image(aug, os.path.join(output_class, save_name))

            metadata.append({
                "file": save_name,
                "class": cls,
                "type": "low_light",
                "factor": factor
            })

            # ---------- 4. ROBOT ----------
            blur_size = random.choice([5,9,15])
            aug = motion_blur(img, blur_size)
            aug = low_light(aug, 0.5)

            save_name = f"{base_name}_robot.png"
            save_image(aug, os.path.join(output_class, save_name))

            metadata.append({
                "file": save_name,
                "class": cls,
                "type": "robot_capture",
                "blur": blur_size
            })

    # Save metadata
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    expand_dataset("dataset/original", "dataset/expanded_5x")