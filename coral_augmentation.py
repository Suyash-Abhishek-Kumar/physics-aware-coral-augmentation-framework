import cv2
import numpy as np
import os
import random

# -----------------------------
# IMAGE LOADER
# -----------------------------
def load_image(path, size=(256,256)):
    img = cv2.imread(path)
    img = cv2.resize(img, size)
    img = img.astype(np.float32) / 255.0
    return img


# -----------------------------
# 1. COLOR ATTENUATION (DEPTH SIMULATION)
# Physics: Red light absorbed fastest underwater
# I = I0 * exp(-beta * depth)
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


# -----------------------------
# 2. TURBIDITY / SCATTERING
# Physics: Light scattering reduces contrast
# I = I * t + A*(1-t)
# -----------------------------
def turbidity(img, strength=0.6):
    haze = np.ones_like(img) * 0.7
    result = img * (1-strength) + haze * strength
    result = cv2.GaussianBlur(result, (9,9), 0)
    return np.clip(result, 0, 1)


# -----------------------------
# 3. LOW LIGHT SIMULATION
# -----------------------------
def low_light(img, factor=0.4):
    result = img * factor
    noise = np.random.normal(0, 0.03, img.shape)
    result += noise
    return np.clip(result, 0, 1)


# -----------------------------
# 4. MARINE SNOW (PARTICLE NOISE)
# -----------------------------
def marine_snow(img, density=0.01):
    h,w,_ = img.shape
    num_particles = int(h*w*density)

    for _ in range(num_particles):
        x = random.randint(0,w-1)
        y = random.randint(0,h-1)
        img[y,x] = [1,1,1]

    return img


# -----------------------------
# 5. MOTION BLUR (UNDERWATER CAMERA)
# -----------------------------
def motion_blur(img, size=15):
    kernel = np.zeros((size,size))
    kernel[int((size-1)/2), :] = np.ones(size)
    kernel /= size

    result = cv2.filter2D(img, -1, kernel)
    return result


# -----------------------------
# DOMAIN GENERATOR (KEY MODULE)
# -----------------------------
def generate_domain(img, domain="deep_water"):
    img = img.copy()

    if domain == "deep_water":
        img = color_attenuation(img, depth=20)
        img = turbidity(img, 0.3)

    elif domain == "turbid_water":
        img = turbidity(img, 0.7)
        img = marine_snow(img, 0.02)

    elif domain == "low_light":
        img = low_light(img, 0.3)

    elif domain == "robot_capture":
        img = motion_blur(img)
        img = low_light(img, 0.5)

    return np.clip(img, 0, 1)


# -----------------------------
# SAVE IMAGE
# -----------------------------
def save_image(img, path):
    img = (img*255).astype(np.uint8)
    cv2.imwrite(path, img)


# -----------------------------
# BATCH AUGMENTATION PIPELINE
# -----------------------------
def augment_dataset(input_dir, output_dir):
    domains = ["deep_water","turbid_water","low_light","robot_capture"]

    os.makedirs(output_dir, exist_ok=True)

    for file in os.listdir(input_dir):
        img_path = os.path.join(input_dir, file)
        img = load_image(img_path)

        for d in domains:
            aug = generate_domain(img, d)
            save_path = os.path.join(output_dir, f"{file}_{d}.png")
            save_image(aug, save_path)


if __name__ == "__main__":
    augment_dataset("dataset/original/test", "dataset/augmented")