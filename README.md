# Physics-Aware Coral Augmentation Framework

## Overview

The **Physics-Aware Coral Augmentation Framework** is a modular pipeline that generates realistic underwater variations of coral reef images. It simulates environmental conditions such as depth-based color attenuation, turbidity, low-light visibility, and underwater noise to create synthetic datasets for robust coral health monitoring and cross-domain learning.

The framework addresses limitations in existing coral reef datasets, including dataset locality, limited variability, and poor generalization across underwater environments.

---

## Motivation

Current coral monitoring models suffer from:

- Dataset bias toward specific geographic regions  
- Limited underwater condition diversity  
- Poor performance under turbidity and illumination changes  
- Small dataset sizes  

This framework generates synthetic underwater domains using physics-inspired models to improve robustness and enable cross-domain coral monitoring.

---

## Key Features

- Physics-inspired underwater image simulation  
- Depth-based wavelength attenuation modeling  
- Turbidity and light scattering simulation  
- Low-light and sensor noise generation  
- Marine particle noise (marine snow)  
- Motion blur simulation  
- Multi-domain dataset generation  
- Modular and extensible architecture  

---

## Pipeline

```

Raw Coral Images → Preprocessing → Underwater Simulation → Augmented Dataset

```

---

## Underwater Physics Models

### Light Attenuation Model

Light intensity decreases exponentially with depth:

```

I = I0 * exp(-βd)

```

Where:

- `I0` = original intensity  
- `d` = depth  
- `β` = wavelength attenuation coefficient  

Red wavelengths attenuate faster than blue wavelengths.

---

### Scattering / Turbidity Model

```

I(x) = J(x)t(x) + A(1 - t(x))

```

Where:

- `J(x)` = original image  
- `t(x)` = transmission  
- `A` = background light  

This produces haze and reduced contrast.

---

## Project Structure

```

project/
coral_augmentation.py
dataset/
original/
augmented/

````

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
````

---

## Usage

Place coral images inside:

```
dataset/original/
```

Run the augmentation pipeline:

```bash
python coral_augmentation.py
```

Generated images will be stored in:

```
dataset/augmented/
```

---

## Generated Domains

The framework simulates multiple underwater environments:

* Deep water (strong color attenuation)
* Turbid water (scattering and particles)
* Low light conditions
* Underwater robotic capture effects

---

## Future Work

* Cross-domain coral health classification
* GAN-based underwater simulation
* Domain adaptation training
* Augmentation quality evaluation metrics
* Multi-dataset benchmarking

---

## Applications

* Coral reef monitoring
* Marine ecosystem research
* Underwater computer vision
* Dataset augmentation for deep learning
* Autonomous underwater vehicle perception

---

This project is intended for academic and research use.

```
```
