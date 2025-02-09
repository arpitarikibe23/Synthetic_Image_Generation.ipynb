

!pip install --upgrade diffusers transformers -q

from pathlib import Path
import tqdm
import torch
import pandas as pd
import numpy as np
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import cv2

import torch

class CFG:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    generator = torch.Generator(device).manual_seed(seed)
    image_gen_steps = 35
    image_gen_model_id = "stabilityai/stable-diffusion-2"
    image_gen_size = (400, 400)
    image_gen_guidance_scale = 9
    prompt_gen_model_id = "gpt2"
    prompt_dataset_size = 6
    prompt_max_length = 12

print(f"Using device: {CFG.device}")  # Output the device being used

image_gen_model = StableDiffusionPipeline.from_pretrained(
    CFG.image_gen_model_id, torch_dtype=torch.float16,
    revision="fp16", use_auth_token='your_hugging_face_auth_token', guidance_scale=9
)
image_gen_model = image_gen_model.to(CFG.device)

def generate_image(prompt, model):
    image = model(
        prompt, num_inference_steps=CFG.image_gen_steps,
        generator=CFG.generator,
        guidance_scale=CFG.image_gen_guidance_scale
    ).images[0]

    image = image.resize(CFG.image_gen_size)
    return image

generate_image("flowers in garden ", image_gen_model)

"""# **Task 2**"""

!pip install pillow torchvision matplotlib opencv-python -q

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

# Define image paths
image_paths = ["/content/drive/MyDrive/Colab Notebooks/Youtube-main/Synthetic Image Generation-main/GenerativeAI/image1.jpg", "/content/drive/MyDrive/Colab Notebooks/Youtube-main/Synthetic Image Generation-main/GenerativeAI/image2.jpg", "/content/drive/MyDrive/Colab Notebooks/Youtube-main/Synthetic Image Generation-main/GenerativeAI/image3.webp"]

# Load and display original images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, image_path in enumerate(image_paths):
    image = Image.open(image_path)
    axes[i].imshow(image)
    axes[i].axis("off")
    axes[i].set_title(f"Original {i+1}")

plt.show()

from google.colab import drive
drive.mount('/content/drive')

# Resize images to 224x224 using PIL
resized_images = [Image.open(img).resize((224, 224)) for img in image_paths]

# Display resized images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, img in enumerate(resized_images):
    axes[i].imshow(img)
    axes[i].axis("off")
    axes[i].set_title(f"Resized {i+1}")

plt.show()

# Define transformation: Convert to tensor and normalize (scale values between 0 and 1)
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to (C, H, W) format
    transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize pixel values to [-1, 1]
])

# Apply transformation to images
tensor_images = [transform(img) for img in resized_images]

# Print shape and pixel range of the first image
print("Tensor Shape:", tensor_images[0].shape)  # Should be (C, 224, 224)
print("Pixel Value Range:", tensor_images[0].min().item(), "to", tensor_images[0].max().item())

# Convert images to grayscale using PIL
gray_images = [img.convert("L") for img in resized_images]

# Convert grayscale images to tensor
gray_tensor_images = [transforms.ToTensor()(img) for img in gray_images]

# Display grayscale images
fig, axes = plt.subplots(1, 3, figsize=(12, 4))

for i, img in enumerate(gray_images):
    axes[i].imshow(img, cmap="gray")
    axes[i].axis("off")
    axes[i].set_title(f"Grayscale {i+1}")

plt.show()

# Save resized and grayscale images
output_dir = "preprocessed_images"
os.makedirs(output_dir, exist_ok=True)

for i, img in enumerate(resized_images):
    img.save(f"{output_dir}/resized_image_{i+1}.png")

for i, img in enumerate(gray_images):
    img.save(f"{output_dir}/grayscale_image_{i+1}.png")

print("✅ Preprocessed images saved successfully in 'preprocessed_images' folder!")




















