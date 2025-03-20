# Stable Diffusion Image Generator

An open source image generatore model 

## Diffusion Image generator integration in Colab 

1. Install necessary packages
  ```
!pip install diffusers transformers accelerate safetensors torch
```
2.  Import the packages
```py
import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os
```
3.  Load the diffusion model
```py
pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0")
pipe.to("cuda")  # Use GPU
```
4. Function to generate the images
```py
def generate_images(prompt, num_images=1):
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images} for prompt: '{prompt}'...")
        image = pipe(prompt, num_inference_steps=50, guidance_scale=8.5).images[0]
        filename = os.path.join(output_dir, f"image_{i+1}.png")
        image.save(filename)
        print(f"Saved: {filename}")
        image.show()
```
5.  User input function
```py
prompt = input("Enter your prompt: ")
num_images = int(input("Enter number of images to generate: "))
```
6.  Generate images and save them
```py
generate_images(prompt, num_images)
print(f"All images are saved in: {output_dir}")
```
7. Final Fine tuned code
```py
'import torch
from diffusers import StableDiffusionXLPipeline, EulerAncestralDiscreteScheduler
import os

# Load Stable Diffusion XL Model (High-Quality)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Enable memory optimizations
pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_model_cpu_offload()

# Set a better scheduler
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# Create output directory
output_dir = "realistic_images"
os.makedirs(output_dir, exist_ok=True)

# Function to generate and save high-quality images
def generate_images(prompt, num_images=1):
    for i in range(num_images):
        print(f"Generating image {i+1}/{num_images} for prompt: '{prompt}'...")
        
        image = pipe(
            prompt,
            num_inference_steps=50,  # More steps for better details
            guidance_scale=7.5  # Control image realism
        ).images[0]
        
        filename = os.path.join(output_dir, f"image_{i+1}.png")
        image.save(filename)
        print(f"Saved: {filename}")
        image.show()

# Get user input
prompt = input("Enter your prompt: ")
num_images = int(input("Enter number of images to generate: "))

# Generate high-quality images
generate_images(prompt, num_images)

print(f"All images are saved in: {output_dir}")
```
