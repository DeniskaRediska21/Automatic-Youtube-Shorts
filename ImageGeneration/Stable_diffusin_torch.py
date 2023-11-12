import torch
torch.cuda.is_available()

from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt


model_id = "dreamlike-art/dreamlike-photoreal-2.0"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe = pipe.to("cuda")

prompts = ["Anime style aesthetic landscape, 90's vintage style, digital art, ultra HD, 8k, photoshopped, sharp focus, surrealism, akira style, detailed line art"]


images = []

for i, prompt in enumerate(prompts):
    image = pipe(prompt).images[0]
    image.save(f'result_{i}.jpg')
    images.append(image)
    
def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)