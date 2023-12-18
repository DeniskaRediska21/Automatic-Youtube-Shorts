import torch
torch.cuda.is_available()
torch.cuda.empty_cache()

from diffusers import StableDiffusionXLPipeline
import torch
import matplotlib.pyplot as plt
from diffusers import StableDiffusionXLImg2ImgPipeline
import numpy as np
import cv2
import cv2 as cv

def zoom_at(img, zoom, coord=None):
    """
    Simple image zooming without boundary checking.
    Centered at "coord", if given, else the image center.

    img: numpy.ndarray of shape (h,w,:)
    zoom: float
    coord: (float, float)
    """
    # Translate to zoomed coordinates
    h, w, _ = [ zoom * i for i in np.shape(img) ]
    
    if coord is None: cx, cy = w/2, h/2
    else: cx, cy = [ zoom*c for c in coord ]
    
    img = cv.resize( img, (0, 0), fx=zoom, fy=zoom)
    img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
               int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
               : ]
    
    return img


model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipe2 = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16",use_safetensors=True)
# pipe = pipe.to("cuda")
pipe.enable_model_cpu_offload()
pipe2.enable_model_cpu_offload()
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

shared= ' matte painting | masterpiece | hyperrealism| highly detailed| insanely detailed| intricate| cinematic lighting| depth of field | by studio ghibli, makoto shinkai, by artgerm, by wlop, by greg rutkowski, volumetric lighting, octane render, 4 k resolution, trending on artstation'

prompt = [f'  {shared}']

images = []


seconds=4
fps=24
kfps=1/4
sequence =int(seconds/kfps)

height=1024
width = 768

# height=1152, width = 2048

image = pipe(prompt[0],height=height, width = width,num_inference_steps=100).images[0]
# image.save(f'result_0.jpg')
images.append(image) 

    
    


from diffusers import DiffusionPipeline
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)

refiner.enable_model_cpu_offload()
n_steps = 100
high_noise_frac = 0.8
images_ref = []
torch.cuda.empty_cache()

strength=0.6
zoom=1.5


for i, image in enumerate(images):
    
    image = refiner(
        prompt[0],
        num_inference_steps=n_steps,
        # denoising_start=high_noise_frac,
        image=image,
    ).images[0]
    images_ref.append(image)
    image.save(f'result_ref_{i}.jpg')
# 'geverate the next frame'
from PIL import Image
for j in range(sequence):
        image = Image.fromarray(zoom_at(np.asarray(image), zoom=zoom, coord=None))
        image = pipe2(prompt=prompt[0],image=image, strength=strength).images[0]
        # image = pipe2(prompt=f'geverate the next frame {shared}',image=image, strength=strength).images[0]
        # image.save(f'result_ref{j+1}.jpg')
        images_ref.append(image) 
    
    
def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images_ref)

weights = np.linspace(0,1,int(fps*kfps))
import cv2

height, width, layers = np.shape(images_ref[0])
size = (width,height)

clips=[]
from moviepy.editor import ImageSequenceClip
from moviepy.editor import *
for j in range(len(images_ref)-1):
    img_array=[]
    for i in range(len(weights)):
        
        img=np.average([np.asarray(images_ref[j]),np.asarray(images_ref[j+1])],weights=[1-weights[i],weights[i]], axis=0)
        img_array.append(img)
    clips.append(ImageSequenceClip(img_array, fps=fps))
    # Write the video to a file in the path provided
    
    
clip=concatenate_videoclips(clips)
clip.write_videofile("full.mp4")

