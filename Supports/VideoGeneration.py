from Supports.VideoEditing import zoom_in_effect, zoom_out_effect
import numpy as np
from Supports.Files import find_filetype




import torch
from diffusers import StableDiffusionXLPipeline
import torch

from diffusers import StableDiffusionXLImg2ImgPipeline
from Supports.ImageEditing import zoom_at

from moviepy.editor import ImageSequenceClip
from moviepy.editor import *
from Supports.ImageEditing import plot_images 
from PIL import Image

from diffusers import DiffusionPipeline
from Supports.GPT import get_gpt_responce

from upscalers import upscale


# import pycuda.driver as cuda
# import pycuda.autoinit

   

def generate_video(responce_split:list[str],duration:float,savePath=False,Title_='', shared='', fps=24,kfps=0.5,height=1024,width = 768,zoom=1.2, kzoom=1.2,strength=0.6,kstrength=0.8,refiner_steps = 100,scale_factor=2,gpt_helper=True)->None:
    if not(savePath):
        savePath=os.path.dirname(os.path.realpath(__file__))
        
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    pipe = StableDiffusionXLPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe2 = StableDiffusionXLImg2ImgPipeline.from_pretrained("stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16",use_safetensors=True)
    
    
    # pipe = pipe.to("cuda")
    pipe.enable_model_cpu_offload()
    pipe2.enable_model_cpu_offload()

        
    n_video_clips = len(find_filetype(savePath,'.mp4'))    
    
    
    refiner = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        text_encoder_2=pipe.text_encoder_2,
        vae=pipe.vae,
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )

    refiner.enable_model_cpu_offload()
    torch.cuda.empty_cache()

        
        
    for k in range(len(responce_split) - n_video_clips):
        import gc
        torch.cuda.empty_cache()
        gc.collect()
        gpt_prompt=f'''Role: [
                        Video producer
                    ]

                    Objective: [
                        Describe an image, which will illustrate this sentance the best.
                        Sentance: {responce_split[k]}
                        Theme of the text: {Title_}
                        Include stage and lighting directions. 
                    ] 

                    Constraints: [
                        No more then 30 words.
                        Don't include word "sure" in your answer.
                        Only include prompt in your answer.
                        Don't include the described sentance or theme in your answer.
                        ]'''
        
        # k+=n_video_clips
        prompt = [get_gpt_responce(gpt_prompt) + shared]

        # prompt = [Title_ + '. ' + responce_split[k] + '. ' + shared]

        images = []


        seconds=np.round(duration[k])
        sequence =int(seconds/kfps)
        
        images_ref = []   
         

        if k==0:
            import gc
            torch.cuda.empty_cache()
            gc.collect()
            with torch.no_grad():
                image = pipe(prompt[0],height=height, width = width,num_inference_steps=100).images[0]
            # image.save(f'result_0.jpg')
            images.append(image)  


            for i, image in enumerate(images):
                with torch.no_grad():
                    image = refiner(
                        prompt[0],
                        num_inference_steps=refiner_steps,
                        # denoising_start=high_noise_frac,
                        image=image,
                    ).images[0]
                torch.cuda.empty_cache()
                gc.collect()
                images_ref.append(image)
                # image.save(f'result_ref_{i}.jpg')
        else:
            if gpt_helper:
                gpt_prompt=f'''Role: [
                        Video producer
                    ]
                    
                    Objective: [
                        Describe an the next frame after this frame.
                        This frame: {prompt}
                        Include stage and lighting directions. 
                    ] 

                    Constraints: [
                        No more then 30 words.
                        Don't include word "sure" in your answer.
                        Only include prompt in your answer.
                        Don't include the described sentance or theme in your answer.
                        ]'''
                prompt = [get_gpt_responce(gpt_prompt)+shared]
                
            with torch.no_grad():
                image = Image.fromarray(zoom_at(np.asarray(image), zoom=kzoom, coord=None))
                image = pipe2(prompt=prompt[0],image=image, strength=kstrength).images[0]  
        
        
        for j in range(sequence):
                torch.cuda.empty_cache()
                gc.collect()
                image = Image.fromarray(zoom_at(np.asarray(image), zoom=zoom, coord=None))
                with torch.no_grad():
                    image = pipe2(prompt=prompt[0],image=image, strength=strength).images[0]
                
                # image = refiner(
                #     prompt[0],
                #     num_inference_steps=refiner_steps,
                ## denoising_start=high_noise_frac,
                #     image=image,
                # ).images[0]
                images_ref.append(image) 
                

            
            


        
        plot_images(images_ref)

        weights = np.linspace(0,1,int(fps*kfps))

        height, width, layers = np.shape(images_ref[0])
        size = (width,height)

        clips=[]
        
        for j in range(len(images_ref)-1):
            img_array=[]
            for i in range(len(weights)):
                
                img=np.average([np.asarray(images_ref[j]),np.asarray(images_ref[j+1])],weights=[1-weights[i],weights[i]], axis=0).astype(np.uint8)
                with torch.no_grad():
                    img_array.append(np.array(upscale('R-ESRGAN General 4xV3', Image.fromarray(img), scale_factor)))
                torch.cuda.empty_cache()
                gc.collect()
            clips.append(ImageSequenceClip(img_array, fps=fps))
            # Write the video to a file in the path provided

            
            
        clip=concatenate_videoclips(clips)
        clip.fx(vfx.speedx, np.round(duration[k])/duration[k])
        # cuda.Device(0).use()
        clip.write_videofile(savePath+f'{k}.mp4')  #,codec='nvenc'
        
        # import cv2
        # import os
        # img_array=[]   
        # for j in range(len(images_ref)-1):
            
        #     for i in range(len(weights)):
                
        #         img=np.average([np.asarray(images_ref[j]),np.asarray(images_ref[j+1])],weights=[1-weights[i],weights[i]], axis=0).astype(np.uint8)
        #         with torch.no_grad():
        #             img_array.append(np.array(upscale('R-ESRGAN General 4xV3', Image.fromarray(img), scale_factor)))
        #         torch.cuda.empty_cache()
        #         gc.collect()
            
        # height, width, layers = np.shape(img_array[0])
        # size = (width,height)
        # video = cv2.VideoWriter(savePath+f'{k}.mp4', 0, fps, size)
        # for image in img_array:
        #     r,g,b=cv2.split(image)
        #     image=cv2.merge((b,g,r))
        #     video.write(image)
        # cv2.destroyAllWindows()
        # video.release()
        # os.system(f'ffmpeg -y -i {savePath+f"{k}.mp4"} -filter_complex "[0:v]setpts={np.round(duration[k])/duration[k]}*PTS[v];[0:a]atempo={duration[k]/np.round(duration[k])}[a]" -map "[v]" -map "[a]" {savePath+f"{k}.mp4"}')
        # torch.cuda.empty_cache()
        # gc.collect()
        