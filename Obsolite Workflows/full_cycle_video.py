
    
import re
import g4f
import os

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|edu|me)"
digits = "([0-9])"
multiple_dots = r'\.{2,}'

def split_into_sentences(text: str) -> list[str]:
    """
    Split the text into sentences.

    If the text contains substrings "<prd>" or "<stop>", they would lead 
    to incorrect splitting because they are used as markers for splitting.

    :param text: text to be split into sentences
    :type text: str

    :return: list of sentences
    :rtype: list[str]
    """
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    text = re.sub(multiple_dots, lambda match: "<prd>" * len(match.group(0)) + "<stop>", text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = [s.strip() for s in sentences]
    if sentences and not sentences[-1]: sentences = sentences[:-1]
    return sentences
# ,'Ants','bees'

generate_music= False
topics=['Clouds']
not_=''
for i in range(len(topics)):
    #  good topic about black holes180
    prompt = f"Write a 120 word short essey about some interesting {topics[i]} fact, always start with the title on a separate line. Make sure to prioritize viewer retention time."
    
    
    
    prompt = f'''Role: [
                            Storyteller.
                        ]

                        Objective: [
                            Write a 120 word short essey about some interesting {topics[i]} fact, always start with the title on a separate line. Make sure to prioritize viewer retention time. 
                        ] 

                        Constraints: [
                            Your answer should have at less than 150 words.
                            Don't include word "sure" in your answer.
                            ]'''
    try:
        provider=g4f.Provider.DeepAi
        response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
    except:
        try:
            provider=g4f.Provider.Aichat
            response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
        except:
            try:
                response = g4f.ChatCompletion.create(model=g4f.Model.gpt_4,provider=g4f.Provider.ChatgptAi	, messages=[{"role": "user", "content": prompt}]) 
            except:
                try:
                    provider=g4f.Provider.ChatgptLogin
                    response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
                except:
                    try:
                        provider=g4f.Provider.AiService
                        response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
                    except:
                        provider=g4f.Provider.Lockchat
                        response = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
 
    # response = g4f.ChatCompletion.create(model='gpt-3.5-turbo', provider=g4f.Provider.DeepAi, messages=[
    #                                      {"role": "user", "content": "Pick a good youtube shorts topic and write a 60-second essey on that. Make sure to prioritize viewer retention time.  Write compact"}], stream=False)

    
    # for message in response:
    #     print(message)  
        

    response = re.sub("\[.*?\]","",response)
    response = re.sub("\"","",response)
    response = re.sub("Topic:","",response)
    response = ".\n".join([ll.rstrip() for ll in response.splitlines() if ll.strip()])
    response = re.sub("\.\.",".",response)
    response = re.sub("Title: ","",response)
    response = re.sub("Voiceover: ","",response)


    # Надо разделить примерно по 20 слов для генерации изображений 
    responce_split = split_into_sentences(response)
    
    for i,sentence in enumerate(responce_split):
        responce_split[i]= re.sub("\.","",sentence)
        
        
    print(responce_split)
    # Дальше генерация изображений
    Title = responce_split[0]
    Title = re.sub("\.","",Title)

    Title = re.sub("Title:","",Title)
    Title_=Title

    del(responce_split[responce_split=='.'])


    prompt = f"Write a short description for a video about {Title_}. Include Tags in the end. Perforn search engine optimization. "
    try:
        provider=g4f.Provider.DeepAi
        description = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
    except:
        try:
            provider=g4f.Provider.Aichat
            description = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
        except:
            try:
                description = g4f.ChatCompletion.create(model=g4f.Model.gpt_4,provider=g4f.Provider.ChatgptAi	, messages=[{"role": "user", "content": prompt}])
 
            except:
                try:
                    provider=g4f.Provider.ChatgptLogin
                    description = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
                except:
                    try:
                        provider=g4f.Provider.AiService
                        description = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
                    except:
                        provider=g4f.Provider.Lockchat
                        description = g4f.ChatCompletion.create(model="gpt-3.5-turbo", provider=provider, messages=[{"role": "user", "content": prompt}]) 
                        
 
        
    Title = re.sub(":","_",Title)
    Title = re.sub(" ","_",Title)

    
    savePath = f"Result//{Title}//"
    try:
        os.mkdir(savePath)
    except:
        print('duplicate directory')
    
    del(responce_split[0])


    with open(savePath+"description.txt", "w") as text_file:
        text_file.write(description)
    with open(savePath+"text.txt", "w") as text_file:
        text_file.write(response)
    with open(savePath+"title.txt", "w") as text_file:
        text_file.write(Title_)
        
    import numpy as np
    print(np.size(response.split()))
 
        
    # TTS Section________________________________________________________________
    import torch
    import torchaudio
    import torch.nn as nn
    import torch.nn.functional as F

    from tortoise.api import TextToSpeech
    from tortoise.utils.audio import load_audio, load_voice, load_voices

    # This will download all the models used by Tortoise from the HF hub.
    # tts = TextToSpeech()
    # If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
    tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half = True)

    # Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
    # preset = "fast"
    preset = "ultra_fast"

    # Pick one of the voices from the output above
    voice = 'geralt'
    # tom freeman myself
    # Load it and send it through Tortoise.
    voice_samples, conditioning_latents = load_voice(voice)


    for i,sentance in enumerate(responce_split):
        gen = tts.tts_with_preset(sentance, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                                preset=preset)
        torchaudio.save(savePath+str(i)+'.wav', gen.squeeze(0).cpu(), 24000)
    

    # IPython.display.Audio('generated.wav')
        
    # IMG Section________________________________________________________________
    import moviepy.editor as mp
    import math
    from PIL import Image
    import numpy


    def zoom_in_effect(clip, zoom_ratio=0.04):
        def effect(get_frame, t):
            img = Image.fromarray(get_frame(t))
            base_size = img.size

            new_size = [
                math.ceil(img.size[0] * (1 + (zoom_ratio * t))),
                math.ceil(img.size[1] * (1 + (zoom_ratio * t)))
            ]

            # The new dimensions must be even.
            new_size[0] = new_size[0] + (new_size[0] % 2)
            new_size[1] = new_size[1] + (new_size[1] % 2)

            img = img.resize(new_size, Image.LANCZOS)

            x = math.ceil((new_size[0] - base_size[0]) / 2)
            y = math.ceil((new_size[1] - base_size[1]) / 2)

            img = img.crop([
                x, y, new_size[0] - x, new_size[1] - y
            ]).resize(base_size, Image.LANCZOS)

            result = numpy.array(img)
            img.close()

            return result

        return clip.fl(effect)


    def zoom_out_effect(clip, zoom_ratio=0.04):
        def effect(get_frame, t):
            img = Image.fromarray(get_frame(t))
            base_size = img.size

            new_size = [
                math.ceil(img.size[0] * np.max([1,(1.5 - (zoom_ratio * t))])),
                math.ceil(img.size[1] * np.max([1,(1.5 - (zoom_ratio * t))]))
            ]

            # The new dimensions must be even.
            new_size[0] = new_size[0] + (new_size[0] % 2)
            new_size[1] = new_size[1] + (new_size[1] % 2)

            img = img.resize(new_size, Image.LANCZOS)

            x = math.ceil((new_size[0] - base_size[0]) / 2)
            y = math.ceil((new_size[1] - base_size[1]) / 2)

            img = img.crop([
                x, y, new_size[0] - x, new_size[1] - y
            ]).resize(base_size, Image.LANCZOS)

            result = numpy.array(img)
            img.close()

            return result

        return clip.fl(effect)
    
    
    import librosa

    from moviepy.editor import *

    import cv2
    import numpy as np
    import glob

    import re
    def sorted_alphanumeric(data):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
        return sorted(data, key=alphanum_key)


    wav_folder = savePath
    audio_clips = [os.path.join(wav_folder,img)
                for img in sorted_alphanumeric(os.listdir(wav_folder))
                if img.endswith(".wav")]
    duration=[]
    for audio_clip in audio_clips:
        duration.append(AudioFileClip(audio_clip).duration)
    
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
    video_clips = []

    for k, prompt in enumerate(responce_split):
        prompt = [Title_ + '. ' + prompt[0]]

        images = []


        seconds=np.round(duration[k])
        fps=24
        kfps=1/2
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
        clip.fx(vfx.speedx, np.round(duration[k])/duration[k])
        clip.write_videofile(savePath+f'{k}.mp4')
        # video_clips.append(savePath+f'{k}.mp4')
        # os.system(f'ffmpeg -y -i {video_clips[-1]} -filter_complex "[0:v]setpts={duration[k]/np.round(duration[k])}*PTS[v];[0:a]atempo={np.round(duration[k])/duration[k]}[a]" -map "[v]" -map "[a]" {video_clips[-1]}')

    video_clips = [os.path.join(wav_folder,img)
            for img in sorted_alphanumeric(os.listdir(wav_folder))
            if img.endswith(".mp4")]
    # Generate videos from images__________________________________________________



    
    


    # img_array = []
    # duration =[]
    # for i,image in enumerate(images_ref):
    #     img = np.array(image)
    #     height, width, layers = img.shape
    #     size = (width,height)
    #     img_array.append(img)
    #     duration.append(librosa.get_duration(path=audio_clips[i]))

    
    
    loaded_video_list = []
    # pyMovie
    from moviepy.editor import *
    fade = 0.25
    for i,video_file in enumerate(video_clips):
        # load the video
        video_file = video_clips[i]
        video_clip = VideoFileClip(video_file)
        # if np.mod(i,2):
        #     video_clip = zoom_out_effect(video_clip, zoom_ratio=0.04)
        # else:
        #     video_clip = zoom_in_effect(video_clip, zoom_ratio=0.04)
            
        # load the audio
        audio_file = audio_clips[i]
        audio_clip = AudioFileClip(audio_file)
        final_audio = CompositeAudioClip([audio_clip])
        video_clip.audio = final_audio
        final_clip = video_clip.fx(vfx.fadein, fade)
        final_clip = final_clip.fx(vfx.fadeout, fade)
        
        loaded_video_list.append(final_clip)
        final_clip.write_videofile(savePath+str(i)+"_with_sound.mp4",codec='libx264',fps = 10)
        
    video_clips_with_sound = [os.path.join(wav_folder,img)
                            for img in sorted_alphanumeric(os.listdir(wav_folder))
                            if img.endswith("_with_sound.mp4")]

    # -----------------------------------------------------
    def addsubs(video_file, caption,output_name):
        from moviepy.editor import VideoFileClip,TextClip,CompositeVideoClip
        video_clip = VideoFileClip(video_file)
        audio=video_clip.audio.set_duration(video_clip.duration)
        # Generate a text clip 
        # font='Georgia-Regular'
        font='Batang'

        txt_clip = TextClip(caption, fontsize = 50, color = 'white', font=font,
        method='caption', align='center',stroke_color='black',stroke_width=2, size=list(3/4*np.array(video_clip.size))) 
            
        # setting position of text in the center and duration will be 10 seconds 
        txt_clip = txt_clip.set_pos('center').set_duration(video_clip.duration) 
        # audio=video_clip.audio
        video_clip = CompositeVideoClip([video_clip, txt_clip]) 
        video_clip = video_clip.fx(vfx.fadein, fade)
        video_clip =video_clip.set_audio(audio)
        video_clip = video_clip.fx(vfx.fadeout, fade).audio_fadein(0.1).audio_fadeout(0.1)
        video_clip.write_videofile(output_name,codec='libx264',fps = 25)
        return video_clip

        
    loaded_video_list = []

    # ---------------------------------------------------------
    for i,video in enumerate(video_clips_with_sound):
        print(f"Adding video file:{video}")
        video_file = f'{savePath}{i}_with_sound.mp4'
        output_name = f'{savePath}{i}_with_subs.mp4'
        final_clip=addsubs(video_file, responce_split[i], output_name)
        loaded_video_list.append(final_clip)

        
    final_video = concatenate_videoclips(loaded_video_list)
    final_video.write_videofile(f"{savePath}final.mp4",fps = 10)
    # if final_video.end/59>1:
    #     # final_video = concatenate_videoclips(loaded_video_list)
    #     final_video = final_video.fx( vfx.speedx, final_video.end/50)
    # final_video.write_videofile(f"{savePath}final_59.mp4",fps = 10)
    # speedup = final_video.end/59
    # !ffmpeg -i Result//final.mp4 -filter_complex f'[0:v]setpts=0.5*PTS[v];[0:a]atempo={speedup}[a]' -map "[v]" -map "[a]" Result//final_59_s.mp4 




    # Music____________________________________________________________________

    
    from transformers_musicGen import AutoProcessor, MusicgenForConditionalGeneration
    import torchaudio_musicGen
    import torch
    from pydub import AudioSegment
    torch.cuda.empty_cache()
    if generate_music:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-large")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
        prompt_music= ["Dreamy syntwave background"]

        inputs = processor(
            text=prompt_music,
            padding=True,
            return_tensors="pt",
        )
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1400)

        import scipy
                    
        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(f"{savePath}a_1_musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

        

        # sound1 = AudioSegment.from_wav(f"{savePath}a_1_musicgen_out.wav")
        # combined_sounds = sound1 + sound2 +sound1+sound1+sound1
        # combined_sounds.export(f"{savePath}soundtrack.wav", format="wav")
        # os.system(f'ffmpeg -y -i {savePath}a_2_musicgen_out.wav -codec:a libmp3lame -filter:a "atempo=50/60" -b:a 320K {savePath}soundtrack.wav')

        # Add music to video___________________________________________________
        audio_file = f"{savePath}a_1_musicgen_out.wav"  
    else:
        sounds = [os.path.join("Result\\Soundtracks",img)
                            for img in sorted_alphanumeric(os.listdir('Result\\Soundtracks'))
                            if img.endswith(".wav")]
        import random 
        audio_file= sounds[random.randint(0,len(sounds))]

    def addsound(video_file, audio_file,output_name):
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, afx
        import moviepy
        import librosa
        video_clip = VideoFileClip(video_file)
        end = video_clip.end
        sound1 = AudioSegment.from_wav(audio_file)
        combined_sounds = sound1*int(np.round(end/librosa.get_duration(path=audio_file)))
        combined_sounds.export(f"{savePath}soundtrack.wav", format="wav")
        os.system(f'ffmpeg -y -i {savePath}soundtrack.wav -codec:a libmp3lame -filter:a "atempo={librosa.get_duration(path=f"{savePath}soundtrack.wav")/video_clip.end}" -b:a 320K {savePath}tmp.wav')
        audio_clip = AudioFileClip(f'{savePath}tmp.wav')
        # audio_clip = moviepy.audio.fx.all.speedx(combined_sounds, librosa.get_duration(path=audio_file)/video_clip.end)
        audio_clip = audio_clip.fx(afx.volumex, 0.05)
        audio_clip =moviepy.audio.fx.all.audio_fadein(audio_clip,2)
        audio_clip =moviepy.audio.fx.all.audio_fadeout(audio_clip,2)

        new_audioclip = CompositeAudioClip([video_clip.audio, audio_clip])
        new_audioclip=new_audioclip.fx(afx.volumex,3)
        video_clip.audio = new_audioclip

        video_clip.write_videofile(output_name,fps = 60)
        return end
        
    
    video_file = f'{savePath}final.mp4'
    output_name = f'{savePath}video_with_music.mp4' 
    output_name3 = f'{savePath}video_with_music.mkv' 
    v1 = addsound(video_file, audio_file,output_name)
    os.system(f'ffmpeg -y -ss 0 -to {v1} -i {output_name} -c copy {output_name}')
    os.system(f'ffmpeg -y -i {output_name} -filter_complex "[0:v]setpts={1/1.2}*PTS[v];[0:a]atempo={1.2}[a]" -map "[v]" -map "[a]" {output_name3}')
    if v1>55:
        output2_name=f"{savePath}video_with_music_59.mkv"
        os.system(f'ffmpeg -y -i {output_name} -filter_complex "[0:v]setpts={55/max(v1,55)}*PTS[v];[0:a]atempo={v1/55}[a]" -map "[v]" -map "[a]" {output2_name}')


    