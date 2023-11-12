import math
from PIL import Image
import numpy
from moviepy.editor import *
from Supports.Files import find_filetype
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeAudioClip, afx
from pydub import AudioSegment
import moviepy
import librosa
import numpy as np



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






def add_sound(savePath:str, fade=0, fps=24)->None:
    audio_clips=find_filetype(savePath,'.wav')
    
    video_clips = find_filetype(savePath,'.mp4')
    fade = 0 #0.25
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
        
        video_clip=video_clip.fx(vfx.speedx, video_clip.duration/audio_clip.duration)
        
        final_audio = CompositeAudioClip([audio_clip])
        video_clip.audio = final_audio
        final_clip = video_clip.fx(vfx.fadein, fade)
        final_clip = final_clip.fx(vfx.fadeout, fade)
        
        final_clip.write_videofile(savePath+str(i)+"_with_sound.mp4",codec='libx264',fps = fps)
            


def addsubs(video_file:str, caption:str,output_name:str, fade =0,fps=24)->None:
    from moviepy.editor import VideoFileClip,TextClip,CompositeVideoClip
    video_clip = VideoFileClip(video_file)
    audio=video_clip.audio.set_duration(video_clip.duration)
    # Generate a text clip 
    # font='Georgia-Regular'
    font='Futura'

    txt_clip = TextClip(caption, fontsize = 50, color = 'white', font=font,
    method='caption', align='center',stroke_color='gray42',stroke_width=2, size=list(3/4*np.array(video_clip.size)))
        
    # setting position of text in the center and duration will be 10 seconds 
    txt_clip = txt_clip.set_position(("center","bottom")).set_duration(video_clip.duration).crossfadein(0.2).crossfadeout(0.2)
    # audio=video_clip.audio
    video_clip = CompositeVideoClip([video_clip, txt_clip]) 
    video_clip = video_clip.fx(vfx.fadein, fade)
    video_clip =video_clip.set_audio(audio)
    video_clip = video_clip.fx(vfx.fadeout, fade).audio_fadein(0.1).audio_fadeout(0.1)
    video_clip.write_videofile(output_name,codec='libx264',fps = fps)
    return video_clip



def addmusic(video_file:str, audio_file:str, savePath:str,output_name:str,fps=24, volume=0.05)->None:
    video_clip = VideoFileClip(video_file)
    end = video_clip.end
    sound1 = AudioSegment.from_wav(audio_file)
    combined_sounds = sound1*int(np.round(end/librosa.get_duration(path=audio_file)))
    combined_sounds.export(f"{savePath}soundtrack.wav", format="wav")
    os.system(f'ffmpeg -y -i {savePath}soundtrack.wav -codec:a libmp3lame -filter:a "atempo={librosa.get_duration(path=f"{savePath}soundtrack.wav")/video_clip.end}" -b:a 320K {savePath}tmp.wav')
    audio_clip = AudioFileClip(f'{savePath}tmp.wav')
    # audio_clip = moviepy.audio.fx.all.speedx(combined_sounds, librosa.get_duration(path=audio_file)/video_clip.end)
    audio_clip = audio_clip.fx(afx.volumex, volume)
    audio_clip =moviepy.audio.fx.all.audio_fadein(audio_clip,2)
    audio_clip =moviepy.audio.fx.all.audio_fadeout(audio_clip,2)

    new_audioclip = CompositeAudioClip([video_clip.audio, audio_clip])
    new_audioclip=new_audioclip.fx(afx.volumex,3)
    video_clip.audio = new_audioclip

    video_clip.write_videofile(output_name,fps = fps)
    return end

def add_subs(savePath:str,responce_split:list[str],fps=24)->None:
    from Supports.VideoEditing import addsubs
    from Supports.Files import find_filetype
    video_clips_with_sound=find_filetype(savePath,'_with_sound.mp4')
    loaded_video_list = []
    for i,video in enumerate(video_clips_with_sound):
        print(f"Adding video file:{video}")
        video_file = f'{savePath}{i}_with_sound.mp4'
        output_name = f'{savePath}{i}_with_subs.mp4'
        final_clip=addsubs(video_file, responce_split[i], output_name)
        loaded_video_list.append(final_clip)

    final_video = concatenate_videoclips(loaded_video_list)
    final_video.write_videofile(f"{savePath}final.mp4",fps = fps)
   
   
from upscalers import upscale
import gc
import torch
from moviepy.editor import *
 
def video_from_images_moviepy(images_ref,duration,savePath,fps,kfps,scale_factor=2):
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
        clip.fx(vfx.speedx, np.round(duration)/duration)
        clip.write_videofile(savePath)  
