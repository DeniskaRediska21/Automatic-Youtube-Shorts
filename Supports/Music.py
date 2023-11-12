from transformers_musicGen import AutoProcessor, MusicgenForConditionalGeneration
import torch
import random 
import scipy
import os
from Supports.Files import find_filetype

torch.cuda.empty_cache()

def music_gen(generate_music=True,savePath=False,prompt_music="Dreamy syntwave background"):
    if not(savePath):
        savePath=os.path.dirname(os.path.realpath(__file__))
   
    if generate_music:
        processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
        model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
        model=model.to('cuda')
        
        
        inputs = processor(
            text=prompt_music,
            padding=True,
            return_tensors="pt",
        ).to('cuda')
        
        with torch.no_grad():
            audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1400)

        
                    
        sampling_rate = model.config.audio_encoder.sampling_rate
        scipy.io.wavfile.write(f"{savePath}a_1_musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].cpu().numpy())

        audio_file = f"{savePath}a_1_musicgen_out.wav"  
    else:
        sounds = find_filetype('Result\\Soundtracks','.wav')
        audio_file= sounds[random.randint(0,len(sounds)-1)]
    return audio_file