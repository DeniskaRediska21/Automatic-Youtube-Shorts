import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices
    
    
def tts(responce_split:list[str], savePath:str, preset="ultra_fast", voice='geralt') -> None:
    # This will download all the models used by Tortoise from the HF hub.
    # tts = TextToSpeech()
    # If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
    with torch.no_grad():
        tts = TextToSpeech(use_deepspeed=False, kv_cache=True, half = True)
    # Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
    # preset = "fast"
    # preset = "standard"

    # Pick one of the voices from the output above
    # voice = 'geralt'
    # tom freeman myself
    # Load it and send it through Tortoise.
    voice_samples, conditioning_latents = load_voice(voice)

    with torch.no_grad():
        for i,sentance in enumerate(responce_split):
            
            gen = tts.tts_with_preset(sentance, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                                        preset=preset)
            torchaudio.save(savePath+str(i)+'.wav', gen.squeeze(0).cpu(), 24000)