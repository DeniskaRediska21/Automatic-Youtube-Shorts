
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

import IPython

from tortoise.api import TextToSpeech
from tortoise.utils.audio import load_audio, load_voice, load_voices

# This will download all the models used by Tortoise from the HF hub.
# tts = TextToSpeech()
# If you want to use deepspeed the pass use_deepspeed=True nearly 2x faster than normal
tts = TextToSpeech(use_deepspeed=False, kv_cache=True)


# This is the text that will be spoken.
text = ['"The Incredible Size of the Universe - Imagine a world without limits.',
 'But what if I told you that our universe is infinite in size, a never-ending expanse of stars, planets, and galaxies?',
 "It's enough to make your head spin, but it's also a reminder that we're just a small part of something much, much bigger.",
 'Prepare to be amazed"!']

# Here's something for the poetically inclined.. (set text=)
"""
Then took the other, as just as fair,
And having perhaps the better claim,
Because it was grassy and wanted wear;
Though as for that the passing there
Had worn them really about the same,"""

# Pick a "preset mode" to determine quality. Options: {"ultra_fast", "fast" (default), "standard", "high_quality"}. See docs in api.py
preset = "ultra_fast"


# Pick one of the voices from the output above
voice = 'geralt'
# tom freeman myself

# Load it and send it through Tortoise.
for i,sentance in enumerate(text):
    voice_samples, conditioning_latents = load_voice(voice)
    gen = tts.tts_with_preset(sentance, voice_samples=voice_samples, conditioning_latents=conditioning_latents, 
                            preset=preset)
    torchaudio.save('generated.wav', gen.squeeze(0).cpu(), 24000)
    IPython.display.Audio('generated.wav')