import transformers_musicGen
from transformers_musicGen import AutoProcessor, MusicgenForConditionalGeneration
# import torchaudio

processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
prompt_music= ["Dream pop syntwave scify background soft"]
inputs = processor(
    text=prompt_music,
    padding=True,
    return_tensors="pt",
)
audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=1400)
# generate_with_chroma 
import scipy

sampling_rate = model.config.audio_encoder.sampling_rate
scipy.io.wavfile.write("1_musicgen_out.wav", rate=sampling_rate, data=audio_values[0, 0].numpy())

# from pydub import AudioSegment

# sound1 = AudioSegment.from_wav("1_musicgen_out.wav")
# sound2 = AudioSegment.from_wav("1_musicgen_out.wav")

# combined_sounds = sound1 + sound2
# combined_sounds.export("2_musicgen_out.wav", format="wav")
# os.system(f'ffmpeg -i 2_musicgen_out.wav -codec:a libmp3lame -filter:a "atempo=55/60" -b:a 320K soundtrack.wav')