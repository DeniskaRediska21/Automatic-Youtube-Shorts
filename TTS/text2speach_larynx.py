from larynx import text_to_speech
from larynx import wavfile
import numpy as np
import os

text_and_audios = text_to_speech('Hi, this is an example of converting text to audio. This is a bot speaking here, not a real human!')
text_and_audios
audios = []

# for _, audio in text_and_audios:
#         audios.append(audio)

# wavfile.write(data=np.concatenate(audios), rate=1, filename="a.wav")
# os.system("a.wav")