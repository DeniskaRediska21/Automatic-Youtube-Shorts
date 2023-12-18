from elevenlabs import voices, generate, play

audio = generate(
  text="Hi! My name is Antoni, nice to meet you!",
  voice="Antoni",
  model="eleven_monolingual_v1"
)

play(audio)