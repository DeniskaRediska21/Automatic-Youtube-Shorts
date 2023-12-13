# Automatic Youtube Shorts

This is a programm that aims at automating the process of making short-format educational videos on specified topics

The final videos include:
+ The script, title, description
+ The voiceower
+ The generated video
+ The suctitles
+ Basic editing


How it all works:

1) The script, title and description is generated using an LLM (ChatGPT throug `G4f` sdk)
2) The voiceower is generated for the provided script, using `tortoise_tts` 
3) The sequence of coherent images is generated using `StableDiffusion`. This includes 1 'keyframe' image for every sentance which is creates
focusing on the sentance provided, and the series of the subframe images, which are placed between keyframe images and are generated using
the `image2image StableDiffusion` model.
4) All of the generated frames are upscaled using `Upscalers` library and refined by the `StableDiffusion refiner` model
5) The background music is generated using the `MusicGen` library
6) All of the generated content is edited together through `moviepy` library



| Mind-Bloiwng:Venus, Our Solar Systems's Fiery Planet | Unbelievable Sunflower Secrets | The Evolution of Rockets: A Journey to the Stars |
|---|---|---|
|![Video 1](Media/Mind-Blowing_Venus, Our Solar System's Fiery Planet!-(1080p).mp4)|![Video 2](Media/Unbelievable Sunflower Secrets!-(1080p).mp4)|![Video 3](Media/The Evolution of Rockets_ A Journey to the Stars-(720p60).mp4)|

<video src="https://github.com/DeniskaRediska21/Automatic-Youtube-Shorts/tree/main/Media/https://github.com/DeniskaRediska21/Automatic-Youtube-Shorts/blob/main/Media/Mind-Blowing_Venus%2C%20Our%20Solar%20System's%20Fiery%20Planet!-(1080p).mp4">


**More outputs can be seen at:**
[Beyond The Concrete Jungle](https://www.youtube.com/@BeyondTheConcreteJungle)
