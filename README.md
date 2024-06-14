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

Remarks:
+ By default 'mistral' model will be used for all the text generation
  
**Outputs can be seen at:**
[Beyond The Concrete Jungle](https://www.youtube.com/@BeyondTheConcreteJungle)

TODO:
[x] Implement the option to use local open source LLMs for text generation
[x] Implement an option to use models other than SDXL 
[x] Implement Latent Consistency models for up to 10x increase in generation speeds, with reduced quality

