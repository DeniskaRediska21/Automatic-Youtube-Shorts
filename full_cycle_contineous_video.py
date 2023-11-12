

import os
from Supports.TextEditing import split_into_sentences, cleenup
from Supports.GPT import get_gpt_responce

generate_music= False
topics=['Deserts', 'Bisons']
not_=''
for topic in range(len(topics)):
    #  good topic about black holes180
    prompt = f'''Role: [
                        Storyteller.
                    ]

                    Objective: [
                        Write a 120 word short essey about some interesting {topics[topic]} fact, always start with the title on a separate line. Make sure to prioritize viewer retention time. 
                    ] 

                    Constraints: [
                        Your answer should have at less than 150 words.
                        Don't include word "sure" in your answer.
                        ]'''
    responce = get_gpt_responce(prompt)
    
    responce, responce_split, Title_, Title = cleenup(responce)

    prompt = f"Write a short description for a video about {Title_}. Include Tags in the end. Perforn search engine optimization. "
 
    savePath = f"Result//{Title}//"
    
    description = get_gpt_responce(prompt)


    
    
    try:
        os.mkdir(savePath)
    except:
        print('duplicate directory')
    
    


    with open(savePath+"description.txt", "w") as text_file:
        text_file.write(description)
    with open(savePath+"text.txt", "w") as text_file:
        text_file.write(responce)
    with open(savePath+"title.txt", "w") as text_file:
        text_file.write(get_gpt_responce(f'Transform this title to be more clickbait. Title: {Title_}. Dont use smiles. No more then 3 words'))
        
        
    # TTS Section________________________________________________________________
    from Supports.TTS import tts
    tts(responce_split, savePath, preset = "ultra_fast", voice = 'geralt')
    
    
    # Video Generation Section________________________________________________________________
    # shared= 'bright colors | light colors | masterpiece | hyperrealism| highly detailed| insanely detailed| intricate| cinematic lighting| depth of field | volumetric lighting, octane render, 4 k resolution, trending on artstation'
    shared= 'surrealist painting | oil painting | bright colors | high contrast | masterpiece | highly detailed| cinematic lighting'

    #  by studio ghibli, makoto shinkai, by artgerm, by wlop, by greg rutkowski, 
    from Supports.VideoGeneration import generate_video
    from moviepy.editor import *
    from Supports.Files import find_filetype
    
    audio_clips=find_filetype(savePath,'.wav')
    duration=[]
    for audio_clip in audio_clips:
        duration.append(AudioFileClip(audio_clip).duration)  
        
        
    generate_video(responce_split,Title_=Title_,duration=duration,savePath=savePath, shared=shared , kstrength=0.5,strength=0.7, zoom=1.03, kzoom=1.1, kfps = 1/4)
    # Generate videos from images__________________________________________________

    from Supports.VideoEditing import add_sound
    add_sound(savePath)
    
    from Supports.VideoEditing import add_subs
    add_subs(savePath,responce_split)
    
    
    # Music____________________________________________________________________
    from Supports.Music import music_gen
    audio_file=music_gen(generate_music,savePath)
    
    
    # Add music to video___________________________________________________
    from Supports.VideoEditing import addmusic

    video_file = f'{savePath}final.mp4'
    output_name = f'{savePath}video_with_music.mp4' 
    output_name2 = f'{savePath}video_with_music.mkv' 
    v1 = addmusic(video_file, audio_file, savePath,output_name, volume=0.35)
    os.system(f'ffmpeg -y -ss 0 -to {v1} -i {output_name} -c copy {output_name}')
    os.system(f'ffmpeg -y -i {output_name} -filter_complex "[0:v]setpts={1/1.2}*PTS[v];[0:a]atempo={1.2}[a]" -map "[v]" -map "[a]" {output_name2}')
    if v1>55:
        output_name3=f"{savePath}video_with_music_59.mkv"
        os.system(f'ffmpeg -y -i {output_name} -filter_complex "[0:v]setpts={55/max(v1,55)}*PTS[v];[0:a]atempo={v1/55}[a]" -map "[v]" -map "[a]" {output_name3}')


    