import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')
for voice in voices:
    print(voice)
    if voice.name == u'Microsoft David Desktop - English (United States)':
            engine.setProperty('voice', voice.id)
            break

engine.say("Topic: The Great Red Spot on Jupiter. Have you ever heard of the Great Red Spot on Jupiter? It's a giant storm larger than the size of the Earth that has been raging on the planet for over 350 years. Can you believe that? It's like one never-ending hurricane. The storm is so strong that it creates wind speeds of up to 400 miles per hour, making it one of the most powerful storms in the solar system. And while scientists have been studying it for hundreds of years, it's still a mystery as to how it has managed to last for so long. One theory is that the storm is sustained by the rotation of Jupiter, which creates a kind of stability that prevents it from breaking up like other storms on the planet. But with the help of new technology and space probes, scientists are getting closer to uncovering the mysteries of the Great Red Spot. They're gathering data and images to better understand the storm's structure, composition, and behavior. It's exciting to think about what we might learn from studying the Great Red Spot. Who knows, maybe it will help us better understand weather patterns on Earth or even other planets in the solar system. Thanks for watching, and don't forget to subscribe to our channel for more fascinating space topics like this one")
engine.runAndWait()