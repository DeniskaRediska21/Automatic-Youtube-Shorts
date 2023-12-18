

import keras_cv
import time
from tensorflow.keras import mixed_precision
import tensorflow as tf
import matplotlib.pyplot as plt
# mixed_precision.set_global_policy("mixed_float16")
supermodel = keras_cv.models.StableDiffusion(jit_compile=True, img_width=360, img_height=640)
start = time.time()
prompt = "a photograph of an astronaut riding a horse"
images = supermodel.text_to_image(prompt, batch_size=1, num_steps=25)
end = time.time()
print('Coldstart time', end - start)

def plot_images(images):
    plt.figure(figsize=(20, 20))
    for i in range(len(images)):
        ax = plt.subplot(1, len(images), i + 1)
        plt.imshow(images[i])
        plt.axis("off")


plot_images(images)