import numpy as np
import cv2 as cv

def zoom_at(img, zoom, coord=None):
    """
    Simple image zooming without boundary checking.
    Centered at "coord", if given, else the image center.

    img: numpy.ndarray of shape (h,w,:)
    zoom: float
    coord: (float, float)
    """
    # Translate to zoomed coordinates
    h, w, _ = [ zoom * i for i in np.shape(img) ]
    
    if coord is None: cx, cy = w/2, h/2
    else: cx, cy = [ zoom*c for c in coord ]

    img = cv.resize( img, (0, 0), fx=zoom, fy=zoom)
    img = img[ int(round(cy - h/zoom * .5)) : int(round(cy + h/zoom * .5)),
            int(round(cx - w/zoom * .5)) : int(round(cx + w/zoom * .5)),
            : ]
    
    return img

def plot_images(images):
	import matplotlib.pyplot as plt
	plt.figure(figsize=(20, 20))
	for i in range(len(images)):
		plt.subplot(1, len(images), i + 1)
		plt.imshow(images[i])
		plt.axis("off")
