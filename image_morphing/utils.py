import numpy as np
import matplotlib.pyplot as plt

def imshow(img):
    if img.ndim == 2:
        plt.imshow(img.astype(np.int))
    else:
        plt.imshow(img[:, :, ::-1].astype(np.int))

def get_color(img, p):
    p = np.array(p)
    p[0] = np.clip(p[0], 0, img.shape[0] - 1)
    p[1] = np.clip(p[1], 0, img.shape[1] - 1)
    p = p.astype(np.int)
    return img[p[0], p[1]]

def save_animation(imgs, file_name='animation.mov', time=1):
    fps = len(imgs) // time
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 's')
    out = cv2.VideoWriter(
        file_name,
        int(fourcc),
        fps,
        imgs[0].shape[:2])
    for img in imgs:
        out.write(img)
