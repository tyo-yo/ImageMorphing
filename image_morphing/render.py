import numpy as np
import cv2
from itertools import product
from image_morphing.utils import get_color, save_animation
import os
from image_morphing.utils import describe, imshow
img0 = cv2.imread('tests/data/nbb/original_A.png')
img1 = cv2.imread('tests/data/nbb/original_b.png')
v = np.load('tests/data/nbb/AtoB.npy').astype(np.float)

def render(img0, img1, v, alpha=0.5):
    X, Y = np.meshgrid(range(img0.shape[0]), range(img0.shape[1]))
    Y = Y[:, :, np.newaxis]
    X = X[:, :, np.newaxis]
    q = np.concatenate([Y, X], axis=2)
    vp = get_color(v, q)
    dampening = 0.8

    for i in range(20):
        p = q - (2.0 * alpha - 1.0) * vp
        new_vp = get_color(v, p)
        vp = dampening * new_vp + (1 - dampening) * vp
        if np.linalg.norm(vp-new_vp) < 1e-5:
            break

    c0 = get_color(img0, p - vp)
    c1 = get_color(img1, p + vp)
    morphed = (1 - alpha) * c0 + alpha * c1
    if morphed.std() > 1:
        morphed = morphed.astype(np.uint8)
    return morphed

def _render_deprecated(img0, img1, v, alpha=0.5):
    morphed = np.zeros_like(img0)
    for y, x in product(range(img0.shape[0]), range(img0.shape[1])):
        q = (y, x)
        vp = get_vp(v, q)
        dampening = 0.8
        for i in range(20):
            p = q - (2.0 * alpha - 1.0) * vp
            new_vp = get_vp(v, p)
            if np.all(vp == new_vp):
                break
            vp = dampening * new_vp + (1 - dampening) * vp

        c0 = get_color(img0, p - vp)
        c1 = get_color(img1, p + vp)
        morphed[q] = (1 - alpha) * c0 + alpha * c1
    return morphed

def render_animation(img0, img1, v, steps=30, save=True, file_name='animation.mov', time=1):
    alpha_list = np.arange(0, 1.0 + 1e-5, 1.0/steps)
    imgs = []
    print('Start Rendering')
    for alpha in alpha_list:
        # print('Rendering: {:.1f} %'.format(alpha*100))
        img = render(img0, img1, v, alpha)
        if img.std() < 1:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
        imgs.append(img)
    if save:
        if os.path.exists(file_name):
            os.remove(file_name)
        save_animation(imgs, file_name=file_name, time=time)
    print('Rendering finished!')
    return imgs

def get_vp(v, p):
    p = np.array(p)
    p[0] = np.clip(p[0], 0, v.shape[0] - 1)
    p[1] = np.clip(p[1], 0, v.shape[1] - 1)
    p = p.astype(np.int)
    return v[p[0], p[1]]
