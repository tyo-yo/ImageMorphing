from image_morphing.np import np, GPU
import cv2
from itertools import product
from image_morphing.utils import get_color, save_animation
import os
from image_morphing.utils import describe, imshow
from image_morphing.utils import resize_img, resize_v
from datetime import datetime

def render(img0, img1, v, alpha=0.5, w =None):
    X, Y = np.meshgrid(np.arange(img0.shape[1]), np.arange(img0.shape[0]))
    Y = Y[:, :, np.newaxis]
    X = X[:, :, np.newaxis]
    q = np.concatenate([Y, X], axis=2)
    vp = get_color(v, q)
    if w is not None:
        wp = get_color(w, q)
    dampening = 0.8

    for i in range(20):
        if w is None:
            p = q - (2.0 * alpha - 1.0) * vp
        else:
            p = q - (2.0 * alpha - 1.0) * vp - 4 * alpha * (1 - alpha) * wp
            wp = get_color(w, p)
        new_vp = get_color(v, p)
        vp = dampening * new_vp + (1 - dampening) * vp
        if np.linalg.norm(vp-new_vp) < 1e-5:
            break

    c0 = get_color(img0, p - vp)
    c1 = get_color(img1, p + vp)
    morphed = (1 - alpha) * c0 + alpha * c1
    if morphed.std() > 1:
        morphed = morphed.astype(np.uint8)
    if GPU:
        morphed = np.asnumpy(morphed)
    return morphed

def render_animation(img0, img1, v, steps=30, save=True, file_name='animation.mov', time=1,  w=None):
    alpha_list = np.arange(0, 1.0 + 1e-5, 1.0/steps)
    if GPU:
        alpha_list = np.asnumpy(alpha_list)
        img0 = np.asarray(img0)
        img1 = np.asarray(img1)
        v = np.asarray(v)
        if w is not None:
            w = np.asarray(w)
    imgs = []
    print('Start Rendering')
    for alpha in alpha_list:
        # print('Rendering: {:.1f} %'.format(alpha*100))
        img = render(img0, img1, v, alpha=alpha, w=w)
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

def render_animation_256(img0_src, img1_src, v, steps=30, save=True, time=1,  w=None, name=None):
    if name is None:
        name = '.cache/anim{:03d}_{}'.format(v.shape[0], datetime.now().strftime('%m%d%H%M'))
    img0_256, img1_256 = resize_img(256, img0_src, img1_src)
    v = resize_v(256, v)
    if w is not None:
        w = resize_v(256, w)
    render_animation(img0_256, img1_256, v, w=w, file_name=name+'.mov', steps=steps, time=time)
