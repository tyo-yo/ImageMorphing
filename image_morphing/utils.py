# import numpy as np
from image_morphing.np import np, GPU
import matplotlib.pyplot as plt
import cv2

# %load_ext autoreload
# %autoreload 2
# from image_morphing.utils import *
# img0_src = cv2.imread('tests/data/nbb/original_A.png')
# img1_src = cv2.imread('tests/data/nbb/original_b.png')
# v_src = np.load('tests/data/nbb/AtoB.npy').astype(np.float)
# p0_src = load_points('tests/data/nbb/correspondence_A_top_100.txt')
# p1_src = load_points('tests/data/nbb/correspondence_Bt_top_100.txt')
# v_opt8 = np.load('.cache/v008_p100.npy')
# v_opt16 = np.load('.cache/v016_p1000.npy')
# v_opt32 = np.load('.cache/v032_p1000.npy')
# v_opt64 = np.load('.cache/v064_p1000.npy')
#
# size = 16
# lr = 1e-3
#
# img0, img1, v, p0, p1 = resize_all(size, img0_src, img1_src, v_src, p0_src, p1_src)
# args = (img0, img1, p0, p1, 1e-3, 1e2)
#
# X = img0
# kernel = np.array([
#     [0, 0, 0],
#     [1, -2, 1],
#     [0, 0, 0]])
# kernel = kernel.reshape([3, 3, 1])
# x = x[np.newaxis, :, : , :]
# x.shape

def imshow(img):
    if GPU:
        img = np.asnumpy(img)
    if img.std() < 1:
        img = img * 255
    if img.ndim == 2:
        plt.imshow(img.astype(np.uint8))
    else:
        plt.imshow(img[:, :, ::-1].astype(np.uint8))

def get_color(img, p):
    p = np.array(p)
    if p.ndim == 1:
        p[0] = np.clip(p[0], 0, img.shape[0] - 1)
        p[1] = np.clip(p[1], 0, img.shape[1] - 1)
        p = p.astype(np.int)
        return img[p[0], p[1]]
    elif p.ndim == 2:
        p[:, 0] = np.clip(p[:, 0], 0, img.shape[0] - 1)
        p[:, 1] = np.clip(p[:, 1], 0, img.shape[1] - 1)
        p = p.astype(np.int)
        return img[p[:, 0], p[:, 1]]
    elif p.ndim == 3 and img.ndim == 4:
        p[:, :, 0] = np.clip(p[:, :, 0], 0, img.shape[1] - 1)
        p[:, :, 1] = np.clip(p[:, :, 1], 0, img.shape[2] - 1)
        p = p.astype(np.int)
        return img[:, p[:, :, 0], p[:, :, 1]]
    elif p.ndim == 3:
        p[:, :, 0] = np.clip(p[:, :, 0], 0, img.shape[0] - 1)
        p[:, :, 1] = np.clip(p[:, :, 1], 0, img.shape[1] - 1)
        p = p.astype(np.int)
        return img[p[:, :, 0], p[:, :, 1]]
    elif p.ndim == 4 and img.ndim == 4:
        p[:, :, :, 0] = np.clip(p[:, :, :, 0], 0, img.shape[1] - 1)
        p[:, :, :, 1] = np.clip(p[:, :, :, 1], 0, img.shape[2] - 1)
        p = p.astype(np.int)
        return img[:, p[:, :, :, 0], p[:, :, :, 1]]
    elif p.ndim == 4:
        p[:, :, :, 0] = np.clip(p[:, :, :, 0], 0, img.shape[0] - 1)
        p[:, :, :, 1] = np.clip(p[:, :, :, 1], 0, img.shape[1] - 1)
        p = p.astype(np.int)
        return img[p[:, :, :, 0], p[:, :, :, 1]]
    elif p.ndim == 5:
        p[:, :, :, :, 0] = np.clip(p[:, :, :, :, 0], 0, img.shape[0] - 1)
        p[:, :, :, :, 1] = np.clip(p[:, :, :, :, 1], 0, img.shape[1] - 1)
        p = p.astype(np.int)
        return img[p[:, :, :, :, 0], p[:, :, :, :, 1]]

def get_v(v, p):
    return get_color(v, p)

def save_animation(imgs, file_name='animation.mov', time=1):
    fps = len(imgs) // time
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 's')
    out = cv2.VideoWriter(
        file_name,
        int(fourcc),
        fps,
        imgs[0].shape[:2])
    for img in imgs:
        if GPU:
            img = np.asnumpy(img)
        out.write(img)

def describe(x):
    print('*** Numpy array info ***')
    print('type : ' + str(type(x)))
    print('shape : ' + str(x.shape))
    print('mean : {:.4f}'.format(x.mean()))
    print('std : {:.4f}'.format(x.std()))
    print('min : {:.4f}, argmin : {}'.format(
        x.min(),
        str(np.unravel_index(x.argmin(), x.shape))))
    print('max : {:.4f}, argmax : {}'.format(
        x.max(),
        str(np.unravel_index(x.argmax(), x.shape))))

def load_points(path):
    with open(path, 'r', encoding='utf-8') as f:
        s = f.read()
        arr = s.splitlines()
        points = [p.split(', ') for p in arr]
        points = [[int(p) for p in l] for l in points]
        return np.array(points).astype(np.int)

def resize_img(size, img0, img1):
    if GPU:
        img0 = np.asnumpy(img0)
        img1 = np.asnumpy(img1)
    img0_low = cv2.resize(img0, (size, size))
    img1_low = cv2.resize(img1, (size, size))
    if GPU:
        img0_low = np.asarray(img0_low)
        img1_low = np.asarray(img1_low)

    if img0_low.std() > 1:
        img0_low = img0_low / 255.0
    if img1_low.std() > 1:
        img1_low = img1_low / 255.0
    return img0_low, img1_low

def resize_v(size, v):
    if GPU:
        v = np.asnumpy(v)
    vx = v[:, :, 0]
    vy = v[:, :, 1]
    vx_low = cv2.resize(vx, (size, size))
    vy_low = cv2.resize(vy, (size, size))
    if GPU:
        vx_low = np.asarray(vx_low)
        vy_low = np.asarray(vy_low)
    vx_low = vx_low[:, :, np.newaxis]
    vy_low = vy_low[:, :, np.newaxis]
    v_low = np.concatenate([vx_low, vy_low], axis=2)
    v_low = v_low / v.shape[0] * size
    return v_low

def resize_p(size, original_size, p0, p1):
    p0_low = p0 / original_size * size
    p1_low = p1 / original_size * size
    return p0_low, p1_low

def resize_all(size, img0, img1, v, p0, p1):
    img0_low, img1_low = resize_img(size, img0, img1)
    v_low = resize_v(size, v)
    p0_low, p1_low = resize_p(size, img0.shape[0], p0, p1)

    return img0_low, img1_low, v_low, p0_low, p1_low


def plot_correspondances(img0, img1, p0, p1):
    if GPU:
        img0, img1 = np.asnumpy(img0), np.asnumpy(img1)
        p0, p1 = np.asnumpy(p0), np.asnumpy(p1)
    img0_point = img0 // 3
    img1_point = img1 // 3
    img0_point[p0[:,0].astype(np.int) ,p0[:,1].astype(np.int), 0:2] = 255
    img1_point[p1[:,0].astype(np.int) ,p1[:,1].astype(np.int), 0:2] = 255
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img0_point[:, :, ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(img1_point[:, :, ::-1])
    plt.show()

def convolve(X, kernel, mode='same'):
    """
    Parameters
    ----------
    X : (データ数, 高さ, 幅, チャンネル)の4次元配列からなる入力データ

    Returns
    -------
    col : 2次元配列
    """
    if X.ndim == 3:
        H, W, C = X.shape
        pad, stride = 1, 1
        filter_h, filter_w = 3, 3
        kernel = kernel.reshape([3, 3])
        out_h = (H + 2*pad - filter_h)//stride + 1 # out_h == h
        out_w = (W + 2*pad - filter_w)//stride + 1 # out_w == w

        img = np.pad(X, [(pad, pad), (pad, pad), (0,0)], 'edge')
        extended = np.zeros((filter_h, filter_w, out_h, out_w, C))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                extended[y, x, :, :, :] = img[y:y_max:stride, x:x_max:stride, :]

        return np.einsum('ij, ijhwc->hwc', kernel, extended)

    elif X.ndim == 4:
        N, H, W, C = X.shape
        pad, stride = 1, 1
        filter_h, filter_w = 3, 3
        kernel = kernel.reshape([3, 3])
        out_h = (H + 2*pad - filter_h)//stride + 1 # out_h == h
        out_w = (W + 2*pad - filter_w)//stride + 1 # out_w == w

        img = np.pad(X, [(0,0), (pad, pad), (pad, pad), (0,0)], 'edge')
        extended = np.zeros((N, filter_h, filter_w, out_h, out_w, C))

        for y in range(filter_h):
            y_max = y + stride*out_h
            for x in range(filter_w):
                x_max = x + stride*out_w
                extended[:, y, x, :, :, :] = img[:, y:y_max:stride, x:x_max:stride, :]

        return np.einsum('ij, nijhwc->nhwc', kernel, extended)
