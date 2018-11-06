import numpy as np
import matplotlib.pyplot as plt
import cv2

def imshow(img):
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
    elif p.ndim == 3:
        p[:, :, 0] = np.clip(p[:, :, 0], 0, img.shape[0] - 1)
        p[:, :, 1] = np.clip(p[:, :, 1], 0, img.shape[1] - 1)
        p = p.astype(np.int)
        return img[p[:, :, 0], p[:, :, 1]]
    elif p.ndim == 4:
        p[:, :, :, 0] = np.clip(p[:, :, :, 0], 0, img.shape[0] - 1)
        p[:, :, :, 1] = np.clip(p[:, :, :, 1], 0, img.shape[1] - 1)
        p = p.astype(np.int)
        return img[p[:, :, :, 0], p[:, :, :, 1]]

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
        out.write(img)

def describe(x):
    print('*** Numpy array info ***')
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
        return np.array(points).astype(np.int)

def resize_all(size, img0, img1, v, p0, p1):
    img0_low = cv2.resize(img0, (size, size))
    img1_low = cv2.resize(img1, (size, size))
    if img0_low.std() > 1:
        img0_low = img0_low / 255.0
    if img1_low.std() > 1:
        img1_low = img1_low / 255.0

    vx = v[:, :, 0]
    vy = v[:, :, 1]
    vx_low = cv2.resize(vx, (size, size))
    vy_low = cv2.resize(vy, (size, size))
    vx_low = vx_low[:, :, np.newaxis]
    vy_low = vy_low[:, :, np.newaxis]
    v_low = np.concatenate([vx_low, vy_low], axis=2)
    v_low = v_low / v.shape[0] * size

    p0_low = p0 / img0.shape[0] * size
    p1_low = p1 / img1.shape[0] * size

    return img0_low, img1_low, v_low, p0_low, p1_low


def plot_correspondances(img0, img1, p0, p1):
    img0_point = img0 // 3
    img1_point = img1 // 3
    img0_point[p0[:,0].astype(int) ,p0[:,1].astype(int), 0:2] = 255
    img1_point[p1[:,0].astype(int) ,p1[:,1].astype(int), 0:2] = 255
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img0_point[:, :, ::-1])
    plt.subplot(1, 2, 2)
    plt.imshow(img1_point[:, :, ::-1])
    plt.show()
