import cv2
from image_morphing.np import np, GPU
from image_morphing.utils import load_points, resize_v
from image_morphing.render import render_animation
from image_morphing.optimize_v import adam
from image_morphing.quadratic_motion_path import adam_w

def image_morphing(img0_path, img1_path, p0_path, p1_path, vmax_size=32, name='animation.mov',
    lr_v=7e-2, tol_v=1e-1, lr_w=7e-2, tol_w=1e-3, lambda_tps=1e-3, gamma_ui=1e2):
    img0_src = cv2.imread(img0_path)
    img1_src = cv2.imread(img1_path)

    p0_src = load_points(p0_path)
    p1_src = load_points(p1_path)

    size = 8
    v = np.random.randn(size, size, 2)
    w = np.random.randn(size, size, 2) * size / 3 + size / 2

    sizes = np.arange(8, vmax_size + 1, 8)
    for size in sizes:
        print('\nOptimization size {:3d} start.'.format(size))
        print('Optimization of v start.')
        v = adam(size, img0_src, img1_src, v, p0_src, p1_src, lr=lr_v, tol=tol_v, render=False)
        print('Optimization of w start.')
        w = adam_w(size, w, v, lr=7e-2, tol=1e-3)
    v_final = resize_v(v=v, size=img0_src.shape[0], size_x=img0_src.shape[1])
    w_final = resize_v(v=w, size=img0_src.shape[0], size_x=img0_src.shape[1])
    v_final = resize_v(v=v, size=img0_src.shape[0])
    w_final = resize_v(v=w, size=img0_src.shape[0])
    render_animation(img0_src, img1_src, v_final, w=w_final, steps=60, file_name=name)
