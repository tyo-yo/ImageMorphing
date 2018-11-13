import cv2
from image_morphing.np import np, GPU
from image_morphing.utils import load_points, resize_v
from image_morphing.render import render_animation
from image_morphing.optimize_v import adam
from image_morphing.quadratic_motion_path import adam_w
import os

def image_morphing(img0_path, img1_path, p0_path, p1_path, vmax_size=32, render_name='animation.mov',
        lr_v=7e-2, tol_v=1e-1, lr_w=7e-2, tol_w=1e-3, lambda_tps=1e-3, gamma_ui=1e2,
        tol_count_v=20, tol_count_w=3, render=False, render_steps=60, render_time=1,
        save_dir='.cache'):
    img0_src = cv2.imread(img0_path)
    img1_src = cv2.imread(img1_path)

    p0_src = load_points(p0_path)
    p1_src = load_points(p1_path)

    size = 8
    v = np.random.randn(size, size, 2)
    w = np.random.randn(size, size, 2)

    # sizes = np.arange(8, vmax_size + 1, 8)
    sizes = 2 ** np.arange(3, 10)
    sizes = sizes[sizes <= vmax_size]
    if GPU:
        sizes = np.asnumpy(sizes)
    for size in sizes:
        print('\nOptimization size {:3d} start.'.format(size))
        name = os.path.join(save_dir, 'v{:03d}'.format(size))
        if os.path.exists(name):
            v = np.load(name)
        else:
            print('Optimization of v start.')
            v = adam(size, img0_src, img1_src, v, p0_src, p1_src, lr=lr_v, tol=tol_v,
                render=render, tol_count=tol_count_v, lambda_tps=lambda_tps,
                gamma_ui=gamma_ui, save_dir=save_dir)
        name = os.path.join(save_dir, 'w{:03d}'.format(size))
        if os.path.exists(name):
            w = np.load(w)
        else:
            print('Optimization of w start.')
            w = adam_w(size, w, v, lr=lr_w, tol=tol_w, tol_count=tol_count_w,
                save_dir=save_dir)
    v_final = resize_v(v=v, size=img0_src.shape[0], size_x=img0_src.shape[1])
    w_final = resize_v(v=w, size=img0_src.shape[0], size_x=img0_src.shape[1])
    img1 = cv2.resize(img1_src, (img0_src.shape[0], img0_src.shape[1]))

    render_path = os.path.join(save_dir, render_name)
    render_animation(img0_src, img1, v_final, w=w_final, steps=render_steps,
        time=render_time, file_name=render_path)
