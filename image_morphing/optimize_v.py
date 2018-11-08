# import numpy as np
from image_morphing.np import np, GPU
from datetime import datetime
from scipy.optimize import minimize
from image_morphing.render import render_animation, render
from image_morphing.utils import resize_all, describe
import time
from image_morphing.utils import get_v, get_color
import matplotlib.pyplot as plt
if GPU:
    from image_morphing.utils import convolve
else:
    from scipy.signal import convolve
_iter = 0
_args = ()
_prev_time = 0
_log_interbal = 0
_LOGINTERBAL = 5
_elapsed_time = 0

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
# lr = 1e-3
# size = 16
# img0, img1, v, p0, p1 = resize_all(size, img0_src, img1_src, v_src, p0_src, p1_src)
# args = (img0, img1, p0, p1, 1e-3, 1e2)
# print('eps:1e-3')
# for i in range(5):
#     v = v_opt16 + np.random.randn(size,size,2) * 1e-1
#     dedv, ev = dEdv(v, *args)
#     ori = E(v, *args)
#     e_1 = E(v - 1e-1 * dedv, *args)
#     e_2 = E(v - 1e-2 * dedv, *args)
#     e_3 = E(v - 1e-3 * dedv, *args)
#     e_4 = E(v - 1e-4 * dedv, *args)
#     e_5 = E(v - 1e-5 * dedv, *args)
#     e_6 = E(v - 1e-6 * dedv, *args)
#     print('ori: {:.2f}, 1e-1: {:.2f}, 1e-2: {:.2f}, 1e-3: {:.2f}, 1e-4: {:.2f}, 1e-5: {:.2f}, 1e-6: {:.2f},'.format(ori,e_1,e_2,e_3,e_4,e_5,e_6))

# v_opt_1, res1, half1 = optimize_v_size(size, img0_src, img1_src, v_src, p0_src,
#     p1_src, tol=1e-1, lambda_tps=1e-3, gamma_ui=1e2)
# v_sgd = sgd(size, img0_src, img1_src, v_src, p0_src, p1_src,
#     tol=1e-3, lambda_tps=1e-3, gamma_ui=1e2, lr=1e-3)
# v_adam = adam(size, img0_src, img1_src, v_src, p0_src, p1_src,
#     tol=1e-3, lambda_tps=1e-3, gamma_ui=1e2, lr=1e-2)

def dEdv(v, *args):
    eps = 1e-2
    shape = v.shape
    size = v.shape[0]
    e = E(v, *args)
    img0, img1, p0, p1, lambda_tps, gamma_ui = args
    v = v.flatten()

    if size > 32:
        n_split = (size // 32) ** 4
    else:
        n_split = 1
    n_part = v.size // n_split
    dedv = np.zeros(v.size)
    for i in range(n_split):
        delta_x = np.zeros([n_part, v.size])
        delta_x[:, n_part*i: n_part*(i+1)] = np.eye(n_part, n_part) * eps
        # delta_x = np.eye(v.size, v.size)[i*n_part: (i+1)*n_part] * eps
        v_part = v.reshape(1, -1) + delta_x # (n_part, v.size)
        dE = _E(v_part, img0, img1, p0, p1, lambda_tps, gamma_ui) # (n_part)
        dedv[i*n_part: (i+1)*n_part] = (dE - e) / eps
    dedv = dedv.reshape(shape)
    return dedv, e

def E_gradE(v, *args):
    e = E(v, *args)
    dedv = dEdv(v, *args)
    return e, dedv

def E(v, *args):
    img0, img1, p0, p1, lambda_tps, gamma_ui = args
    v = v.reshape((img0.shape[0], img0.shape[1], 2))
    e_sim = E_SIM(v, img0, img1)
    # e_sim = 1
    e_tps = E_TPS(v)
    e_ui = E_UI(v, p0, p1)
    return e_sim + lambda_tps * e_tps + gamma_ui * e_ui

def _E(v, *args):
    img0, img1, p0, p1, lambda_tps, gamma_ui = args
    v = v.reshape((-1, img0.shape[0], img0.shape[1], 2))
    e_sim = E_SIM(v, img0, img1) # about 1
    e_tps = E_TPS(v) # about 50
    e_ui = E_UI(v, p0, p1) # about 4
    return e_sim + lambda_tps * e_tps + gamma_ui * e_ui

def log_cb(vk):
    global _iter, _args, _prev_time, _log_interbal, _elapsed_time
    now = time.time()
    _log_interbal += now - _prev_time
    _elapsed_time += now - _prev_time
    if _log_interbal > _LOGINTERBAL or _iter == 0:
        if GPU:
            ev = np.asnumpy(E(vk, *_args))
        else:
            ev = E(vk, *_args)
        print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(
            _iter, ev, _elapsed_time))
        _log_interbal = 0
        img0, _, _, _, _, _ = _args
        v = vk.reshape([img0.shape[0], img0.shape[1], 2])
        np.save('.cache/v.npy', v)

    _prev_time = now
    _iter += 1

def optimize_v_size(size, img0_src, img1_src, v_src, p0_src, p1_src, method=None,
    tol=1e-1, lambda_tps=1e-3, gamma_ui=1e2):
    global _args, _iter, _prev_time, _elapsed_time
    img0, img1, v0, p0, p1 = resize_all(size, img0_src, img1_src, v_src, p0_src, p1_src)
    _args = (img0, img1, p0, p1, lambda_tps, gamma_ui)
    _iter = 0
    _prev_time = start = time.time()
    _elapsed_time = 0
    res = minimize(E, v0, args=_args, method=method, callback=log_cb, tol=tol)
    end = time.time()
    print('Time: {:.3f}'.format(end - start))
    print(res.message)
    v_opt = res.x
    v_opt = v_opt.reshape(v0.shape)
    name = '.cache/v{:03d}_{}'.format(size, datetime.now().strftime('%m%d%H%M'))
    # render_animation(img0, img1, v_opt, file_name=name+'.mov')
    img0_256, img1_256, v256, _, _ = resize_all(256, img0_src, img1_src, v_opt, p0_src, p1_src)
    render_animation(img0_256, img1_256, v256, file_name=name+'.mov')
    half = render(img0, img1, v_opt)
    np.save(name + '.npy', v_opt)
    return v_opt, res, half

# def optimize_v_bfgs(size, img0_src, img1_src, v_src, p0_src, p1_src, tol=1e-1, lambda_tps=1e-3, gamma_ui=1e2):
#     global _args, _iter, _prev_time, _elapsed_time
#     img0, img1, v0, p0, p1 = resize_all(size, img0_src, img1_src, v_src, p0_src, p1_src)
#     _args = (img0, img1, p0, p1, lambda_tps, gamma_ui)
#     _iter = 0
#     _prev_time = start = time.time()
#     _elapsed_time = 0
#     res = minimize(E_gradE, v0, args=_args, method=None, jac=True, callback=log_cb, tol=tol)
#     end = time.time()
#     print('Time: {:.3f}'.format(end - start))
#     print(res.message)
#     v_opt = res.x
#     v_opt = v_opt.reshape(v0.shape)
#     name = '.cache/v{:03d}_{}'.format(size, datetime.now().strftime('%m%d%H%M'))
#     # render_animation(img0, img1, v_opt, file_name=name+'.mov')
#     img0_256, img1_256, v256, _, _ = resize_all(256, img0_src, img1_src, v_opt, p0_src, p1_src)
#     render_animation(img0_256, img1_256, v256, file_name=name+'.mov')
#     half = render(img0, img1, v_opt)
#     np.save(name + '.npy', v_opt)
    # return v_opt, res, half

def sgd(size, img0_src, img1_src, v_src, p0_src, p1_src, tol=1e-1, lambda_tps=1e-3, gamma_ui=1e2, lr=1e-2, return_half=False):
    img0, img1, v, p0, p1 = resize_all(size, img0_src, img1_src, v_src, p0_src, p1_src)
    args = (img0, img1, p0, p1, lambda_tps, gamma_ui)
    iter = 0
    start = prev = time.time()

    log_interbal = 0
    prev_e = 1e10
    while 1:
        dedv, e = dEdv(v, *args)
        # describe(dedv)
        v = v - lr * dedv
        iter += 1
        log_interbal += time.time() - prev
        prev = time.time()

        if log_interbal > _LOGINTERBAL:
            if GPU:
                ev = np.asnumpy(E(v, *args))
            else:
                ev = E(v, *args)
            print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(
                iter, ev, time.time() - start))
            log_interbal = 0
        if abs(prev_e - e) < tol:
            break
        prev_e = e
    end = time.time()
    print('Time: {:.3f}'.format(end - start))

    name = '.cache/v{:03d}_{}'.format(size, datetime.now().strftime('%m%d%H%M'))
    img0_256, img1_256, v256, _, _ = resize_all(256, img0_src, img1_src, v, p0_src, p1_src)
    render_animation(img0_256, img1_256, v256, file_name=name+'.mov')
    np.save(name + '.npy', v)
    if return_half:
        half = render(img0, img1, v)
        return v, half
    else:
        return v

def adam(size, img0_src, img1_src, v_src, p0_src, p1_src, tol=1e-1,
    lambda_tps=1e-3, gamma_ui=1e2, lr=1e-4, return_half=False,
    beta1=0.9, beta2=0.999, eps=1e-8):
    img0, img1, v, p0, p1 = resize_all(size, img0_src, img1_src, v_src, p0_src, p1_src)
    args = (img0, img1, p0, p1, lambda_tps, gamma_ui)
    iter = 0
    start = prev = time.time()

    log_interbal = 0
    prev_e = 1e10

    m = np.zeros_like(v)
    vt = np.zeros_like(v)

    while 1:
        grads, e = dEdv(v, *args)

        iter += 1
        lr_t  = lr * np.sqrt(1.0 - beta2**iter) / (1.0 - beta1**iter)

        m += (1 - beta1) * (grads - m)
        vt += (1 - beta2) * (grads**2 - vt)

        v -= lr_t * m / (np.sqrt(vt) + 1e-7)

        #unbias_m += (1 - beta1) * (grads - m) # correct bias
        #unbisa_b += (1 - beta2) * (grads*grads - vt) # correct bias
        #params += lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)

        log_interbal += time.time() - prev
        prev = time.time()

        if log_interbal > _LOGINTERBAL or iter == 1:
            if GPU:
                ev = np.asnumpy(E(v, *args))
            else:
                ev = E(v, *args)
            print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(
                iter, ev, time.time() - start))
            log_interbal = 0
        if abs(prev_e - e) < tol:
            break
        prev_e = e
    end = time.time()

    if GPU:
        ev = np.asnumpy(E(v, *args))
    else:
        ev = E(v, *args)
    print('Optimization finished!')
    print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(
        iter, ev, end - start))

    name = '.cache/v{:03d}_{}'.format(size, datetime.now().strftime('%m%d%H%M'))
    img0_256, img1_256, v256, _, _ = resize_all(256, img0_src, img1_src, v, p0_src, p1_src)
    render_animation(img0_256, img1_256, v256, file_name=name+'.mov')
    np.save(name + '.npy', v)
    if return_half:
        half = render(img0, img1, v)
        return v, half
    else:
        return v

def SIM(x, y, C2=58.5, C3=29.3):
    '''
    # Arguments:
        x: numpy array, if n_dim == 4, shape should be (batch, h, w, RGB)
        y: numpy array, if n_dim == 4, shape should be (batch, h, w, RGB)
    # Returns:
        sim: float
    '''
    if x.std() < 1:
        x = x * 255
    if y.std() < 1:
        y = y * 255
    if x.ndim == 3 and y.ndim == 3:
        sigma_x = x.std()
        sigma_y = y.std()
        covvariance = np.cov(x.flatten(), y.flatten())
        cov_xy = covvariance[0, 1]
        var_x = covvariance[0, 0]
        var_y = covvariance[1, 1]
        c = (2 * sigma_x * sigma_y + C2) / (var_x + var_y + C2)
        s = (abs(cov_xy) + C3) / (sigma_x * sigma_y + C3)
        return c * s

    elif x.ndim == 4 and y.ndim == 4:
        x = x.reshape([x.shape[0], -1])
        y = y.reshape([y.shape[0], -1])

        mu_x = x.mean(axis=1, keepdims=True)
        mu_y = y.mean(axis=1, keepdims=True)
        sigma_x = x.std(axis=1)
        sigma_y = y.std(axis=1)
        cov_xy = ((x-mu_x)*(y-mu_y)).mean(axis=1)
        var_x = ((x-mu_x)**2).mean(axis=1)
        var_y = ((y-mu_y)**2).mean(axis=1)

        c = (2 * sigma_x * sigma_y + C2) / (var_x + var_y + C2)
        s = (abs(cov_xy) + C3) / (sigma_x * sigma_y + C3)
        return (c * s).mean()

    elif x.ndim == 5 and y.ndim == 5:
        x = x.reshape([x.shape[0], x.shape[1], -1])
        y = y.reshape([y.shape[0], y.shape[1], -1])

        mu_x = x.mean(axis=2, keepdims=True)
        mu_y = y.mean(axis=2, keepdims=True)
        sigma_x = x.std(axis=2)
        sigma_y = y.std(axis=2)
        cov_xy = ((x-mu_x)*(y-mu_y)).mean(axis=2)
        var_x = ((x-mu_x)**2).mean(axis=2)
        var_y = ((y-mu_y)**2).mean(axis=2)

        c = (2 * sigma_x * sigma_y + C2) / (var_x + var_y + C2)
        s = (abs(cov_xy) + C3) / (sigma_x * sigma_y + C3)
        return (c * s).mean(axis=1)

def E_SIM(v, img0, img1, p=None):
    '''
    if p is None, compute and return sum of E_SIM(p)
    '''
    if v.ndim == 2 and v.shape[0] == v.shape[1]:
        Y, X = np.meshgrid(np.arange(-2,3), np.arange(-2,3))
        Y = Y[:, :, np.newaxis]
        X = X[:, :, np.newaxis]
        grid = np.concatenate([Y, X], axis=2)
        grid = grid[np.newaxis, :, :, :] # (1, 5, 5, 2)

        X, Y = np.meshgrid(range(2, img0.shape[0]-2), range(2, img0.shape[1] -2))
        Y = Y[:, :, np.newaxis]
        X = X[:, :, np.newaxis]
        p_all = np.concatenate([Y, X], axis=2)
        p_all = p_all.reshape([-1, 2])
        p_all = p_all[:, np.newaxis, np.newaxis, :] # (num of p, 1, 1, 2)

        grids = grid + p_all # (1, num of p, 5, 5, 2)

        size = int((v.size / 4) ** 0.25)
        v = v.reshape(-1, size, size, 2) # (N, size, size, 2)
        grids_v = get_color(v, grids) # (N, num of p, 5, 5, 2)

        grids_img0 = get_color(img0, grids - grids_v)
        grids_img1 = get_color(img1, grids + grids_v) # (N, num of p, 5, 5, 3)
        sim = SIM(grids_img0, grids_img1)
        return -1 * sim

    elif p is not None:
        Y, X = np.meshgrid(np.arange(-2,3), np.arange(-2,3))
        Y = Y[:, :, np.newaxis]
        X = X[:, :, np.newaxis]
        grid = np.concatenate([Y, X], axis=2)
        grid += p
        # now grid is 5*5 points around p
        grid_v = get_v(v, grid)
        grid_img0 = get_color(img0, grid - grid_v)
        grid_img1 = get_color(img1, grid + grid_v)
        sim = SIM(grid_img0, grid_img1)
        return -1 * sim
    else:
        Y, X = np.meshgrid(np.arange(-2,3), np.arange(-2,3))
        Y = Y[:, :, np.newaxis]
        X = X[:, :, np.newaxis]
        grid = np.concatenate([Y, X], axis=2)
        grid = grid[np.newaxis, :, :, :] # (1, 5, 5, 2)

        X, Y = np.meshgrid(np.arange(2, img0.shape[0]-2), np.arange(2, img0.shape[1] -2))
        Y = Y[:, :, np.newaxis]
        X = X[:, :, np.newaxis]
        p_all = np.concatenate([Y, X], axis=2)
        p_all = p_all.reshape([-1, 2])
        p_all = p_all[:, np.newaxis, np.newaxis, :] # (num of p, 1, 1, 2)

        grids = grid + p_all # (num of p, 5, 5, 2)
        grids_v = get_v(v, grids)
        grids_img0 = get_color(img0, grids - grids_v)
        grids_img1 = get_color(img1, grids + grids_v)
        sim = SIM(grids_img0, grids_img1)
        return -1 * sim


def E_TPS(v):
    '''
    # Arguments:
        v: np.array, shape = (height, width, 2)
    '''
    if v.ndim == 3:
        kernel_dx2 = np.array([
            [0, 0, 0],
            [1, -2, 1],
            [0, 0, 0]])
        kernel_dx2 = kernel_dx2.reshape([3, 3, 1])

        kernel_dxdy = np.array([
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]]) / 4
        kernel_dxdy = kernel_dxdy.reshape([3, 3, 1])

        kernel_dy2 = np.array([
        [0, 1, 0],
        [0, -2, 0],
        [0, 1, 0]])
        kernel_dy2 = kernel_dy2.reshape([3, 3, 1])

        dx2 = convolve(v, kernel_dx2, mode='same')
        dxdy = convolve(v, kernel_dxdy, mode='same')
        dy2 = convolve(v, kernel_dy2, mode='same')

        tps = (dx2**2 + 2 * dxdy**2 + dy2**2)
        return tps.sum()
    elif v.ndim == 4:
        kernel_dx2 = np.array([
            [0, 0, 0],
            [1, -2, 1],
            [0, 0, 0]])
        kernel_dx2 = kernel_dx2.reshape([1, 3, 3, 1])

        kernel_dxdy = np.array([
            [1, 0, -1],
            [0, 0, 0],
            [-1, 0, 1]]) / 4
        kernel_dxdy = kernel_dxdy.reshape([1, 3, 3, 1])

        kernel_dy2 = np.array([
        [0, 1, 0],
        [0, -2, 0],
        [0, 1, 0]])
        kernel_dy2 = kernel_dy2.reshape([1, 3, 3, 1])
        dx2 = convolve(v, kernel_dx2, mode='same')
        dxdy = convolve(v, kernel_dxdy, mode='same')
        dy2 = convolve(v, kernel_dy2, mode='same')

        tps = (dx2**2 + 2 * dxdy**2 + dy2**2)
        return tps.sum(axis=(1,2,3))


def E_UI(v, p0, p1):
    if v.ndim == 3:
        p = (p1 + p0) / 2
        p = p[:, np.newaxis, :]

        v_true = (p1 - p0) / 2

        tmp = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [1., 1.]])
        tmp = tmp[np.newaxis, :, :]
        p_around = tmp + np.floor(p)

        tmp = np.array([
            [1., 1.],
            [0., 0.]])
        tmp = tmp[np.newaxis, :, :] # (1, 2, 2)
        p_dif = p - np.floor(p) # (n, 1, 2)
        p_dif = np.concatenate([-p_dif, p_dif], axis=1) # (n, 2, 2)
        p_dif = p_dif + tmp # (10, 2, 2)
        # if original p[0] is [2.2, 5.3]
        # p_dif[0] is [[0.8, 0.7],[0.2, 0.3]]

        p_dif_x = p_dif[:, :, 0] # p_dif_x[0] is [0.8, 0.2]
        p_dif_y = p_dif[:, :, 1] # p_dif_x[1] is [0.7, 0.3]
        p_dif_x = p_dif_x[:, np.newaxis, :]
        p_dif_y = p_dif_y[:, :, np.newaxis]
        b = np.matmul(p_dif_y, p_dif_x) # b[0] is [[0.56, 0.14], [0.24, 0.06]]
        b = b.reshape(-1, 4, 1)

        v_around = get_v(v, p_around)

        v_true = v_true[:, np.newaxis, :]
        ui = b * (v_true - v_around) ** 2

        return ui.mean(axis=1).sum()

    elif v.ndim == 4:
        p = (p1 + p0) / 2
        p = p[:, np.newaxis, :] # (n , 1, 2)

        v_true = (p1 - p0) / 2

        tmp = np.array([
            [0., 0.],
            [1., 0.],
            [0., 1.],
            [1., 1.]])
        tmp = tmp[np.newaxis, :, :] # (1, 4, 2)
        p_around = tmp + np.floor(p) # (n, 4, 2)

        tmp = np.array([
            [1., 1.],
            [0., 0.]])
        tmp = tmp[np.newaxis, :, :] # (1, 2, 2)
        p_dif = p - np.floor(p) # (n, 1, 2)
        p_dif = np.concatenate([-p_dif, p_dif], axis=1) # (n, 2, 2)
        p_dif = p_dif + tmp # (n, 2, 2)
        # if original p[0] is [2.2, 5.3]
        # p_dif[0] is [[0.8, 0.7],[0.2, 0.3]]

        p_dif_x = p_dif[:, :, 0] # p_dif_x[0] is [0.8, 0.2]
        p_dif_y = p_dif[:, :, 1] # p_dif_x[1] is [0.7, 0.3]
        p_dif_x = p_dif_x[:, np.newaxis, :]
        p_dif_y = p_dif_y[:, :, np.newaxis]
        b = np.matmul(p_dif_y, p_dif_x) # b[0] is [[0.56, 0.14], [0.24, 0.06]]
        b = b.reshape(1, -1, 4, 1) # (1, n, 4, 1)
        p_around.shape
        v.shape
        v_around = get_v(v, p_around) # (N, n, 4, 2)

        v_true = v_true[np.newaxis, :, np.newaxis, :] # (1, n, 1, 2)
        ui = b * (v_true - v_around) ** 2 # (N, n, 4, 2)

        return ui.mean(axis=2).sum(axis=(1, 2))

def plot_sim(x0, y0):
    plt.figure(figsize=(10,10))
    for i in range(16):
        idx = np.random.randint(x0.shape[0])
        im0 = x0[idx]
        im1 = y0[idx]
        plt.subplot(4, 4, i+1)
        plt.grid(b=False)
        img = np.concatenate([im0, im1], axis=1)
        sim = SIM(im0, im1)
        plt.imshow(img[:,:,::-1])
        plt.title('SIM: {:.2f}'.format(sim))
    plt.show()
