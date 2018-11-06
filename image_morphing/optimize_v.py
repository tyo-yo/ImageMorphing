import numpy as np
from image_morphing.utils import get_v, get_color
from scipy.signal import convolve
from datetime import datetime
from scipy.optimize import minimize
from image_morphing.render import render_animation, render
from image_morphing.utils import resize_all, describe
import time
from numba import jit

_iter = 0
_args = ()
_prev_time = 0
_log_interbal = 0
_LOGINTERBAL = 1
_elapsed_time = 0


def E(v, *args):
    img0, img1, p0, p1, lambda_tps, gamma_ui = args
    v = v.reshape((img0.shape[0], img0.shape[1], 2))
    e_sim = E_SIM(v, img0, img1)
    e_tps = E_TPS(v)
    e_ui = E_UI(v, p0, p1)
    return e_sim + lambda_tps * e_tps + gamma_ui * e_ui

def log_cb(vk):
    global _iter, _args, _prev_time, _log_interbal, _elapsed_time
    now = time.time()
    _log_interbal += now - _prev_time
    _elapsed_time += now - _prev_time
    if _log_interbal > _LOGINTERBAL or _iter == 0:
    # if _iter % 1 == 0:
        print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(
            _iter, E(vk, *_args), _elapsed_time))
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
    name = '.cache/v{:03d}_{}'.format(size, datetime.now().strftime('%H%M'))
    # render_animation(img0, img1, v_opt, file_name=name+'.mov')
    img0_256, img1_256, v256, _, _ = resize_all(256, img0_src, img1_src, v_opt, p0_src, p1_src)
    render_animation(img0_256, img1_256, v256, file_name=name+'.mov')
    half = render(img0, img1, v_opt)
    np.save(name + '.npy', v_opt)
    return v_opt, res, half



def SIM(x, y, C2=58.5, C3=29.3):
    '''
    # Arguments:
        x: numpy array, if n_dim == 4, shape should be (batch, h, w, RGB)
        y: numpy array, if n_dim == 4, shape should be (batch, h, w, RGB)
    # Returns:
        sim: float
    '''
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


def E_SIM(v, img0, img1, p=None):
    '''
    if p is None, compute and return sum of E_SIM(p)
    '''
    if p is not None:
        Y, X = np.meshgrid(range(-2,3), range(-2,3))
        Y = Y[:, :, np.newaxis]
        X = X[:, :, np.newaxis]
        grid = np.concatenate([Y, X], axis=2)
        grid += p
        # now grid is 5*5 points around p
        grid_v = get_v(v, grid)
        grid_img0 = get_color(img0, grid - grid_v)
        grid_img1 = get_color(img1, grid + grid_v)
        sim = SIM(grid_img0, grid_img1)
        return sim
    else:
        Y, X = np.meshgrid(range(-2,3), range(-2,3))
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

        grids = grid + p_all # (num of p, 5, 5, 2)
        grids_v = get_v(v, grids)
        grids_img0 = get_color(img0, grids - grids_v)
        grids_img1 = get_color(img1, grids + grids_v)
        sim = SIM(grids_img0, grids_img1)
        return sim


def E_TPS(v):
    '''
    # Arguments:
        v: np.array, shape = (height, width, 2)
    '''
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


def E_UI(v, p0, p1):
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
