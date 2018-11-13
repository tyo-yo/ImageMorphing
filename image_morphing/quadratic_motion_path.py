from image_morphing.np import np, GPU
import cv2
from itertools import product
import os
from image_morphing.utils import *
from image_morphing.render import render_animation_256
from datetime import datetime
import time
from scipy.optimize import minimize
import os

_iter = 0
_args = ()
_prev_time = 0
_log_interbal = 0
_LOGINTERBAL = 10
_elapsed_time = 0

def log_cb(wk):
    global _iter, _args, _prev_time, _log_interbal, _elapsed_time
    now = time.time()
    _log_interbal += now - _prev_time
    _elapsed_time += now - _prev_time
    if _log_interbal > _LOGINTERBAL or _iter == 0:
        if GPU:
            ew = np.asnumpy(E(wk, *_args))
        else:
            ew = Ew(wk, *_args)
        print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(_iter, ew[0], _elapsed_time))
        _log_interbal = 0
    _prev_time = now
    _iter += 1

def optimize_w_scipy(size, w_src, v_src, img0_src=None, img1_src=None, tol=1e-1, beta=1):
    global _args, _iter, _prev_time, _elapsed_time
    w = resize_v(size, w_src)
    v = resize_v(size, v_src)
    _args = (v, beta)
    _iter = 0
    _prev_time = start = time.time()
    _elapsed_time = 0
    res = minimize(Ew, w, args=_args, callback=log_cb, tol=tol)
    end = time.time()
    w_opt = res.x
    if GPU:
        ew = np.asnumpy(Ew(w_opt, *_args))
    else:
        ew = Ew(w_opt, *_args)
    print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(
        _iter, ew[0], _elapsed_time))
    print(res.message)
    w_opt = w_opt.reshape(w.shape)
    name = '.cache/w{:03d}_{}'.format(size, datetime.now().strftime('%m%d%H%M'))
    np.save(name + '.npy', w_opt)
    if img0_src is not None and img1_src is not None:
        render_animation_256(img0_src, img1_src, v, w=w, name=name)
    return w_opt, res

def adam_w(size, w_src, v_src, img0_src=None, img1_src=None, tol=1e-3,
    beta=1, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, tol_count=3, save_dir='.cache'):
    name = os.path.join(save_dir, 'w{:03d}'.format(size))
    w = resize_v(size, w_src)
    v = resize_v(size, v_src)
    args = (v, beta)
    iter = 0
    start = prev = time.time()

    log_interbal = 0
    prev_ew = 1e10
    count=0
    m = np.zeros_like(w)
    vt = np.zeros_like(w)
    try:
        while 1:
            grads, ew = dEwdw(w, *args)

            iter += 1
            lr_t  = lr * np.sqrt(1.0 - beta2**iter) / (1.0 - beta1**iter)

            m += (1 - beta1) * (grads - m)
            vt += (1 - beta2) * (grads**2 - vt)
            w -= lr_t * m / (np.sqrt(vt) + 1e-7)

            # unbias_m = m / (1 - beta1 ** iter)
            # unbias_vt = vt / (1 - beta2 ** iter) # correct bias
            # w -= lr_t * unbias_m / (np.sqrt(unbias_vt) + 1e-7)

            log_interbal += time.time() - prev
            prev = time.time()

            if log_interbal > _LOGINTERBAL or iter == 1:
                if GPU:
                    ew_numpy = np.asnumpy(ew)
                    print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(iter, ew_numpy[0], time.time() - start))
                else:
                    print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(iter, ew[0], time.time() - start))
                log_interbal = 0
            if abs(prev_ew - ew) < tol and ew < prev_ew:
                break
            elif prev_ew < ew:
                count+= 1
                if count > tol_count:
                    break
            prev_ew = ew
    except KeyboardInterrupt:
        print('Interrupted')
        if img0_src is not None and img1_src is not None:
            render_animation_256(img0_src, img1_src, v, w=w, name=name+'.mov')
        np.save(name + '.npy', w)
        return w

    end = time.time()

    if GPU:
        ew = np.asnumpy(Ew(w, *args))
    else:
        ew = Ew(w, *args)
    print('iter {:4d}, E: {:.4f}, time: {:.1f} s'.format(
        iter, ew[0], end - start))
    print('Optimization of w finished!')
    if img0_src is not None and img1_src is not None:
        render_animation_256(img0_src, img1_src, v, w=w, name=name)
    np.save(name + '.npy', w)
    return w


def Ed(w, *args):
    v = args[0]
    if w.ndim == 3:
        size = w.shape[0]
        n_adj = (size * size) - size - 1
        adj = np.zeros([size, size, 4, 2], dtype=np.int)
        for y, x in product(range(1, size), range(1, size)):
            i = y * size + x
            adj[y, x, 0] = np.array([i, i - size - 1])
            adj[y, x, 1] = np.array([i, i - size])
            adj[y, x, 2] = np.array([i, i - size + 1])
            adj[y, x, 3] = np.array([i, i - 1])
        adj = adj[1:, 1:]
        adj = adj.reshape([-1, 2])

        v_fl = v.reshape([-1,2])
        vpi = v_fl[adj[:, 0], :]
        vpj = v_fl[adj[:, 1], :]

        X, Y = np.meshgrid(np.arange(size), np.arange(size))
        p = np.zeros([2, size, size], dtype=np.int)
        p[0] = X
        p[1] = Y
        p =  p.transpose(1,2,0)
        p = p[:, :, ::-1]
        p = p.reshape(-1, 2)
        i, j = adj[:, 0], adj[:, 1]

        d_half = p[j] - p[i] + w[p[i, 0], p[i, 1]] - w[p[j, 0], p[j, 0]] # (n_adj, 2)

        d0 = p[j] - p[i] - (v[p[j, 0], p[j, 1]] - v[p[i, 0], p[i, 1]])
        d1 = p[j] - p[i] + (v[p[j, 0], p[j, 1]] - v[p[i, 0], p[i, 1]])
        d0_norm = np.linalg.norm(d0, axis=1, keepdims=True)
        d1_norm = np.linalg.norm(d1, axis=1, keepdims=True)
        d0_hat = d0 / d0_norm
        d1_hat = d1 / d1_norm
        d_half_tilde =  (d0_norm * d1_norm) ** 0.5 * (d0_hat + d1_hat) / 2.0
        ed = ((d_half - d_half_tilde) ** 2).sum()
        return ed

    if w.ndim == 4:
        n = w.shape[0]
        size = w.shape[1]
        n_adj = (size * size) - size - 1
        adj = np.zeros([size, size, 4, 2], dtype=np.int)
        for y, x in product(range(1, size), range(1, size)):
            i = y * size + x
            adj[y, x, 0] = np.array([i, i - size - 1])
            adj[y, x, 1] = np.array([i, i - size])
            adj[y, x, 2] = np.array([i, i - size + 1])
            adj[y, x, 3] = np.array([i, i - 1])
        adj = adj[1:, 1:]
        adj = adj.reshape([-1, 2])

        v_fl = v.reshape([-1,2])
        vpi = v_fl[adj[:, 0], :]
        vpj = v_fl[adj[:, 1], :]

        X, Y = np.meshgrid(np.arange(size), np.arange(size))
        p = np.zeros([2, size, size], dtype=np.int)
        p[0] = X
        p[1] = Y
        p =  p.transpose(1,2,0)
        p = p[:, :, ::-1]
        p = p.reshape(-1, 2)
        i, j = adj[:, 0], adj[:, 1]

        d_half = p[j] - p[i] + w[:, p[i, 0], p[i, 1]] - w[:, p[j, 0], p[j, 0]] # (n_adj, 2)

        d0 = p[j] - p[i] - (v[p[j, 0], p[j, 1]] - v[p[i, 0], p[i, 1]])
        d1 = p[j] - p[i] + (v[p[j, 0], p[j, 1]] - v[p[i, 0], p[i, 1]])
        d0_norm = np.linalg.norm(d0, axis=1, keepdims=True)
        d1_norm = np.linalg.norm(d1, axis=1, keepdims=True)
        d0_hat = d0 / d0_norm
        d1_hat = d1 / d1_norm
        d_half_tilde =  (d0_norm * d1_norm) ** 0.5 * (d0_hat + d1_hat) / 2.0
        ed = ((d_half - d_half_tilde) ** 2).sum(axis=(1,2))
        return ed

def Er(w, *args):
    v = args[0]
    if w.ndim == 3:
        norm = np.linalg.norm(v, axis=2)
        mask = (norm < 1).astype(np.int)
        er = ((1 - norm) * (w ** 2).sum(axis=2) * mask).sum()
        return er
    if w.ndim == 4:
        norm = np.linalg.norm(v, axis=2)
        mask = (norm < 1).astype(np.int)
        er = ((1 - norm) * (w ** 2).sum(axis=3) * mask).sum(axis=(1,2))
        return er

def Ew(w, *args):
    v, beta = args
    w = w.reshape([-1, v.shape[0], v.shape[1], v.shape[2]])
    ed = Ed(w, v)
    er = Er(w, v)
    return ed + beta * er

def dEwdw(w, *args):
    shape = w.shape
    size = w.shape[0]
    eps = size * 1e-5
    ew = Ew(w, *args)
    v, beta = args
    w_fl = w.flatten()

    if size > 32:
        n_split = (size // 32) ** 4
    else:
        n_split = 1
    n_part = w_fl.size // n_split
    dewdw = np.zeros(w_fl.size)
    for i in range(n_split):
        delta_x = np.zeros([n_part, w_fl.size])
        delta_x[:, n_part*i: n_part*(i+1)] = np.eye(n_part, n_part) * eps
        w_part = w_fl.reshape(1, -1) + delta_x # (n_part, v.size)
        dew = Ew(w_part, v, beta) # (n_part)
        dewdw[i*n_part: (i+1)*n_part] = (dew - ew) / eps
    dewdw = dewdw.reshape(shape)
    return dewdw, ew
