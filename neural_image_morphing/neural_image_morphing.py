import os
from neural_best_buddies.models import vgg19_model
from neural_best_buddies.algorithms import neural_best_buddies as NBBs
from neural_best_buddies.util import util
from neural_image_morphing.config import Config
from image_morphing.morpher import image_morphing
from image_morphing.np import np, GPU
from image_morphing.utils import resize_v
from image_morphing.render import render_animation
import numpy
import cv2

class NeuralImageMorphing():
    def __init__(self, cfg):
        self.cfg = cfg
        self.opt = self.cfg.opt

    def run_nbb(self):
        self.cfg.init_gpu()
        self.vgg19 = vgg19_model.define_Vgg19(self.opt)
        self.save_dir = os.path.join(self.opt.results_dir, self.opt.name)
        nbbs = NBBs.sparse_semantic_correspondence(
            self.vgg19, self.opt.gpu_ids, self.opt.tau, self.opt.border_size,
            self.save_dir, self.opt.k_per_level, self.opt.k_final, self.opt.fast)
        A = util.read_image(self.opt.datarootA, self.opt.imageSize)
        B = util.read_image(self.opt.datarootB, self.opt.imageSize)

        points = nbbs.run(A, B)

    def run_morphing(self):
        opt = self.opt
        self.save_dir = os.path.join(self.opt.results_dir, self.opt.name)
        p0_path = os.path.join(self.save_dir,
            'correspondence_A_top_{}.txt'.format(opt.k_final))
        p1_path = os.path.join(self.save_dir,
            'correspondence_Bt_top_{}.txt'.format(opt.k_final))
        mov_name = self.opt.name + '.mov'
        image_morphing(opt.datarootA, opt.datarootB, p0_path, p1_path,
            render_name=mov_name, vmax_size=opt.vmax_size,
            lr_v=opt.lr_v, tol_v=opt.tol_v, lr_w=opt.lr_w, tol_w=opt.tol_w,
            lambda_tps=opt.lambda_tps, gamma_ui=opt.gamma_ui, render=opt.render_logs,
            render_steps=opt.render_steps, render_time=opt.render_time, save_dir=self.save_dir
        )

    def run(self):
        self.run_nbb()
        self.run_morphing()

    def render(self, file_name='animation.mov'):
        sizes = numpy.arange(8, self.opt.vmax_size + 1, 8)
        save_dir = os.path.join(self.opt.results_dir, self.opt.name)
        for size in sizes:
            name = os.path.join(save_dir, 'v{:03d}.npy'.format(size))
            if os.path.exists(name):
                v = np.load(name)
            name = os.path.join(save_dir, 'w{:03d}.npy'.format(size))
            if os.path.exists(name):
                w = np.load(name)

        if v is None or w is None:
            print('vXXX.npy or wXXX.npy does not exist.')
            return
        else:
            img0_src = cv2.imread(self.opt.datarootA)
            img1_src = cv2.imread(self.opt.datarootB)
            # v_final = resize_v(v=v, size=img0_src.shape[0], size_x=img0_src.shape[0])
            # w_final = resize_v(v=w, size=img0_src.shape[0], size_x=img0_src.shape[0])
            v_final = resize_v(v=v, size=img0_src.shape[1], size_x=img0_src.shape[0])
            w_final = resize_v(v=w, size=img0_src.shape[1], size_x=img0_src.shape[0])
            img1 = cv2.resize(img1_src, (img0_src.shape[1], img0_src.shape[0]))

            save_path = os.path.join(save_dir, file_name)
            render_animation(img0_src, img1, v_final, w=w_final,
                steps=self.opt.render_steps, time=self.opt.render_time,
                file_name=save_path)
        return
