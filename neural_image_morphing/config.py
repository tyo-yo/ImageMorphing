import os
from neural_best_buddies.util import util
from image_morphing.np import np, GPU
import torch
import easydict
from datetime import datetime

class Config():
    def __init__(self):
        self.GPU = GPU
        if GPU:
            gpu_ids = '0'
        else:
            gpu_ids = '-1'
        name = datetime.now().strftime('%m%d%H%M')
        self.opt = easydict.EasyDict({
            'datarootA' : './data/a.png',
            'datarootB' : './data/b.png',
            'imageSize' : 224,
            'gpu_ids' : gpu_ids,
            'tau' : 0.0,
            'border_size' : 7,
            'input_nc' : 3,
            'batchSize' : 1,
            'k_per_level' : float('inf'),
            'k_final' : 15,
            'fast' : True,
            'name' : name,
            'results_dir' : './results' ,
            'save_path' : 'None',
            'niter_decay' : 100,
            'beta1' : 0.5,
            'lr' : 0.05,
            'gamma' : 1,
            'convergence_threshold' : 0.001,
            'vmax_size' : 32,
            'lr_v' :  0.07,
            'tol_v' : 0.1,
            'tol_count_v' : 3,
            'lr_w' : 0.07,
            'tol_w' : 1e-3,
            'tol_count_w' : 3,
            'lambda_tps' : 1e-3,
            'gamma_ui' : 1e2,
            'render_logs' : True,
            'render_time' : 1,
            'render_steps' : 60
        })
        self.initialized = False


    def init_gpu(self):
        if not self.initialized:
            str_ids = self.opt.gpu_ids.split(',')
            self.opt.gpu_ids = []
            for str_id in str_ids:
                id = int(str_id)
                if id >= 0:
                    self.opt.gpu_ids.append(id)

            # set gpu ids
            if self.GPU:
                torch.cuda.set_device(self.opt.gpu_ids[0])
            self.initialized = True

    def save(self):
        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.results_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
