import unittest
import cv2
import numpy as np
from image_morphing.render import render_animation

class TestRender(unittest.TestCase):
    def test_render(self):
        img0 = cv2.imread('tests/data/nbb/original_A.png')
        img1 = cv2.imread('tests/data/nbb/original_B.png')
        v = np.load('tests/data/nbb/AtoB.npy')
        render_animation(img0, img1, v, file_name='test/data/render/animation.mov')
        return
