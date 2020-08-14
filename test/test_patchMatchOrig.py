import unittest
# from ..PatchMatchOrig import upSample_nnf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

class TestPatchMatchOrigh():

    def test_upSample_nnf(self):
        print("------------test_upSample_nnf:")

        def upSample_nnf(nnf, size=None):
            ah, aw, ac = nnf.shape

            if size is None:
                size = [ah * 2, aw * 2, ac]

            bh, bw, bc = size
            ratio_h, ratio_w = bh / ah, bw / aw
            target = np.zeros(shape=(size[0], size[1], size[2])).astype(np.int)

            for by in range(bh):
                for bx in range(bw):
                    quot_h, quot_w = int(by // ratio_h), int(bx // ratio_w)
                    # print(quot_h, quot_w)
                    rem_h, rem_w = (by - quot_h * ratio_h), (bx - quot_w * ratio_w)
                    vy, vx, vz = nnf[quot_h, quot_w]
                    vy = int(ratio_h * vy + rem_h)
                    vx = int(ratio_w * vx + rem_w)
                    target[by, bx] = [vy, vx, vz]

            return target

        img_path = '../A.png'
        img = Image.open(img_path)
        img = np.array(img)
        plt.imshow(img)
        plt.show()
        target = upSample_nnf(img)
        plt.imshow(target)
        plt.show()


if __name__ == "__main__":
    test = TestPatchMatchOrigh()
    test.test_upSample_nnf()
