import unittest
from project.layers.convolution import Convolution
import torch
import numpy as np


class ConvolutionTestCase(unittest.TestCase):
    def setUp(self):
        x_shape = (10, 3, 16, 16)
        self.x = torch.randn(x_shape, dtype=torch.float64)

        w_shape = (50, 3, 2, 2)
        self.w = torch.randn(w_shape, dtype=torch.float64)

        b_shape = (10, 1, 1)
        self.b = torch.randn(b_shape, dtype=torch.float64)

        self.c = Convolution(self.w, self.b)

    def test_forward(self):
        forward = self.c.forward(self.x)
        numpy = forward.numpy()

        x = self.x.numpy()
        wt = self.w.numpy()
        b = self.b.numpy()

        n, c, h, w = x.shape
        fn, c, fh, fw = wt.shape

        out_h = int(1 + (h + 2 * self.c.padding - fh) / self.c.stride)
        out_w = int(1 + (w + 2 * self.c.padding - fw) / self.c.stride)

        output = np.zeros((n, fn, out_h, out_w), dtype=np.float64)
        for on in range(n):
            for ofn in range(fn):
                for oc in range(c):
                    for oh in range(out_h):
                        for ow in range(out_w):
                            output[on, ofn, oh, ow] += np.sum(x[on, oc, oh:oh + fh, ow:ow + fw] * wt[ofn, oc])
                output[on, ofn] += b[ofn]

        #실수 오차 발생
        self.assertTrue(np.allclose(numpy, output, rtol=1e-05, atol=1e-08))


if __name__ == '__main__':
    unittest.main()
