import unittest
import torch
import numpy as np

from project.layers.convolution import Convolution


class ConvolutionTestCase(unittest.TestCase):
    def setUp(self):
        x_shape = (5, 2, 5, 5)
        self.x = torch.randint(0, 5, x_shape, dtype=torch.float64)

        w_shape = (7, 2, 3, 3)
        self.w = torch.randint(0, 5, w_shape, dtype=torch.float64)

        b_shape = (7, 1, 1)
        self.b = torch.randint(0, 5, b_shape, dtype=torch.float64)

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

        # 실수 오차 발생
        self.assertTrue(np.allclose(numpy, output, rtol=1e-05, atol=1e-08))

    def test_backward(self):
        y = self.c.forward(self.x)

        dy_shape = y.shape
        dy = torch.randint(0, 5, dy_shape, dtype=torch.float64)

        backward = self.c.backward(dy)
        numpy = backward.numpy()

        print(numpy)


if __name__ == '__main__':
    unittest.main()