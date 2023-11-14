import unittest

import torch
import project.layers.affine as affine


class AffineTest(unittest.TestCase):

    def setUp(self):
        self.affine = affine.Affine(16*16*32, 12, True)

    def test_forward(self):
        test = torch.randn((1, 16, 16, 32), dtype=torch.float32).to('mps')
        test = test.reshape(test.size(0), -1)


        linear = torch.nn.Linear(16 * 16 * 32, 12, bias=True)
        linear.to('mps')
        out = linear(test)

        print(out.shape)
        print(out)

        self.affine.weight = linear.weight
        # self.affine.b = linear.bias
        out = self.affine.forward(test)
        print(out.shape)
        print(out)

    def test_backward(self):
        test = torch.randn((1, 16, 16, 32), dtype=torch.float32).to('mps')
        test = test.reshape(test.size(0), -1)

        linear = torch.nn.Linear(16 * 16 * 32, 12, bias=True)
        linear.to('mps')
        out = linear(test)
        backward = out.backward(torch.ones_like(out))
        print(backward)

        self.affine.weight = linear.weight
        self.affine.b = linear.bias

        out = self.affine.forward(test)
        backward = self.affine.backward(torch.ones_like(out))
        print(backward)

        assert backward.shape == test.shape







