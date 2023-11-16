import unittest

import torch
import project.layers.affine as affine


class AffineTest(unittest.TestCase):

    def setUp(self):
        self.device = 'cpu'

        self.linear = torch.nn.Linear(16 * 16 * 32, 12, bias=True, device=self.device)

        self.affine = affine.Affine(16 * 16 * 32, 12, True, device=self.device)
        self.affine.weight = self.linear.weight
        self.affine.b = self.linear.bias

    def test_forward(self):
        test = torch.randn((1, 16, 16, 32), dtype=torch.float32).to(self.device)
        test = test.reshape(test.size(0), -1)

        linear_out = self.linear(test)
        affine_out = self.affine.forward(test)

        print(linear_out)
        print(affine_out)

        self.assertTrue(torch.allclose(linear_out, affine_out, rtol=1e-8, atol=1e-5))

    def test_backward(self):
        test = torch.randn((1, 16, 16, 32), dtype=torch.float32).to(self.device)
        test = test.reshape(test.size(0), -1)

        out_linear = self.linear.forward(test)
        loss_linear = torch.sum(out_linear)
        loss_linear.backward()
        dout_linear = self.linear.weight
        print(dout_linear)

        out_affine = self.affine.forward(test)
        loss_affine = torch.sum(out_affine)
        loss_affine.backward()
        dout_affine = self.affine.weight
        print(dout_affine)

        self.assertTrue(torch.allclose(dout_linear, dout_affine, rtol=1e-8, atol=1e-5))
