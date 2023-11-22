import unittest

import torch

from project.layers import batch_norm


class BatchNormTest(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'

        self.bn = torch.nn.BatchNorm2d(2, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.bn2 = batch_norm.BatchNormalization(2, momentum=0.9, affine=True)

    def test_forward(self):
        test = torch.randn((2, 2, 4, 4), dtype=torch.float32)

        bn_out = self.bn.forward(test)
        bn2_out = self.bn2.forward(test)

        self.assertTrue(torch.allclose(bn_out, bn2_out, rtol=1e-8, atol=1e-5))

    def test_backward(self):
        test = torch.randn((2, 2, 2, 2), dtype=torch.float32)

        out_bn = self.bn.forward(test)

        loss_bn = torch.sum(out_bn, dim=(0,2,3))

        print("loss", loss_bn)
        loss_bn = torch.sum(out_bn)
        loss_bn.backward()

        print(self.bn.weight.grad)
        print(torch.sum(self.bn.weight.grad))
        print(self.bn.bias.grad)

        out_bn2 = self.bn2.forward(test)
        self.bn2.backward(out_bn2)


        print(self.bn2.weight.grad)
        print(self.bn2.bias.grad)

        self.assertTrue(torch.allclose(out_bn, out_bn2, rtol=1e-8, atol=1e-8))
        # self.assertTrue(torch.allclose(self.bn.bias.grad, self.bn2.bias.grad, rtol=1e-13, atol=1e-13))
        # self.assertTrue(torch.allclose(self.bn.weight.grad, self.bn2.weight.grad, rtol=1e-13, atol=1e-13))
