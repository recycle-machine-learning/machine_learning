import unittest

import torch

from project.layers import batch_norm


class BatchNormTest(unittest.TestCase):
    def setUp(self):
        self.device = 'cpu'

        self.bn = torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        self.bn2 = batch_norm.BatchNormalization(1, epsilon=1e-05, momentum=0.1, affine=True)

    def test_forward(self):
        test = torch.randn((2, 1, 4, 4), dtype=torch.float32)

        bn_out = self.bn.forward(test)
        print(bn_out)

        bn2_out = self.bn2.forward(test)


        print(bn2_out)

        self.assertTrue(torch.allclose(bn_out, bn2_out, rtol=1e-8, atol=1e-5))

    def test_backward(self):
        test = torch.randn((2, 1, 4, 4), dtype=torch.float32)

        out_bn = self.bn.forward(test)
        loss_bn = torch.sum(out_bn)
        loss_bn.backward()

        print(self.bn.weight.grad)

        out_bn2 = self.bn2.forward(test)
        loss_bn2 = torch.sum(out_bn2)
        loss_bn2.backward()

        print(self.bn2.weight.grad)

        self.assertTrue(self.bn.weight.grad, self.bn2.weight.grad)