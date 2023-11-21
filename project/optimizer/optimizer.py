class Optimizer:
    def __init__(self, params, lr):
        self.lr = lr

        self.params = []
        self.init_params(params)

    def init_params(self, params):
        for param in params:
            self.params.append(param)

    def zero_grad(self):
        for param in self.params:
            param.grad = None

    def step(self):
        pass
