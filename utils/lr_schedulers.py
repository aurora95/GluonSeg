from mxnet.lr_scheduler import LRScheduler

class PolyScheduler(LRScheduler):
    def __init__(self, base_lr, lr_power, total_steps):
        super(PolyScheduler, self).__init__(base_lr=base_lr)
        self.lr_power = lr_power
        self.total_steps = total_steps

    def __call__(self, num_update):
        lr = self.base_lr * ((1 - float(num_update)/self.total_steps) ** self.lr_power)
        return lr
