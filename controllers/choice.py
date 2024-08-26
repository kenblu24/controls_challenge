from typing import override
from . import BaseController
from . import pid


# genome = [0.08582296707938367, 0.12810866517604869, 0.19527305555877467, -0.10766175831842266]
# genome = [0.126703440654386, 0.122940798062283, 0.0673089654719858, -0.0420852231414253]
genome = [0.137238378977777, 0.118833817200919, 0.0663160330331095, -0.0350704327986692]


class Controller(BaseController):
    def __init__(self):
        # self.controller = pid.PIDController(p=0.12, i=0.115, d=0.005, scale=1, imax=1)
        # self.controller = pid.PID4Controller(p=0.12, i=0.115, d=0.005, d2=-0.0008, scale=1, imax=1)
        self.controller = pid.PID4Controller(*genome, scale=1, imax=1)
        # self.controller = pid.PIDDController()

    @override
    def update(self, *args, **kwargs):
        return self.controller.update(*args, **kwargs)
