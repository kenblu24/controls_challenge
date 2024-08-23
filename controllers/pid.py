
from . import BaseController
import numpy as np
from time import time_ns
from math import pi, isnan
from typing import override


class Controller(BaseController):
    """
    A simple PID controller
    """
    def __init__(self,):
        self.p = 0.3
        self.i = 0.05
        self.d = -0.1
        self.error_integral = 0
        self.prev_error = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = (target_lataccel - current_lataccel)
        self.error_integral += error
        error_diff = error - self.prev_error
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff


"""
PID by https://github.com/kidswong999
modified from https://github.com/openmv/openmv/blob/master/scripts/libraries/pid.py
Example:
from pid import PID
pid1 = PID(p=0.07, i=0, imax=90)
while(True):
    error = 50 #error should be calculated, target - measure
    output=pid1.get_pid(error,1)
    #control value with output
"""


class LivePID:
    _kp = _ki = _kd = _integrator = _imax = 0
    _last_error = _last_derivative = _last_t = 0
    _RC = 1 / (2 * pi * 20)

    def __init__(self, p=0., i=0., d=0., imax=None, scale=1):
        self._kp = float(p)
        self._ki = float(i)
        self._kd = float(d)
        self._imax = abs(imax) if imax is not None else None
        self._last_derivative = float("nan")
        self.scale = float(scale)

    def get_pid(self, error, scaler=1):
        tnow = time_ns() * 1e-3
        dt = tnow - self._last_t
        output = 0
        if self._last_t == 0 or dt > 1000:
            dt = 0
            self.reset_I()
        self._last_t = tnow
        delta_time = float(dt) / float(1000)
        output += error * self._kp
        if abs(self._kd) > 0 and dt > 0:
            if isnan(self._last_derivative):
                derivative = 0
                self._last_derivative = 0
            else:
                derivative = (error - self._last_error) / delta_time
            derivative = self._last_derivative + ((delta_time / (self._RC + delta_time)) * (derivative - self._last_derivative))
            self._last_error = error
            self._last_derivative = derivative
            output += self._kd * derivative
        output *= scaler
        if abs(self._ki) > 0 and dt > 0:
            self._integrator += (error * self._ki) * scaler * delta_time
            if self._integrator < -self._imax:
                self._integrator = -self._imax
            elif self._integrator > self._imax:
                self._integrator = self._imax
            output += self._integrator
        return output

    def reset_I(self):
        self._integrator = 0
        self._last_derivative = float("nan")


def clamp(x, xmin, xmax=None):
    if xmax is None:
        xmax = abs(xmin)
        xmin = -1 * abs(xmin)
    return max(xmin, min(x, xmax))


# modified to use discrete time
class DeadPID(LivePID):
    @override
    def get_pid(self, error, scaler=1):
        dt = 1
        output = 0
        output += error * self._kp
        if abs(self._kd) > 0:
            if isnan(self._last_derivative):
                derivative = 0
                self._last_derivative = 0
            else:
                derivative = (error - self._last_error) / dt
            # derivative = self._last_derivative + ((dt / (self._RC + dt)) * (derivative - self._last_derivative))
            self._last_error = error
            self._last_derivative = derivative
            output += self._kd * derivative
        output *= scaler * self.scale
        self._integrator += (error * self._ki) * scaler * self.scale * dt
        if self._imax is not None:
            self._integrator = clamp(self._integrator, self._imax)
        output += self._integrator
        return output


class PIDController(DeadPID, BaseController):
    @override
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        return self.get_pid(error)

class PID4Controller(PIDController):
    def __init__(self, p=0.0, i=0.0, d=0.0, d2=0.0, imax=None, scale=1):
        super().__init__(p, i, d, imax, scale)
        self._kd2 = d2

    @override
    def get_pid(self, error, scaler=1):
        dt = 1
        output = 0
        output += error * self._kp
        if abs(self._kd) > 0:
            if isnan(self._last_derivative):
                derivative = 0
                derivative2 = 0
                self._last_derivative = 0
            else:
                derivative = (error - self._last_error) / dt
            derivative2 = (derivative - self._last_derivative) / dt
            # print(derivative2)
            derivative = self._last_derivative + ((dt / (self._RC + dt)) * (derivative - self._last_derivative))
            self._last_error = error
            self._last_derivative = derivative
            output += self._kd * derivative + self._kd2 * derivative2
        output *= scaler * self.scale
        self._integrator += (error * self._ki) * scaler * self.scale * dt
        if self._imax is not None:
            self._integrator = clamp(self._integrator, self._imax)
        output += self._integrator
        return output


class PIDDController(BaseController):
    """
    A simple PID controller
    """

    def __init__(
        self,
    ):
        self.p = 0.12
        self.i = 0.115
        self.d = 0.005
        self.dd = 0.005
        self.error_integral = 0
        self.prev_error = 0
        self.prev_error_diff = 0

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        error = target_lataccel - current_lataccel
        self.error_integral += error
        error_diff = error - self.prev_error
        error_diff2 = error_diff - self.prev_error_diff
        self.prev_error_diff = error_diff
        self.prev_error = error
        return self.p * error + self.i * self.error_integral + self.d * error_diff + self.dd * error_diff2
