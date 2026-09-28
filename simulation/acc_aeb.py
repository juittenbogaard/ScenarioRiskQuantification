""" Model of ACC with AEB

Creation date: 2026 08 31
Author(s): Jeroen Uittenbogaard

Modifications:
"""

import numpy as np
from .acc import ACC, ACCParameters, ACCState


class ACCAEBParameters(ACCParameters):
    """ Parameters for the ACCAEB. """
    aeb_threshold: float = 1.0 # ttc threshold for AEB activation
    max_decel: float = 10 # maximum deceleration for AEB

    def __init__(self, **kwargs):
        ACCParameters.__init__(self, **kwargs)


class ACCAEBState(ACCState):
    """ State of the ACCAEB. """
    aeb: bool = False


class ACCAEB(ACC):
    """ Class for simulation of the ACC with AEB. """
    def __init__(self):
        ACC.__init__(self)
        self.parms = ACCAEBParameters()
        self.state = ACCAEBState()

        self.nstep = 0

    def init_simulation(self, parms: ACCAEBParameters) -> None:
        ACC.init_simulation(self, parms)
        self.parms.aeb_threshold = parms.aeb_threshold
        self.parms.max_decel = parms.max_decel
        self.state.aeb = False

    def acceleration(self, gap: float, vhost: float, vdiff: float) -> float:
        """ Compute the acceleration based on the gap, vhost, vdiff.

        :param gap: Gap with preceding vehicle.
        :param vhost: Speed of host vehicle.
        :param vdiff: Difference in speed between leading and host vehicle.
        :return: The acceleration.
        """
        # If target is out of range, use the cruise control.
        if gap > self.parms.sensor_range or (gap < 0 and self.parms.cruise_after_collision):
            return self._acceleration_cc(vhost)
        if gap/vdiff < self.parms.aeb_threshold or self.state.aeb==True:
            self.state.aeb = True
            return -self.parms.max_decel
        return min(self._acceleration_cc(vhost),
                    self._acceleration_acc(gap, vhost, vdiff))
