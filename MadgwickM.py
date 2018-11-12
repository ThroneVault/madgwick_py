# -*- coding: utf-8 -*-
"""
    Copyright (c) 2015 Jonas Böer, jonas.boeer@student.kit.edu

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import warnings
import numpy as np
from numpy.linalg import norm
import quaternion


class MadgwickM:
    T_DEFAULT = 1/100
    Q0 = np.quaternion(1, 0, 0, 0)

    def __init__(self, sampleperiod, beta, quat=Q0):
        """
        Initialize the class with the given parameters.
        :param sampleperiod: The sample period
        :param quaternion: Initial np.quaternion
        :param beta: Algorithm gain beta
        :return:
        """
        self.samplePeriod = sampleperiod
        self.quaternion = quat
        self.beta = beta

    def update(self, gyro, accel, mag):
        """
        Perform one update step with data from a AHRS sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        :param magnetometer: A three-element array containing the magnetometer data. Can be any unit since a normalized value is used.
        :return:
        """
        qnp = self.quaternion
        q = quaternion.as_float_array(self.quaternion)

        gyroscope = gyro.flatten()
        accelerometer = accel.flatten()
        magnetometer = mag.flatten()

        if norm(magnetometer)>200.:
            raise ValueError("magnetometer value a bit off")
        if norm(accelerometer)>30.:
            raise ValueError("accelerometer value a bit off")

        # Normalise accelerometer measurement
        if norm(accelerometer) is 0 or norm(accelerometer) is np.nan:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Validate magnetometer measurement
        if norm(magnetometer) is 0 or norm(magnetometer) is np.nan:
            warnings.warn("magnetometer is zero")
            return

        # discard the vertical component of magnetic field measurement
        # magnetometer reading in Earth's frame quaternion
        m_e = qnp * np.quaternion(0, *magnetometer) * qnp.conj()
        # set vertical component to 0 and return to sensor frame
        m_M = qnp.conj() * np.quaternion(0, m_e.x, m_e.y, 0) * qnp
        magnetometer=m_M.imag
        # assume that the actual magnetic field only has a horizontal component pointing north
        # b=np.array([0, 1, 0])

        # Normalise magnetometer measurement
        if norm(magnetometer) is 0 or norm(magnetometer) is np.nan:
            warnings.warn("ADJUSTED magnetometer is zero")
            return
        magnetometer /= norm(magnetometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
            2*(0.5 - q[2]**2 - q[3]**2) - magnetometer[0],
            2*(q[1]*q[2] - q[0]*q[3]) - magnetometer[1],
            2*(q[0]*q[2] + q[1]*q[3]) - magnetometer[2]
        ])
        # delta f / delta quaternion
        j = np.array([
            [-2*q[2],                  2*q[3],                  -2*q[0],             2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],              2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0],
            [0,                         0,                      -4*q[2],            -4*q[3]],
            [-2*q[3],                   2*q[2],                 2*q[1],             -2*q[0]],
            [2*q[2],                    2*q[3],                 2*q[0],              2*q[1]]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (qnp * np.quaternion(0, *gyroscope)) * 0.5 - np.quaternion(*(self.beta * step))

        # Integrate to yield quaternion
        qnp += qdot * self.samplePeriod
        self.quaternion = qnp.normalized()  # normalise quaternion


        
    def getQuatnp(self):
        return self.quaternion.copy()