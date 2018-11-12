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


class Madgwick:
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

        # Normalise magnetometer measurement
        if norm(magnetometer) is 0 or norm(magnetometer) is np.nan:
            warnings.warn("magnetometer is zero")
            return
        magnetometer /= norm(magnetometer)
        # magnetometer reading in Earth's frame quaternion
        # Sources of interference fixed in the sensor frame,
        # termed hard iron biases, can be removed through calibration
        h = qnp * np.quaternion(0, *magnetometer) * qnp.conj()
        # the following is based on assumption that magnetic field is in the direction of the north
        # (no East/West component), but does have a vertical component (inclination). This is not to
        # manipulate the measurement, but to obtain the inclination of the field as the reference.
        # See section III.D
        b = np.array([0, norm(h.imag[0:2]), 0, h.z])

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2],
            2*b[1]*(0.5 - q[2]**2 - q[3]**2) + 2*b[3]*(q[1]*q[3] - q[0]*q[2]) - magnetometer[0],
            2*b[1]*(q[1]*q[2] - q[0]*q[3]) + 2*b[3]*(q[0]*q[1] + q[2]*q[3]) - magnetometer[1],
            2*b[1]*(q[0]*q[2] + q[1]*q[3]) + 2*b[3]*(0.5 - q[1]**2 - q[2]**2) - magnetometer[2]
        ])
        j = np.array([
            [-2*q[2],                  2*q[3],                  -2*q[0],                  2*q[1]],
            [2*q[1],                   2*q[0],                  2*q[3],                   2*q[2]],
            [0,                        -4*q[1],                 -4*q[2],                  0],
            [-2*b[3]*q[2],             2*b[3]*q[3],             -4*b[1]*q[2]-2*b[3]*q[0], -4*b[1]*q[3]+2*b[3]*q[1]],
            [-2*b[1]*q[3]+2*b[3]*q[1], 2*b[1]*q[2]+2*b[3]*q[0], 2*b[1]*q[1]+2*b[3]*q[3],  -2*b[1]*q[0]+2*b[3]*q[2]],
            [2*b[1]*q[2],              2*b[1]*q[3]-4*b[3]*q[1], 2*b[1]*q[0]-4*b[3]*q[2],  2*b[1]*q[1]]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (qnp * np.quaternion(0, *gyroscope)) * 0.5 - np.quaternion(*(self.beta * step))

        # Integrate to yield quaternion
        qnp += qdot * self.samplePeriod
        self.quaternion = qnp.normalized()  # normalise quaternion

    def update_imu(self, gyroscope, accelerometer):
        """
        Perform one update step with data from a IMU sensor array
        :param gyroscope: A three-element array containing the gyroscope data in radians per second.
        :param accelerometer: A three-element array containing the accelerometer data. Can be any unit since a normalized value is used.
        """
        q = self.quaternion

        gyroscope = np.array(gyroscope, dtype=float).flatten()
        accelerometer = np.array(accelerometer, dtype=float).flatten()

        # Normalise accelerometer measurement
        if norm(accelerometer) is 0:
            warnings.warn("accelerometer is zero")
            return
        accelerometer /= norm(accelerometer)

        # Gradient descent algorithm corrective step
        f = np.array([
            2*(q[1]*q[3] - q[0]*q[2]) - accelerometer[0],
            2*(q[0]*q[1] + q[2]*q[3]) - accelerometer[1],
            2*(0.5 - q[1]**2 - q[2]**2) - accelerometer[2]
        ])
        j = np.array([
            [-2*q[2], 2*q[3], -2*q[0], 2*q[1]],
            [2*q[1], 2*q[0], 2*q[3], 2*q[2]],
            [0, -4*q[1], -4*q[2], 0]
        ])
        step = j.T.dot(f)
        step /= norm(step)  # normalise step magnitude

        # Compute rate of change of quaternion
        qdot = (q * np.quaternion(0, *gyroscope)) * 0.5 - self.BETA0 * step.T

        # Integrate to yield quaternion
        q += qdot * self.samplePeriod
        self.quaternion = q.normalised()  # normalise quaternion
        
    def getQuatnp(self):
        return self.quaternion.copy()