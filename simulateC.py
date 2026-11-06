"""Same control objectives as simulate.py, flown on GeneralCode's compiled trirotor sim.

GeneralCode's classes are used directly through the compiled `GeneralCode` module (`make sim`):
  Trirotor            physics + onboard TrirotorLogic (rate controller, mixer, KF, IMU noise)
  TrirotorController  offboard position/attitude controller
The environment loop is written here in Python and follows GeneralCode's reference sim
(PC-Apps/TrirotorSimulatorShared/TrirotorSimulation.cpp): physics and onboard logic every 2 ms,
offboard controller every 20 ms, commands delivered after a 30 ms radio delay. No GPS is fed:
hover (tiltrotor) control only uses the onboard angular-rate estimate, and the commands carry the
flag that disables the onboard estimator-height safety check.
"""
from __future__ import print_function, division

import argparse
import collections
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

REPO_DIR = os.path.dirname(os.path.abspath(__file__))
# compiled module, then this repo's submodules in third_party/ (py3dmath, pyplot3d)
_THIRD_PARTY = os.path.join(REPO_DIR, "third_party")
sys.path[:0] = [os.path.join(REPO_DIR, "trirotor_sim", "build"),
                os.path.join(_THIRD_PARTY, "py3dmath"), _THIRD_PARTY]
try:
    import GeneralCode
except ImportError as e:
    sys.exit("Could not import the compiled GeneralCode module ({}). Run `make sim` first.".format(e))

parser = argparse.ArgumentParser()
parser.add_argument("--no-animate", action="store_true", help="skip the 3D animation")
args = parser.parse_args()

np.random.seed(0)

#==============================================================================
# Define the simulation (same as simulate.py)
#==============================================================================
dt = 0.002
endTime = 5

dt_us = int(round(dt * 1e6))
offboardPeriod_us = 20000  # offboard controller at 50 Hz
radioDelay_us = 30000      # offboard -> vehicle command latency

#==============================================================================
# Define the vehicle (GeneralCode constants; randomize fields of `consts` here
# for domain randomization)
#==============================================================================
vehicleId = 118  # DARPA trirotor (DarpaConstants::Trirotor veh1)
consts = GeneralCode.QuadcopterConstants(vehicleId)

aeroOffset = np.array([consts.aeroRefPointToComOffset_x,
                       consts.aeroRefPointToComOffset_y,
                       consts.aeroRefPointToComOffset_z])
# offboard controller uses aeroCoeffs, the simulated plant uses simAeroCoeffs (as in the reference sim)
ctrlAeroModel = GeneralCode.LinearAerodynamicModel(consts.mass, GeneralCode.AIR_DENSITY,
                                                   consts.wingParams, consts.aeroCoeffs, aeroOffset)
simAeroModel = GeneralCode.LinearAerodynamicModel(consts.mass, GeneralCode.AIR_DENSITY,
                                                  consts.wingParams, consts.simAeroCoeffs, aeroOffset)

timer = GeneralCode.ManualTimer()
servo = consts.tiltServoDynamics
trirotor = GeneralCode.Trirotor(
    timer, consts.mass, np.array([consts.inertia_xx, consts.inertia_yy, consts.inertia_zz]),
    consts.dx, consts.dy1, consts.dy2, consts.dz,
    centre_of_mass_error=np.zeros(3),
    motor_min_speed=0.0, motor_max_speed=consts.motorMaxSpeed,
    prop_thrust_from_speed_sqr=consts.propellerThrustFromSpeedSqr,
    prop_torque_from_speed_sqr=consts.propellerTorqueFromThrust * consts.propellerThrustFromSpeedSqr,
    motor_time_const=consts.motorTimeConst, motor_inertia=consts.motorInertia,
    servo_slew_rate=servo.slewRate, servo_time_constant=servo.timeConstant,
    servo_kp=servo.Kp, servo_command_delay=servo.commandDelay,
    lin_drag_coeff_b=np.array([consts.linDragCoeffBx, consts.linDragCoeffBy, consts.linDragCoeffBz]),
    aero_model=simAeroModel, vehicle_id=vehicleId,
    onboard_logic_period=dt, constants_override=consts)  # onboard logic knows the true constants

controller = GeneralCode.TrirotorController(ctrlAeroModel)
controller.set_parameters(consts.posControl_natFreq, consts.posControl_damping,
                          consts.attControl_timeConst_x, consts.attControl_timeConst_y,
                          consts.attControl_timeConst_z, consts.attControl_integralGain,
                          consts.attControl_integralLimit_x, consts.attControl_integralLimit_y,
                          consts.attControl_integralLimit_z)
cmdFlags = GeneralCode.FLAG_DISABLE_ONBOARD_STATE_SAFETY_CHECKS

#==============================================================================
# Define the trajectory (same draws as simulate.py)
#==============================================================================
pos0 = np.random.uniform(-2, 2, 3)  # position
pos0[2] += 2  # make sure it starts above the ground
vel0 = np.random.uniform(-2, 2, 3)  # velocity
att0 = np.random.uniform(-1, 1, 4)  # attitude quaternion (w, x, y, z)
att0 /= np.linalg.norm(att0)  # normalize

posf = np.array([0, 0, 2])  # position
velf = np.array([0, 0, 0])  # velocity
accf = np.array([0, 0, 0])  # acceleration
yawf = 0.0

angVel0 = np.zeros(3)  # start at zero angular velocity

trirotor.set_position(pos0)
trirotor.set_velocity(vel0)
trirotor.set_attitude(att0)
trirotor.set_angular_velocity(angVel0)
# start the onboard estimator at the true state (otherwise it fights a fictitious initial error)
trirotor.set_kf_position(pos0)
trirotor.set_kf_velocity(vel0)
trirotor.set_kf_attitude(att0)
trirotor.set_kf_angular_velocity(angVel0)

#==============================================================================
# Run the simulation
#==============================================================================
numSteps = int(endTime / dt)
nMot = 3

times = np.zeros([numSteps, 1])
posHistory = np.zeros([numSteps, 3])
velHistory = np.zeros([numSteps, 3])
attHistory = np.zeros([numSteps, 3])  # yaw, pitch, roll
angVelHistory = np.zeros([numSteps, 3])
motForcesHistory = np.zeros([numSteps, nMot])
tiltHistory = np.zeros([numSteps, 2])
cmdAngVelHistory = np.zeros([numSteps, 3])

radioQueue = collections.deque()  # (delivery time [us], radio message)
offboardElapsed_us = 0
cmdAngVel = np.zeros(3)

for index in range(numSteps):
    # physics + onboard logic, then advance time (same order as the reference sim)
    trirotor.run()
    timer.advance_us(dt_us)
    t_us = timer.micro_seconds()

    # offboard controller (true state, like simulate.py)
    offboardElapsed_us += dt_us
    if offboardElapsed_us > offboardPeriod_us:
        offboardElapsed_us -= offboardPeriod_us
        cmdAngVel, cmdVerticalThrust, cmdForwardThrust = controller.run(
            trirotor.position(), trirotor.velocity(), trirotor.attitude(),
            posf, velf, accf, 0.0, yawf, offboardPeriod_us * 1e-6)
        msg = GeneralCode.make_tiltrotor_command(cmdFlags, cmdVerticalThrust, cmdForwardThrust,
                                                 cmdAngVel)
        radioQueue.append((t_us + radioDelay_us, msg))

    # deliver commands whose radio delay has elapsed
    while radioQueue and radioQueue[0][0] <= t_us:
        trirotor.set_command_radio_msg(radioQueue.popleft()[1])

    times[index] = t_us * 1e-6
    posHistory[index, :] = trirotor.position()
    velHistory[index, :] = trirotor.velocity()
    attHistory[index, :] = trirotor.euler_ypr()
    angVelHistory[index, :] = trirotor.angular_velocity()
    motForcesHistory[index, :] = trirotor.motor_forces()
    tiltHistory[index, :] = trirotor.tilt_angles()
    cmdAngVelHistory[index, :] = cmdAngVel

print("Final position: {}, goal: {}".format(np.round(posHistory[-1], 3), posf))

#==============================================================================
# Make the plots 
#==============================================================================
fig, ax = plt.subplots(6, 1, sharex=True, figsize=(8, 12))

ax[0].plot(times, posHistory[:, 0], label='x')
ax[0].plot(times, posHistory[:, 1], label='y')
ax[0].plot(times, posHistory[:, 2], label='z')
ax[0].plot(times, np.ones_like(times) * posf[0], '--', color='C0')
ax[0].plot(times, np.ones_like(times) * posf[1], '--', color='C1')
ax[0].plot(times, np.ones_like(times) * posf[2], '--', color='C2')
ax[1].plot(times, velHistory)
ax[2].plot(times, attHistory[:, 0] * 180 / np.pi, label='Y')
ax[2].plot(times, attHistory[:, 1] * 180 / np.pi, label='P')
ax[2].plot(times, attHistory[:, 2] * 180 / np.pi, label='R')
ax[3].plot(times, angVelHistory[:, 0], label='p')
ax[3].plot(times, angVelHistory[:, 1], label='q')
ax[3].plot(times, angVelHistory[:, 2], label='r')
for i in range(nMot):
    ax[4].plot(times, motForcesHistory[:, i], label='mot{}'.format(i))
ax[5].plot(times, tiltHistory[:, 0] * 180 / np.pi, label='tilt0 (right)')
ax[5].plot(times, tiltHistory[:, 1] * 180 / np.pi, label='tilt1 (left)')

ax[-1].set_xlabel('Time [s]')

ax[0].set_ylabel('Pos')
ax[1].set_ylabel('Vel')
ax[2].set_ylabel('Att [deg]')
ax[3].set_ylabel('AngVel (in B)')
ax[4].set_ylabel('MotForces [N]')
ax[5].set_ylabel('Tilt [deg]')

ax[0].set_xlim([0, endTime])
for a in (ax[0], ax[2], ax[3], ax[4], ax[5]):
    a.legend()

plt.show()

#==============================================================================
# Save the data (data/trirotor_data.csv)
#==============================================================================
data_dict = {
    'Time': times.flatten(),
    'PosX': posHistory[:, 0],
    'PosY': posHistory[:, 1],
    'PosZ': posHistory[:, 2],
    'VelX': velHistory[:, 0],
    'VelY': velHistory[:, 1],
    'VelZ': velHistory[:, 2],
    'Yaw': attHistory[:, 0],
    'Pitch': attHistory[:, 1],
    'Roll': attHistory[:, 2],
}
data = pd.DataFrame(data_dict)
data_path = os.path.join(REPO_DIR, 'data', 'trirotor_data.csv')
data.to_csv(data_path, index=False)
print("Saved {}".format(data_path))

#==============================================================================
# Animate (draws the quad-shaped body for now)
#==============================================================================
if not args.no_animate:
    from pyplot3d.utils import ypr_to_R
    from animate import animate_quadcopter_history

    data = pd.read_csv(data_path)
    times = data['Time'].values
    posHistory = data[['PosX', 'PosY', 'PosZ']].values
    attHistory = data[['Yaw', 'Pitch', 'Roll']].values
    x = posHistory.T
    steps = len(times)

    R = np.zeros((3, 3, steps))
    for i in range(steps):
        R[:, :, i] = ypr_to_R(attHistory[i, :], degrees=False)

    animate_quadcopter_history(times, x, R, arm_length=0.4)
