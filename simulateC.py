"""GeneralCode's compiled trirotor simulator and controllers: every controller variant for a flight mode is flown from the same perturbed
trim state, and the results are overlaid.

  --mode forward (default)  15 m/s level-flight trim from GeneralCode's flight condition solver
      rates / solver   TrirotorFixedWingController.run              -> trirotorForwardWingCmd
                       (onboard fixed-wing rate controller + forward mixer: motors, tilts, surfaces)
      px4 / solver     .run_attitude_setpoint                       -> trirotorAttitudeCmd
                       (onboard PX4 attitude + rate controllers -> surfaces, motors fixed forward)
      rates / TECS     .run_tecs                                    -> trirotorForwardWingCmd
      px4 / TECS       .run_tecs_attitude_setpoint                  -> trirotorAttitudeCmd
  --mode hover              level hover
      trirotor         TrirotorController   -> trirotorTiltrotorCmd (onboard tiltrotor rate ctrl + mixer)
      quad             QuadcopterController -> externalRatesCmd     (onboard external-rates mode)
"""
from __future__ import print_function, division

import argparse
import collections
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial.transform import Rotation as R

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
parser.add_argument("--mode", choices=["forward", "hover"], default="forward",
                    help="flight mode whose controller variants are compared (default: forward)")
parser.add_argument("--no-animate", action="store_true", help="skip the 3D animation (hover mode)")
args = parser.parse_args()
forward = args.mode == "forward"

np.random.seed(5)


def quat_from_ypr(ypr):
    """(yaw, pitch, roll) -> quaternion (w, x, y, z); same convention as GeneralCode's FromEulerYPR."""
    x, y, z, w = R.from_euler("ZYX", ypr).as_quat()
    return np.array([w, x, y, z])


def ypr_from_quat(q):
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("ZYX")


#==============================================================================
# Simulation settings
#==============================================================================
dt = 0.002
endTime = 10 if forward else 5  # [s] logged time after release

dt_us = int(round(dt * 1e6))
offboardPeriod_us = 10000  # offboard controller at 100 Hz (mainLoopFrequency in the ROS loiter node)
radioDelay_us = 30000      # offboard -> vehicle command latency
warmup_us = 2000000        # held on the trim state while the actuators settle

#==============================================================================
# Vehicle (GeneralCode constants; fields of `consts` can be randomized here)
#==============================================================================
vehicleId = 118  # DARPA trirotor (DarpaConstants::Trirotor veh1)
consts = GeneralCode.QuadcopterConstants(vehicleId)

aeroOffset = np.array([consts.aeroRefPointToComOffset_x,
                       consts.aeroRefPointToComOffset_y,
                       consts.aeroRefPointToComOffset_z])
# offboard controllers use aeroCoeffs, the simulated plant uses simAeroCoeffs (as in the reference sim)
ctrlAeroModel = GeneralCode.LinearAerodynamicModel(consts.mass, GeneralCode.AIR_DENSITY,
                                                   consts.wingParams, consts.aeroCoeffs, aeroOffset)
simAeroModel = GeneralCode.LinearAerodynamicModel(consts.mass, GeneralCode.AIR_DENSITY,
                                                  consts.wingParams, consts.simAeroCoeffs, aeroOffset)

cmdFlags = GeneralCode.FLAG_DISABLE_ONBOARD_STATE_SAFETY_CHECKS
groundWind = np.zeros(3)


def make_vehicle():
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
    return timer, trirotor


#==============================================================================
# Trim state and initial condition
#==============================================================================
if forward:
    airspeed = 15.0   # [m/s] (no wind: equal to ground speed)
    altitude = 20.0   # [m]
    velTrim = np.array([airspeed, 0.0, 0.0])  # heading +x
    pos0 = np.array([0.0, 0.0, altitude])
    # level flight: required proper acceleration = gravity only
    trim = GeneralCode.CoordinatedFlightConditionSolver(ctrlAeroModel).solve(
        velTrim, np.array([0.0, 0.0, 9.81]))
    attTrim = trim.att
    print("Trim at {} m/s: yaw/pitch/roll {} deg, alpha {:.2f} deg, thrust {:.3f}, status {}".format(
        airspeed, np.round(np.degrees(ypr_from_quat(attTrim)), 2), np.degrees(trim.alpha), trim.T,
        trim.status.name))
else:
    velTrim = np.zeros(3)
    pos0 = np.array([0.0, 0.0, 2.0])
    attTrim = np.array([1.0, 0.0, 0.0, 0.0])  # level, yaw 0

yprTrim = ypr_from_quat(attTrim)
yprInit = yprTrim + np.radians(np.random.uniform(-10, 10, 3))
attInit = quat_from_ypr(yprInit)
print("Initial yaw/pitch/roll {} deg (trim {} deg)".format(np.round(np.degrees(yprInit), 2),
                                                          np.round(np.degrees(yprTrim), 2)))


def trim_position(t_rel):
    """Where the vehicle would be if it stayed exactly on the trim state (t_rel = time since release)."""
    return pos0 + velTrim * t_rel


def hold_reference(pos):
    """Reference holding the trim condition. Forward flight is path following along the straight
    trim line, the zero-curvature case of GeneralCode's LoiterTrajectory::GetReference: the reference
    is the point on the line closest to the vehicle (so no along-track error, only cross-track and
    height error), moving at the trim velocity. Hover keeps the start point."""
    if forward:
        trackDir = velTrim / np.linalg.norm(velTrim)
        return pos0 + np.dot(pos - pos0, trackDir) * trackDir, velTrim, np.zeros(3)
    return pos0, velTrim, np.zeros(3)


#==============================================================================
# Controller variants: each returns a function trirotor -> (cmd_ang_vel, radio message)
#==============================================================================
def forward_controller(onboard, guidance):
    fw = consts.fixedWingControlConsts
    tc = consts.tecsControlConsts
    lg = consts.lateralGuidanceConsts
    ctrl = GeneralCode.TrirotorFixedWingController(ctrlAeroModel)
    ctrl.set_parameters(fw.posControl_natFreq_x, fw.posControl_natFreq_y, fw.posControl_natFreq_z,
                        fw.posControl_damping_x, fw.posControl_damping_y, fw.posControl_damping_z,
                        fw.attControl_timeConst_x, fw.attControl_timeConst_y, fw.attControl_timeConst_z,
                        fw.attControl_integralGain, fw.attControl_integralLimit_x,
                        fw.attControl_integralLimit_y, fw.attControl_integralLimit_z)
    ctrl.set_lateral_guidance_parameters(lg.maxBankAngle, lg.minHeadingSpeed)
    ctrl.set_tecs_parameters(tc.throttle_Kp, tc.throttle_Ki, tc.pitch_Kp, tc.pitch_Ki,
                             tc.energyDistributionWeight, tc.heightRateTimeConst,
                             tc.maxClimbRateDemand, tc.minClimbRateDemand, tc.maxAirspeedDemandRate,
                             tc.minPitchAngle, tc.maxPitchAngle, tc.minThrust, tc.maxThrust,
                             tc.throttleTrim, tc.pitchTrim, tc.throttleIntegratorLimit,
                             tc.pitchIntegratorLimit, tc.loadFactorCompensationGain)
    period = offboardPeriod_us * 1e-6
    lastGood = [attTrim, trim.T]  # attitude setpoint held if the solver goes degenerate

    def command(tri):
        pos, vel, att = tri.position(), tri.velocity(), tri.attitude()
        desPos, desVel, desAcc = hold_reference(pos)
        if guidance == "tecs" and onboard == "px4":
            desAtt, thrust = ctrl.run_tecs_attitude_setpoint(pos, vel, att, desPos, desVel, desAcc,
                                                             groundWind, period)
            return np.zeros(3), GeneralCode.make_attitude_command(cmdFlags, thrust, desAtt)
        if guidance == "tecs":
            angVel, thrust = ctrl.run_tecs(pos, vel, att, desPos, desVel, desAcc, groundWind, period)
            return angVel, GeneralCode.make_forward_wing_command(cmdFlags, thrust, angVel)
        if onboard == "px4":
            # like the lab's TrirotorLoiterFlightControl node: hold the last good setpoint if the solver
            # is degenerate, fly the best effort if it can't bracket a solution
            desAtt, thrust, status = ctrl.run_attitude_setpoint(pos, vel, desPos, desVel, desAcc,
                                                                groundWind)
            if status == GeneralCode.SolveStatus.DEGENERATE:
                desAtt, thrust = lastGood
            else:
                lastGood[:] = [desAtt, thrust]
            return np.zeros(3), GeneralCode.make_attitude_command(cmdFlags, thrust, desAtt)
        angVel, thrust = ctrl.run(pos, vel, att, desPos, desVel, desAcc, groundWind, period)
        return angVel, GeneralCode.make_forward_wing_command(cmdFlags, thrust, angVel)

    return command


def hover_trirotor_controller():
    ctrl = GeneralCode.TrirotorController(ctrlAeroModel)
    ctrl.set_parameters(consts.posControl_natFreq, consts.posControl_damping,
                        consts.attControl_timeConst_x, consts.attControl_timeConst_y,
                        consts.attControl_timeConst_z, consts.attControl_integralGain,
                        consts.attControl_integralLimit_x, consts.attControl_integralLimit_y,
                        consts.attControl_integralLimit_z)

    def command(tri):
        desPos, desVel, desAcc = hold_reference(tri.position())
        angVel, verticalThrust, forwardThrust = ctrl.run(
            tri.position(), tri.velocity(), tri.attitude(), desPos, desVel, desAcc, 0.0, yprTrim[0],
            offboardPeriod_us * 1e-6)
        return angVel, GeneralCode.make_tiltrotor_command(cmdFlags, verticalThrust, forwardThrust, angVel)

    return command


def hover_quad_controller():
    ctrl = GeneralCode.QuadcopterController()
    ctrl.set_parameters(consts.posControl_natFreq, consts.posControl_damping,
                        consts.attControl_timeConst_xy, consts.attControl_timeConst_z)

    def command(tri):
        desPos, desVel, desAcc = hold_reference(tri.position())
        angVel, thrust = ctrl.run(tri.position(), tri.velocity(), tri.attitude(),
                                  desPos, desVel, desAcc, yprTrim[0])
        return angVel, GeneralCode.make_rates_command(cmdFlags, thrust, angVel)

    return command


if forward:
    variants = collections.OrderedDict([
        ("rates / solver", lambda: forward_controller("rates", "solver")),
        ("px4 / solver", lambda: forward_controller("px4", "solver")),
        ("rates / TECS", lambda: forward_controller("rates", "tecs")),
        ("px4 / TECS", lambda: forward_controller("px4", "tecs")),
    ])
else:
    variants = collections.OrderedDict([
        ("trirotor", hover_trirotor_controller),
        ("quad", hover_quad_controller),
    ])


#==============================================================================
# Run one variant
#==============================================================================
def set_state(tri, pos, vel, att, angVel):
    """Set the true state and start the onboard estimator at it."""
    tri.set_position(pos)
    tri.set_velocity(vel)
    tri.set_attitude(att)
    tri.set_angular_velocity(angVel)
    tri.set_kf_position(pos)
    tri.set_kf_velocity(vel)
    tri.set_kf_attitude(att)
    tri.set_kf_angular_velocity(angVel)


def run_variant(make_command):
    timer, tri = make_vehicle()
    command = make_command()
    numSteps = int(endTime / dt)
    h = {k: np.zeros([numSteps, n]) for k, n in [
        ("time", 1), ("pos", 3), ("posErr", 3), ("vel", 3), ("att", 3), ("angVel", 3),
        ("motForces", 3), ("tilt", 2), ("aileron", 2), ("tail", 2), ("cmdAngVel", 3)]}

    set_state(tri, trim_position(-warmup_us * 1e-6), velTrim, attTrim, np.zeros(3))
    radioQueue = collections.deque()  # (delivery time [us], radio message)
    offboardElapsed_us = 0
    cmdAngVel = np.zeros(3)
    released = False
    index = 0

    while index < numSteps:
        # physics + onboard logic, then advance time (same order as the reference sim)
        tri.run()

        timer.advance_us(dt_us)
        t_us = timer.micro_seconds()
        t_rel = (t_us - warmup_us) * 1e-6  # time since release

        if not released:
            if t_us < warmup_us:  # warm-up: hold the trim state while the actuators settle
                set_state(tri, trim_position(t_rel), velTrim, attTrim, np.zeros(3))
            else:                 # release with the perturbed attitude
                set_state(tri, trim_position(t_rel), velTrim, attInit, np.zeros(3))
                released = True

        # offboard controller on the true state; keep the onboard KF on the true state too
        offboardElapsed_us += dt_us
        if offboardElapsed_us > offboardPeriod_us:
            offboardElapsed_us -= offboardPeriod_us
            tri.set_kf_position(tri.position())
            tri.set_kf_velocity(tri.velocity())
            tri.set_kf_attitude(tri.attitude())
            cmdAngVel, msg = command(tri)
            radioQueue.append((t_us + radioDelay_us, msg))

        # deliver commands whose radio delay has elapsed
        while radioQueue and radioQueue[0][0] <= t_us:
            tri.set_command_radio_msg(radioQueue.popleft()[1])

        if not released or t_rel <= 0:
            continue  # warm-up is not logged

        #record state history
        h["time"][index] = t_rel
        h["pos"][index] = tri.position()
        h["posErr"][index] = tri.position() - trim_position(t_rel)
        h["vel"][index] = tri.velocity()
        h["att"][index] = tri.euler_ypr()
        h["angVel"][index] = tri.angular_velocity()
        h["motForces"][index] = tri.motor_forces()
        h["tilt"][index] = tri.tilt_angles()
        h["aileron"][index] = tri.aileron_angles()
        h["tail"][index] = tri.tail_angles()
        h["cmdAngVel"][index] = cmdAngVel
        index += 1
    return h


results = collections.OrderedDict()
for name, make_command in variants.items():
    results[name] = run_variant(make_command)
    h = results[name]
    print("{:15s} final pos err (x, y, z) {} m, speed {:.2f} m/s, yaw/pitch/roll {} deg".format(
        name, np.round(h["posErr"][-1], 3), np.linalg.norm(h["vel"][-1]),
        np.round(np.degrees(h["att"][-1]), 2)))

#==============================================================================
# Make the plots (one colour per variant)
#==============================================================================
deg = 180 / np.pi
panels = [  # (column, row, label, function of history)
    (0, 0, 'x - trim [m]' if forward else 'x err [m]', lambda h: h["posErr"][:, 0]),
    (0, 1, 'y err [m]', lambda h: h["posErr"][:, 1]),
    (0, 2, 'z err [m]', lambda h: h["posErr"][:, 2]),
    (0, 3, 'Speed [m/s]', lambda h: np.linalg.norm(h["vel"], axis=1)),
    (0, 4, 'Roll [deg]', lambda h: h["att"][:, 2] * deg),
    (0, 5, 'Pitch [deg]', lambda h: h["att"][:, 1] * deg),
    (0, 6, 'Yaw [deg]', lambda h: h["att"][:, 0] * deg),
    (1, 0, 'p [rad/s]', lambda h: h["angVel"][:, 0]),
    (1, 1, 'q [rad/s]', lambda h: h["angVel"][:, 1]),
    (1, 2, 'r [rad/s]', lambda h: h["angVel"][:, 2]),
    (1, 3, 'Motor forces\n0+1+2 [N]', lambda h: h["motForces"].sum(axis=1)),
    (1, 4, 'Tilt (right)\n[deg]', lambda h: h["tilt"][:, 0] * deg),
    (1, 5, 'Aileron (right)\n[deg]', lambda h: h["aileron"][:, 0] * deg),
    (1, 6, 'Ruddervator\n(right) [deg]', lambda h: h["tail"][:, 0] * deg),
]
trimLines = {(0, 3): np.linalg.norm(velTrim), (0, 4): yprTrim[2] * deg,
             (0, 5): yprTrim[1] * deg, (0, 6): yprTrim[0] * deg}

fig, ax = plt.subplots(7, 2, sharex=True, figsize=(13, 14))
fig.suptitle("simulateC: {} mode, controller variants".format(args.mode))
for col, row, label, f in panels:
    a = ax[row, col]
    for i, (name, h) in enumerate(results.items()):
        a.plot(h["time"], f(h), color='C{}'.format(i), label=name, lw=1)
    if (col, row) in trimLines:
        a.axhline(trimLines[(col, row)], ls='--', color='k', lw=0.8)
    a.set_ylabel(label)
ax[0, 0].legend(loc='upper right', fontsize=7)
for col in range(2):
    ax[-1, col].set_xlabel('Time [s]')
    ax[0, col].set_xlim([0, endTime])
fig.tight_layout()

plt.show()

#==============================================================================
# Save the data (data/trirotor_data.csv; one block of rows per variant)
#==============================================================================
frames = []
for name, h in results.items():
    frames.append(pd.DataFrame({
        'Variant': name,
        'Time': h["time"].flatten(),
        'PosX': h["pos"][:, 0],
        'PosY': h["pos"][:, 1],
        'PosZ': h["pos"][:, 2],
        'VelX': h["vel"][:, 0],
        'VelY': h["vel"][:, 1],
        'VelZ': h["vel"][:, 2],
        'Yaw': h["att"][:, 0],
        'Pitch': h["att"][:, 1],
        'Roll': h["att"][:, 2],
    }))
data_path = os.path.join(REPO_DIR, 'data', 'trirotor_data.csv')
pd.concat(frames).to_csv(data_path, index=False)
print("Saved {}".format(data_path))

#==============================================================================
# Animate the first variant (hover only: the animation's fixed +/-2 m view can't follow forward
# flight)
#==============================================================================
if forward and not args.no_animate:
    print("Animation skipped in forward mode (fixed +/-2 m view).")
elif not args.no_animate:
    from pyplot3d.utils import ypr_to_R
    from animate import animate_quadcopter_history

    data = pd.read_csv(data_path)
    data = data[data['Variant'] == next(iter(results))]
    times = data['Time'].values
    posHistory = data[['PosX', 'PosY', 'PosZ']].values
    attHistory = data[['Yaw', 'Pitch', 'Roll']].values
    x = posHistory.T
    steps = len(times)

    Rm = np.zeros((3, 3, steps))
    for i in range(steps):
        Rm[:, :, i] = ypr_to_R(attHistory[i, :], degrees=False)

    animate_quadcopter_history(times, x, Rm, arm_length=0.4)
