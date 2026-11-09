// pybind11 bindings GeneralCode to Python as module `GeneralCode`.
// inheriting the function of trirotor simulation and controller from GeneralCode, 
// to use in the learning environment
// Vectors go in/out as numpy arrays; quaternions are (w, x, y, z).
#include <array>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>

#include "Common/DataTypes/RadioTypes.hpp"
#include "Common/Math/Rotation.hpp"
#include "Common/Math/Vec3.hpp"
#include "Common/Time/ManualTimer.hpp"
#include "Components/Logic/DarpaConstants.hpp"
#include "Components/Logic/PX4FixedWingAngularVelocityController.hpp"
#include "Components/Logic/PX4FixedWingAttitudeController.hpp"
#include "Components/Logic/QuadcopterConstants.hpp"
#include "Components/Logic/TrirotorMixer.hpp"
#include "Components/Offboard/CoordinatedFlightConditionSolver.hpp"
#include "Components/Offboard/LinearAerodynamicModel.hpp"
#include "Components/Offboard/QuadcopterController.hpp"
#include "Components/Offboard/TrirotorController.hpp"
#include "Components/Offboard/TrirotorFixedWingController.hpp"
#include "Components/Simulation/Trirotor_T.hpp"

namespace py = pybind11;

using Consts = Onboard::QuadcopterConstants;
using AeroCoeffs = Consts::AerodynamicCoefficients;
using RawMessage = RadioTypes::RadioMessageDecoded::RawMessage;

namespace {
Vec3d V(Eigen::Vector3d const &v) { return Vec3d(v.x(), v.y(), v.z()); }
Eigen::Vector3d E(Vec3d const &v) { return Eigen::Vector3d(v.x, v.y, v.z); }
Rotationd Q(Eigen::Vector4d const &q) {
  Rotationd r(q[0], q[1], q[2], q[3]);
  r.Normalise();
  return r;
}
Eigen::Vector4d Q(Rotationd const &r) { return Eigen::Vector4d(r[0], r[1], r[2], r[3]); }
Vec3f Vf(Eigen::Vector3d const &v) { return Vec3f(float(v.x()), float(v.y()), float(v.z())); }
Eigen::Vector3d Ef(Vec3f const &v) { return Eigen::Vector3d(v.x, v.y, v.z); }
Rotationf Qf(Eigen::Vector4d const &q) { return Rotationf(Q(q)); }
// inverse of ServoCalibration::CmdFromNormalized
float NormalizedFromCmd(Consts::ServoCalibration const &s, float cmd) {
  float const offset = cmd - s.zeroCmd;
  float const toMax = s.maxCmd - s.zeroCmd;
  if (offset * toMax >= 0.0f) {
    return toMax != 0.0f ? offset / toMax : 0.0f;
  }
  float const toMin = s.minCmd - s.zeroCmd;
  return toMin != 0.0f ? -offset / toMin : 0.0f;
}
}  // namespace

// float[2] struct members as read/write 2-element lists
#define ARRAY2_FIELD(cls, type, name)                                              \
  def_property(                                                                    \
      #name, [](type const &s) { return std::array<float, 2>{s.name[0], s.name[1]}; }, \
      [](type &s, std::array<float, 2> const &v) {                                 \
        s.name[0] = v[0];                                                          \
        s.name[1] = v[1];                                                          \
      })

PYBIND11_MODULE(GeneralCode, m) {
  m.doc() = "GeneralCode (darpa-trirotor) trirotor simulator and controllers";

  // ---------------------------------------------------------------- time
  py::class_<ManualTimer>(m, "ManualTimer")
      .def(py::init<>())
      .def("reset_us", &ManualTimer::ResetMicroseconds, py::arg("t_us"))
      .def("advance_us", &ManualTimer::AdvanceMicroSeconds, py::arg("dt_us"))
      .def("micro_seconds", &ManualTimer::GetMicroSeconds)
      .def("seconds", [](ManualTimer const &t) { return t.GetSeconds<double>(); });

  // ---------------------------------------------------------------- constants
  py::class_<Consts::ServoDynamics>(m, "ServoDynamics")
      .def_readwrite("commandDelay", &Consts::ServoDynamics::commandDelay)
      .def_readwrite("slewRate", &Consts::ServoDynamics::slewRate)
      .def_readwrite("timeConstant", &Consts::ServoDynamics::timeConstant)
      .def_readwrite("Kp", &Consts::ServoDynamics::Kp);

  py::class_<Consts::WingParameters>(m, "WingParameters")
      .def_readwrite("S", &Consts::WingParameters::S)
      .def_readwrite("b", &Consts::WingParameters::b)
      .def_readwrite("AR", &Consts::WingParameters::AR)
      .def_readwrite("c_root", &Consts::WingParameters::c_root)
      .def_readwrite("c_tip", &Consts::WingParameters::c_tip)
      .def_readwrite("taperRatio", &Consts::WingParameters::taperRatio)
      .def_readwrite("c_mean", &Consts::WingParameters::c_mean);

  py::class_<AeroCoeffs>(m, "AerodynamicCoefficients")
      .def_readwrite("CL_0", &AeroCoeffs::CL_0)
      .def_readwrite("CL_alpha", &AeroCoeffs::CL_alpha)
      .def_readwrite("CL_q", &AeroCoeffs::CL_q)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, CL_delta_aileron)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, CL_delta_tail)
      .def_readwrite("CD_0", &AeroCoeffs::CD_0)
      .def_readwrite("CD_q", &AeroCoeffs::CD_q)
      .def_readwrite("stallAlpha", &AeroCoeffs::stallAlpha)
      .def_readwrite("stallBlendWidth", &AeroCoeffs::stallBlendWidth)
      .def_readwrite("CL_stall_max", &AeroCoeffs::CL_stall_max)
      .def_readwrite("CD_stall_max", &AeroCoeffs::CD_stall_max)
      .def_readwrite("postStallControlEffectiveness", &AeroCoeffs::postStallControlEffectiveness)
      .def_readwrite("Cm_0", &AeroCoeffs::Cm_0)
      .def_readwrite("Cm_alpha", &AeroCoeffs::Cm_alpha)
      .def_readwrite("Cm_q", &AeroCoeffs::Cm_q)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, Cm_delta_aileron)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, Cm_delta_tail)
      .def_readwrite("CY_0", &AeroCoeffs::CY_0)
      .def_readwrite("CY_beta", &AeroCoeffs::CY_beta)
      .def_readwrite("CY_p", &AeroCoeffs::CY_p)
      .def_readwrite("CY_r", &AeroCoeffs::CY_r)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, CY_delta_aileron)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, CY_delta_tail)
      .def_readwrite("Cl_0", &AeroCoeffs::Cl_0)
      .def_readwrite("Cl_beta", &AeroCoeffs::Cl_beta)
      .def_readwrite("Cl_p", &AeroCoeffs::Cl_p)
      .def_readwrite("Cl_r", &AeroCoeffs::Cl_r)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, Cl_delta_aileron)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, Cl_delta_tail)
      .def_readwrite("Cn_0", &AeroCoeffs::Cn_0)
      .def_readwrite("Cn_beta", &AeroCoeffs::Cn_beta)
      .def_readwrite("Cn_p", &AeroCoeffs::Cn_p)
      .def_readwrite("Cn_r", &AeroCoeffs::Cn_r)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, Cn_delta_aileron)
      .ARRAY2_FIELD(AeroCoeffs, AeroCoeffs, Cn_delta_tail);

  using FWConsts = Consts::FixedWingControllerConstants;
  py::class_<FWConsts>(m, "FixedWingControllerConstants")
      .def_readwrite("posControl_natFreq_x", &FWConsts::posControl_natFreq_x)
      .def_readwrite("posControl_natFreq_y", &FWConsts::posControl_natFreq_y)
      .def_readwrite("posControl_natFreq_z", &FWConsts::posControl_natFreq_z)
      .def_readwrite("posControl_damping_x", &FWConsts::posControl_damping_x)
      .def_readwrite("posControl_damping_y", &FWConsts::posControl_damping_y)
      .def_readwrite("posControl_damping_z", &FWConsts::posControl_damping_z)
      .def_readwrite("attControl_timeConst_x", &FWConsts::attControl_timeConst_x)
      .def_readwrite("attControl_timeConst_y", &FWConsts::attControl_timeConst_y)
      .def_readwrite("attControl_timeConst_z", &FWConsts::attControl_timeConst_z)
      .def_readwrite("attControl_integralGain", &FWConsts::attControl_integralGain)
      .def_readwrite("attControl_integralLimit_x", &FWConsts::attControl_integralLimit_x)
      .def_readwrite("attControl_integralLimit_y", &FWConsts::attControl_integralLimit_y)
      .def_readwrite("attControl_integralLimit_z", &FWConsts::attControl_integralLimit_z)
      .def_readwrite("angVelControl_timeConst_x", &FWConsts::angVelControl_timeConst_x)
      .def_readwrite("angVelControl_timeConst_y", &FWConsts::angVelControl_timeConst_y)
      .def_readwrite("angVelControl_timeConst_z", &FWConsts::angVelControl_timeConst_z)
      .def_readwrite("angVelControl_Ki_x", &FWConsts::angVelControl_Ki_x)
      .def_readwrite("angVelControl_Ki_y", &FWConsts::angVelControl_Ki_y)
      .def_readwrite("angVelControl_Ki_z", &FWConsts::angVelControl_Ki_z)
      .def_readwrite("angVelControl_useIntegral", &FWConsts::angVelControl_useIntegral);

  using TecsConsts = Consts::TECSControllerConstants;
  py::class_<TecsConsts>(m, "TECSControllerConstants")
      .def_readwrite("throttle_Kp", &TecsConsts::throttle_Kp)
      .def_readwrite("throttle_Ki", &TecsConsts::throttle_Ki)
      .def_readwrite("pitch_Kp", &TecsConsts::pitch_Kp)
      .def_readwrite("pitch_Ki", &TecsConsts::pitch_Ki)
      .def_readwrite("energyDistributionWeight", &TecsConsts::energyDistributionWeight)
      .def_readwrite("heightRateTimeConst", &TecsConsts::heightRateTimeConst)
      .def_readwrite("maxClimbRateDemand", &TecsConsts::maxClimbRateDemand)
      .def_readwrite("minClimbRateDemand", &TecsConsts::minClimbRateDemand)
      .def_readwrite("maxAirspeedDemandRate", &TecsConsts::maxAirspeedDemandRate)
      .def_readwrite("minCruiseAirspeed", &TecsConsts::minCruiseAirspeed)
      .def_readwrite("maxCruiseAirspeed", &TecsConsts::maxCruiseAirspeed)
      .def_readwrite("minPitchAngle", &TecsConsts::minPitchAngle)
      .def_readwrite("maxPitchAngle", &TecsConsts::maxPitchAngle)
      .def_readwrite("minThrust", &TecsConsts::minThrust)
      .def_readwrite("maxThrust", &TecsConsts::maxThrust)
      .def_readwrite("throttleTrim", &TecsConsts::throttleTrim)
      .def_readwrite("pitchTrim", &TecsConsts::pitchTrim)
      .def_readwrite("throttleIntegratorLimit", &TecsConsts::throttleIntegratorLimit)
      .def_readwrite("pitchIntegratorLimit", &TecsConsts::pitchIntegratorLimit)
      .def_readwrite("loadFactorCompensationGain", &TecsConsts::loadFactorCompensationGain);

  using TecsOut = Onboard::TrirotorTecsController::TecsOutput;
  py::class_<TecsOut>(m, "TecsOutput")
      .def_readonly("thrust", &TecsOut::thrust)            // [N]
      .def_readonly("pitchAngle", &TecsOut::pitchAngle)    // [rad]
      .def_readonly("heightDemandFiltered", &TecsOut::heightDemandFiltered)
      .def_readonly("airspeedDemandFiltered", &TecsOut::airspeedDemandFiltered)
      .def_readonly("speError", &TecsOut::speError)
      .def_readonly("skeError", &TecsOut::skeError)
      .def_readonly("steError", &TecsOut::steError)
      .def_readonly("sebError", &TecsOut::sebError)
      .def_readonly("climbRateDemand", &TecsOut::climbRateDemand)
      .def_readonly("throttleIntegrator", &TecsOut::throttleIntegrator)
      .def_readonly("pitchIntegrator", &TecsOut::pitchIntegrator);

  using LGConsts = Consts::LateralGuidanceConstants;
  py::class_<LGConsts>(m, "LateralGuidanceConstants")
      .def_readwrite("maxBankAngle", &LGConsts::maxBankAngle)
      .def_readwrite("minHeadingSpeed", &LGConsts::minHeadingSpeed);

  // inertia_xx/yy/zz setters also update inertiaMatrix (used onboard), as the reference sim does.
  auto inertiaProp = [](int i) {
    return std::make_pair(
        [i](Consts const &c) { return i == 0 ? c.inertia_xx : (i == 1 ? c.inertia_yy : c.inertia_zz); },
        [i](Consts &c, float v) {
          (i == 0 ? c.inertia_xx : (i == 1 ? c.inertia_yy : c.inertia_zz)) = v;
          c.inertiaMatrix(i, i) = v;
        });
  };
  auto ixx = inertiaProp(0), iyy = inertiaProp(1), izz = inertiaProp(2);

  py::class_<Consts>(m, "QuadcopterConstants")
      .def(py::init([](unsigned vehicleId) {
             return Consts(Consts::GetVehicleTypeFromID(vehicleId));
           }),
           py::arg("vehicle_id"))
      .def("copy", [](Consts const &c) { return Consts(c); })
      // control surfaces, (aileron R, L) and (tail R, L): normalized deflection in [-1, 1] <-> raw servo cmd
      .def("surface_cmds_from_normalized",
           [](Consts const &c, std::array<float, 2> const &aileron, std::array<float, 2> const &tail) {
             return py::make_tuple(
                 std::array<float, 2>{c.ailerons[0].CmdFromNormalized(aileron[0]),
                                      c.ailerons[1].CmdFromNormalized(aileron[1])},
                 std::array<float, 2>{c.tailSurfaces[0].CmdFromNormalized(tail[0]),
                                      c.tailSurfaces[1].CmdFromNormalized(tail[1])});
           },
           py::arg("aileron"), py::arg("tail"))
      .def("surface_normalized_from_cmds",
           [](Consts const &c, std::array<float, 2> const &aileron, std::array<float, 2> const &tail) {
             return py::make_tuple(
                 std::array<float, 2>{NormalizedFromCmd(c.ailerons[0], aileron[0]),
                                      NormalizedFromCmd(c.ailerons[1], aileron[1])},
                 std::array<float, 2>{NormalizedFromCmd(c.tailSurfaces[0], tail[0]),
                                      NormalizedFromCmd(c.tailSurfaces[1], tail[1])});
           },
           py::arg("aileron_cmds"), py::arg("tail_cmds"))
      .def_readwrite("mass", &Consts::mass)
      .def_property("inertia_xx", ixx.first, ixx.second)
      .def_property("inertia_yy", iyy.first, iyy.second)
      .def_property("inertia_zz", izz.first, izz.second)
      .def_readwrite("dx", &Consts::dx)
      .def_readwrite("dy1", &Consts::dy1)
      .def_readwrite("dy2", &Consts::dy2)
      .def_readwrite("dz", &Consts::dz)
      .def_readwrite("propellerDiameter", &Consts::propellerDiameter)
      .def_readwrite("propellerThrustFromSpeedSqr", &Consts::propellerThrustFromSpeedSqr)
      .def_readwrite("propellerTorqueFromThrust", &Consts::propellerTorqueFromThrust)
      .def_readwrite("motorTimeConst", &Consts::motorTimeConst)
      .def_readwrite("motorInertia", &Consts::motorInertia)
      .def_readwrite("motorMinSpeed", &Consts::motorMinSpeed)
      .def_readwrite("motorMaxSpeed", &Consts::motorMaxSpeed)
      .def_readwrite("linDragCoeffBx", &Consts::linDragCoeffBx)
      .def_readwrite("linDragCoeffBy", &Consts::linDragCoeffBy)
      .def_readwrite("linDragCoeffBz", &Consts::linDragCoeffBz)
      .def_readwrite("aeroRefPointToComOffset_x", &Consts::aeroRefPointToComOffset_x)
      .def_readwrite("aeroRefPointToComOffset_y", &Consts::aeroRefPointToComOffset_y)
      .def_readwrite("aeroRefPointToComOffset_z", &Consts::aeroRefPointToComOffset_z)
      .def_readwrite("posControl_natFreq", &Consts::posControl_natFreq)
      .def_readwrite("posControl_damping", &Consts::posControl_damping)
      .def_readwrite("attControl_timeConst_x", &Consts::attControl_timeConst_x)
      .def_readwrite("attControl_timeConst_y", &Consts::attControl_timeConst_y)
      .def_readwrite("attControl_timeConst_xy", &Consts::attControl_timeConst_xy)
      .def_readwrite("attControl_timeConst_z", &Consts::attControl_timeConst_z)
      .def_readwrite("attControl_integralGain", &Consts::attControl_integralGain)
      .def_readwrite("attControl_integralLimit_x", &Consts::attControl_integralLimit_x)
      .def_readwrite("attControl_integralLimit_y", &Consts::attControl_integralLimit_y)
      .def_readwrite("attControl_integralLimit_z", &Consts::attControl_integralLimit_z)
      .def_readwrite("tiltServoDynamics", &Consts::tiltServoDynamics)
      .def_readwrite("aileronServoDynamics", &Consts::aileronServoDynamics)
      .def_readwrite("tailServoDynamics", &Consts::tailServoDynamics)
      .def_readwrite("maxThrustPerPropeller", &Consts::maxThrustPerPropeller)
      .def_readwrite("stabilizeMaxRoll", &Consts::stabilizeMaxRoll)
      .def_readwrite("stabilizeMaxPitch", &Consts::stabilizeMaxPitch)
      .def_readwrite("stabilizeMaxYawRate", &Consts::stabilizeMaxYawRate)
      .def_readwrite("fixedWingControlConsts", &Consts::fixedWingControlConsts)
      .def_readwrite("lateralGuidanceConsts", &Consts::lateralGuidanceConsts)
      .def_readwrite("tecsControlConsts", &Consts::tecsControlConsts)
      .def_readwrite("wingParams", &Consts::wingParams)
      .def_readwrite("aeroCoeffs", &Consts::aeroCoeffs)
      .def_readwrite("simAeroCoeffs", &Consts::simAeroCoeffs);

  m.attr("AIR_DENSITY") = Onboard::DarpaConstants(Onboard::DarpaConstants::Trirotor).airDensity;

  // ---------------------------------------------------------------- aero model
  py::class_<Offboard::LinearAerodynamicModel>(m, "LinearAerodynamicModel")
      .def(py::init([](double mass, double airDensity, Consts::WingParameters const &wing,
                       AeroCoeffs const &coeffs, Eigen::Vector3d const &aeroRefPointToComOffset) {
             return Offboard::LinearAerodynamicModel(mass, airDensity, wing, coeffs,
                                                     V(aeroRefPointToComOffset));
           }),
           py::arg("mass"), py::arg("air_density"), py::arg("wing_params"), py::arg("coeffs"),
           py::arg("aero_ref_point_to_com_offset"));

  // ---------------------------------------------------------------- radio
  py::class_<RawMessage>(m, "RadioMessage");
  m.attr("FLAG_DISABLE_ONBOARD_STATE_SAFETY_CHECKS") =
      int(RadioTypes::ReservedFlags::disableOnboardStateSafetyChecks);
  m.def("make_tiltrotor_command",
        [](int flags, float verticalThrust, float forwardThrust, Eigen::Vector3d const &angVel,
           std::array<float, 2> aileronAngles, std::array<float, 2> tailAngles) {
          RawMessage msg;
          RadioTypes::RadioMessageDecoded::CreateTrirotorTiltrotorCommand(
              uint8_t(flags), verticalThrust, forwardThrust,
              Vec3f(float(angVel.x()), float(angVel.y()), float(angVel.z())),
              aileronAngles.data(), tailAngles.data(), msg.raw);
          return msg;
        },
        py::arg("flags"), py::arg("vertical_thrust"), py::arg("forward_thrust"), py::arg("ang_vel"),
        py::arg("aileron_angles") = std::array<float, 2>{0, 0},
        py::arg("tail_angles") = std::array<float, 2>{0, 0});
  // total mass-normalized thrust + body rates -> onboard external-rates mode (FS_EXTERNAL_RATES_CONTROL)
  m.def("make_rates_command",
        [](int flags, float thrust, Eigen::Vector3d const &angVel) {
          RawMessage msg;
          RadioTypes::RadioMessageDecoded::CreateRatesCommand(
              uint8_t(flags), thrust, Vec3f(float(angVel.x()), float(angVel.y()), float(angVel.z())),
              msg.raw);
          return msg;
        },
        py::arg("flags"), py::arg("thrust"), py::arg("ang_vel"));
  m.def("make_forward_wing_command",
        [](int flags, float forwardThrust, Eigen::Vector3d const &angVel) {
          RawMessage msg;
          RadioTypes::RadioMessageDecoded::CreateTrirotorForwardWingCommand(
              uint8_t(flags), forwardThrust,
              Vec3f(float(angVel.x()), float(angVel.y()), float(angVel.z())), msg.raw);
          return msg;
        },
        py::arg("flags"), py::arg("forward_thrust"), py::arg("ang_vel"));
  // direct actuator command (FS_TRIMOTOR_SPEEDS_CONTROL): motor speeds [rad/s] and servo values ordered
  // (tilt R, tilt L, aileron R, aileron L, tail R, tail L). With FLAG_SPEEDS_RAW_SERVO_CMDS these are
  // raw servo commands, otherwise angles [rad].
  m.attr("FLAG_SPEEDS_RAW_SERVO_CMDS") = int(RadioTypes::RadioMessageDecoded::SPEEDS_FLAG_RAW_SERVO_CMDS);
  m.def("make_speeds_command",
        [](int flags, Eigen::Vector3d const &motorSpeeds, std::array<float, 6> const &servos) {
          RawMessage msg;
          float const speeds[3] = {float(motorSpeeds[0]), float(motorSpeeds[1]), float(motorSpeeds[2])};
          uint16_t const dshot[3] = {0, 0, 0};
          RadioTypes::RadioMessageDecoded::CreateTrimotorSpeedsCommand(uint8_t(flags), speeds, servos.data(),
                                                                       dshot, msg.raw);
          return msg;
        },
        py::arg("flags"), py::arg("motor_speeds"), py::arg("servos"));
  // attitude setpoint + total thrust [N] -> onboard PX4 attitude/rate loops (FS_TRIMOTOR_ATTITUDE_CONTROL)
  m.def("make_attitude_command",
        [](int flags, float thrust, Eigen::Vector4d const &att) {
          RawMessage msg;
          Rotationd const q = Q(att);
          float const desAtt[4] = {float(q[0]), float(q[1]), float(q[2]), float(q[3])};
          RadioTypes::RadioMessageDecoded::CreateTrirotorAttitudeCommand(uint8_t(flags), thrust,
                                                                       desAtt, msg.raw);
          return msg;
        },
        py::arg("flags"), py::arg("thrust"), py::arg("att"));

  // ---------------------------------------------------------------- vehicle (physics + onboard logic)
  py::class_<Simulation::Trirotor>(m, "Trirotor")
      .def(py::init([](ManualTimer *timer, double mass, Eigen::Vector3d const &inertiaDiag,
                       double dx, double dy1, double dy2, double dz,
                       Eigen::Vector3d const &centreOfMassError, double motorMinSpeed,
                       double motorMaxSpeed, double propThrustFromSpeedSqr,
                       double propTorqueFromSpeedSqr, double motorTimeConst, double motorInertia,
                       double servoSlewRate, double servoTimeConstant, double servoKp,
                       double servoCommandDelay, Eigen::Vector3d const &linDragCoeffB,
                       Offboard::LinearAerodynamicModel const &aeroModel, unsigned vehicleId,
                       double onboardLogicPeriod, Consts const *constantsOverride) {
             Eigen::Matrix<double, 3, 3> inertia = inertiaDiag.asDiagonal();
             return new Simulation::Trirotor(
                 timer, mass, inertia, dx, dy1, dy2, dz, V(centreOfMassError), motorMinSpeed,
                 motorMaxSpeed, propThrustFromSpeedSqr, propTorqueFromSpeedSqr, motorTimeConst,
                 motorInertia, servoSlewRate, servoTimeConstant, servoKp, servoCommandDelay,
                 V(linDragCoeffB), aeroModel, uint8_t(vehicleId),
                 Consts::GetVehicleTypeFromID(vehicleId), Onboard::DarpaConstants::Trirotor,
                 onboardLogicPeriod, constantsOverride);
           }),
           py::arg("timer"), py::arg("mass"), py::arg("inertia_diag"), py::arg("dx"),
           py::arg("dy1"), py::arg("dy2"), py::arg("dz"), py::arg("centre_of_mass_error"),
           py::arg("motor_min_speed"), py::arg("motor_max_speed"),
           py::arg("prop_thrust_from_speed_sqr"), py::arg("prop_torque_from_speed_sqr"),
           py::arg("motor_time_const"), py::arg("motor_inertia"), py::arg("servo_slew_rate"),
           py::arg("servo_time_constant"), py::arg("servo_kp"), py::arg("servo_command_delay"),
           py::arg("lin_drag_coeff_b"), py::arg("aero_model"), py::arg("vehicle_id"),
           py::arg("onboard_logic_period"), py::arg("constants_override") = nullptr,
           py::keep_alive<1, 2>())  // the vehicle keeps a pointer to the timer
      .def("run", &Simulation::Trirotor::Run)
      // state (SimulationObject6DOF setters, avoiding Trirotor_T's printing overrides)
      .def("position", [](Simulation::Trirotor const &t) { return E(t.GetPosition()); })
      .def("velocity", [](Simulation::Trirotor const &t) { return E(t.GetVelocity()); })
      .def("attitude", [](Simulation::Trirotor const &t) { return Q(t.GetAttitude()); })
      .def("euler_ypr", [](Simulation::Trirotor const &t) { return E(t.GetAttitude().ToEulerYPR()); })
      .def("angular_velocity", [](Simulation::Trirotor const &t) { return E(t.GetAngularVelocity()); })
      .def("set_position", [](Simulation::Trirotor &t, Eigen::Vector3d const &v) { t.SetPosition(V(v)); })
      .def("set_velocity",
           [](Simulation::Trirotor &t, Eigen::Vector3d const &v) { t.SimulationObject6DOF::SetVelocity(V(v)); })
      .def("set_attitude", [](Simulation::Trirotor &t, Eigen::Vector4d const &q) { t.SetAttitude(Q(q)); })
      .def("set_angular_velocity", [](Simulation::Trirotor &t, Eigen::Vector3d const &v) {
        t.SimulationObject6DOF::SetAngularVelocity(V(v));
      })
      // onboard estimator seeding
      .def("set_kf_position", [](Simulation::Trirotor &t, Eigen::Vector3d const &v) { t.SetKFPosition(V(v)); })
      .def("set_kf_velocity", [](Simulation::Trirotor &t, Eigen::Vector3d const &v) { t.SetKFVelocity(V(v)); })
      .def("set_kf_attitude", [](Simulation::Trirotor &t, Eigen::Vector4d const &q) { t.SetKFAttitude(Q(q)); })
      .def("set_kf_angular_velocity",
           [](Simulation::Trirotor &t, Eigen::Vector3d const &v) { t.SetKFAngularVelocity(V(v)); })
      // commands and disturbances
      .def("set_command_radio_msg", &Simulation::Trirotor::SetCommandRadioMsg, py::arg("msg"))
      .def("set_wind", [](Simulation::Trirotor &t, Eigen::Vector3d const &v) { t.SetWindVector(V(v)); })
      .def("set_external_force", [](Simulation::Trirotor &t, Eigen::Vector3d const &v) { t.SetExternalForce(V(v)); })
      .def("set_external_torque", [](Simulation::Trirotor &t, Eigen::Vector3d const &v) { t.SetExternalTorque(V(v)); })
      // actuators
      .def("motor_forces", [](Simulation::Trirotor const &t) {
        return Eigen::Vector3d(t.GetMotorForce(0), t.GetMotorForce(1), t.GetMotorForce(2));
      })
      .def("motor_thrust_cmds", [](Simulation::Trirotor const &t) {
        return Eigen::Vector3d(t.GetMotorThrustCmd(0), t.GetMotorThrustCmd(1), t.GetMotorThrustCmd(2));
      })
      .def("tilt_angles", [](Simulation::Trirotor const &t) {
        return Eigen::Vector2d(t.GetSimMotorAngle(0), t.GetSimMotorAngle(1));
      })
      .def("aileron_angles", [](Simulation::Trirotor const &t) {  // actual (R, L) [rad]
        return Eigen::Vector2d(t.GetSimAileronAngle(0), t.GetSimAileronAngle(1));
      })
      .def("tail_angles", [](Simulation::Trirotor const &t) {  // actual (R, L) ruddervators [rad]
        return Eigen::Vector2d(t.GetSimTailSurfaceAngle(0), t.GetSimTailSurfaceAngle(1));
      });

  // ---------------------------------------------------------------- offboard controller
  py::class_<Offboard::TrirotorController>(m, "TrirotorController")
      .def(py::init<Offboard::LinearAerodynamicModel>(), py::arg("aero_model"))
      .def("set_parameters", &Offboard::TrirotorController::SetParameters,
           py::arg("pos_nat_freq"), py::arg("pos_damping"), py::arg("att_time_const_x"),
           py::arg("att_time_const_y"), py::arg("att_time_const_z"), py::arg("att_integral_gain"),
           py::arg("att_integral_limit_x"), py::arg("att_integral_limit_y"),
           py::arg("att_integral_limit_z"))
      // returns (cmd_ang_vel, cmd_vertical_thrust, cmd_forward_thrust)
      .def("run",
           [](Offboard::TrirotorController const &c, Eigen::Vector3d const &pos,
              Eigen::Vector3d const &vel, Eigen::Vector4d const &att, Eigen::Vector3d const &desPos,
              Eigen::Vector3d const &desVel, Eigen::Vector3d const &desAcc, double desPitch,
              double desYaw, double dt) {
             Vec3d angVel(0, 0, 0);
             double verticalThrust = 0, forwardThrust = 0;
             c.Run(V(pos), V(vel), Q(att), V(desPos), V(desVel), V(desAcc), desPitch, desYaw,
                   angVel, verticalThrust, forwardThrust, float(dt));
             return py::make_tuple(E(angVel), verticalThrust, forwardThrust);
           },
           py::arg("pos"), py::arg("vel"), py::arg("att"), py::arg("des_pos"), py::arg("des_vel"),
           py::arg("des_acc"), py::arg("des_pitch"), py::arg("des_yaw"), py::arg("dt"));

  // Generic multirotor position/attitude controller (the reference sim's hover branch with
  // forwardControl = false): returns (cmd_ang_vel, cmd_thrust) -> make_rates_command
  py::class_<Offboard::QuadcopterController>(m, "QuadcopterController")
      .def(py::init<>())
      .def("set_parameters", &Offboard::QuadcopterController::SetParameters,
           py::arg("pos_nat_freq"), py::arg("pos_damping"), py::arg("att_time_const_xy"),
           py::arg("att_time_const_z"))
      .def("run",
           [](Offboard::QuadcopterController const &c, Eigen::Vector3d const &pos,
              Eigen::Vector3d const &vel, Eigen::Vector4d const &att, Eigen::Vector3d const &desPos,
              Eigen::Vector3d const &desVel, Eigen::Vector3d const &desAcc, double desYaw) {
             Vec3d angVel(0, 0, 0);
             double thrust = 0;
             c.Run(V(pos), V(vel), Q(att), V(desPos), V(desVel), V(desAcc), desYaw, angVel, thrust);
             return py::make_tuple(E(angVel), thrust);
           },
           py::arg("pos"), py::arg("vel"), py::arg("att"), py::arg("des_pos"), py::arg("des_vel"),
           py::arg("des_acc"), py::arg("des_yaw"));

  // ---------------------------------------------------------------- flight condition solver
  using FCS = Offboard::CoordinatedFlightConditionSolver;
  py::enum_<FCS::SolveStatus>(m, "SolveStatus")
      .value("OK", FCS::SOLVE_OK)                  // exact solution inside the AoA limits
      .value("NO_BRACKET", FCS::SOLVE_NO_BRACKET)  // not achievable inside the AoA limits, best effort
      .value("DEGENERATE", FCS::SOLVE_DEGENERATE)  // airspeed ~0: identity attitude, zero thrust
      .export_values();

  py::class_<FCS::FlightCondition>(m, "FlightCondition")
      .def_property_readonly("att", [](FCS::FlightCondition const &f) { return Q(f.desAtt); })  // (w,x,y,z)
      .def_readonly("alpha", &FCS::FlightCondition::alpha)  // angle of attack [rad]
      .def_readonly("T", &FCS::FlightCondition::T)          // thrust
      .def_readonly("have_bracket", &FCS::FlightCondition::have_bracket)
      .def_readonly("status", &FCS::FlightCondition::status);

  // Attitude + thrust for a given air-relative velocity and required proper acceleration (incl. the
  // +9.81 z gravity offset, as TrirotorFixedWingController passes it).
  py::class_<FCS>(m, "CoordinatedFlightConditionSolver")
      .def(py::init<Offboard::LinearAerodynamicModel>(), py::arg("aero_model"))
      .def("solve",
           [](FCS const &s, Eigen::Vector3d const &airVel, Eigen::Vector3d const &properAcc) {
             return s.SolveFlightConditionSolver(V(airVel), V(properAcc));
           },
           py::arg("air_velocity"), py::arg("proper_acc"))
      .def("T_par", &FCS::GetTpar)
      .def("T_perp", &FCS::GetTperp);

  // ---------------------------------------------------------------- forward flight (offboard)
  py::class_<Offboard::TrirotorFixedWingController>(m, "TrirotorFixedWingController")
      .def(py::init<Offboard::LinearAerodynamicModel>(), py::arg("aero_model"))
      .def("set_parameters", &Offboard::TrirotorFixedWingController::SetParameters,
           py::arg("pos_nat_freq_x"), py::arg("pos_nat_freq_y"), py::arg("pos_nat_freq_z"),
           py::arg("pos_damping_x"), py::arg("pos_damping_y"), py::arg("pos_damping_z"),
           py::arg("att_time_const_x"), py::arg("att_time_const_y"), py::arg("att_time_const_z"),
           py::arg("att_integral_gain"), py::arg("att_integral_limit_x"),
           py::arg("att_integral_limit_y"), py::arg("att_integral_limit_z"))
      .def("set_lateral_guidance_parameters",
           &Offboard::TrirotorFixedWingController::SetLateralGuidanceParameters,
           py::arg("max_bank_angle"), py::arg("min_heading_speed"))
      .def("set_tecs_parameters", &Offboard::TrirotorFixedWingController::SetTecsParameters,
           py::arg("throttle_Kp"), py::arg("throttle_Ki"), py::arg("pitch_Kp"), py::arg("pitch_Ki"),
           py::arg("energy_distribution_weight"), py::arg("height_rate_time_const"),
           py::arg("max_climb_rate_demand"), py::arg("min_climb_rate_demand"),
           py::arg("max_airspeed_demand_rate"), py::arg("min_pitch_angle"), py::arg("max_pitch_angle"),
           py::arg("min_thrust"), py::arg("max_thrust"), py::arg("throttle_trim"), py::arg("pitch_trim"),
           py::arg("throttle_integrator_limit"), py::arg("pitch_integrator_limit"),
           py::arg("load_factor_compensation_gain"))
      .def("reset_tecs", &Offboard::TrirotorFixedWingController::ResetTecs)
      // TECS (energy loops for thrust/pitch + lateral guidance for bank) + offboard attitude loop:
      // returns (cmd_ang_vel, cmd_forward_thrust). Steady cruise only, not the VTOL transition.
      .def("run_tecs",
           [](Offboard::TrirotorFixedWingController const &c, Eigen::Vector3d const &pos,
              Eigen::Vector3d const &vel, Eigen::Vector4d const &att, Eigen::Vector3d const &desPos,
              Eigen::Vector3d const &desVel, Eigen::Vector3d const &desAcc,
              Eigen::Vector3d const &groundWind, double dt) {
             Vec3d angVel(0, 0, 0);
             double forwardThrust = 0;
             c.RunTecs(V(pos), V(vel), Q(att), V(desPos), V(desVel), V(desAcc), V(groundWind), angVel,
                       forwardThrust, float(dt));
             return py::make_tuple(E(angVel), forwardThrust);
           },
           py::arg("pos"), py::arg("vel"), py::arg("att"), py::arg("des_pos"), py::arg("des_vel"),
           py::arg("des_acc"), py::arg("ground_wind"), py::arg("dt"))
      // TECS without the attitude loop: returns (att_setpoint (w,x,y,z), thrust [N]) -> make_attitude_command
      .def("run_tecs_attitude_setpoint",
           [](Offboard::TrirotorFixedWingController const &c, Eigen::Vector3d const &pos,
              Eigen::Vector3d const &vel, Eigen::Vector4d const &att, Eigen::Vector3d const &desPos,
              Eigen::Vector3d const &desVel, Eigen::Vector3d const &desAcc,
              Eigen::Vector3d const &groundWind, double dt) {
             Rotationd desAtt;
             double thrust = 0;
             c.RunTecsAttitudeSetpoint(V(pos), V(vel), Q(att), V(desPos), V(desVel), V(desAcc),
                                       V(groundWind), desAtt, thrust, float(dt));
             return py::make_tuple(Q(desAtt), thrust);
           },
           py::arg("pos"), py::arg("vel"), py::arg("att"), py::arg("des_pos"), py::arg("des_vel"),
           py::arg("des_acc"), py::arg("ground_wind"), py::arg("dt"))
      .def("tecs_output", &Offboard::TrirotorFixedWingController::GetTecsOutput)
      .def("tecs_cmd_bank_angle", &Offboard::TrirotorFixedWingController::GetTecsCmdBankAngle)
      .def("tecs_cmd_yaw_angle", &Offboard::TrirotorFixedWingController::GetTecsCmdYawAngle)
      .def("set_min_max_acceleration", &Offboard::TrirotorFixedWingController::SetMinMaxAcceleration,
           py::arg("min_vertical_proper_acc"), py::arg("max_proper_acc"), py::arg("min_proper_acc") = -1.0)
      .def("reset_att_integral", &Offboard::TrirotorFixedWingController::ResetAttIntegral)
      .def("att_integral", [](Offboard::TrirotorFixedWingController const &c) { return Ef(c.GetAttIntegral()); })
      // Position control + flight condition solver, no attitude loop (the PX4 attitude loop runs
      // onboard): returns (att_setpoint (w,x,y,z), thrust [N], status) -> make_attitude_command
      .def("run_attitude_setpoint",
           [](Offboard::TrirotorFixedWingController const &c, Eigen::Vector3d const &pos,
              Eigen::Vector3d const &vel, Eigen::Vector3d const &desPos, Eigen::Vector3d const &desVel,
              Eigen::Vector3d const &desAcc, Eigen::Vector3d const &groundWind) {
             Rotationd desAtt;
             double thrust = 0;
             auto const status = c.RunAttitudeSetpoint(V(pos), V(vel), V(desPos), V(desVel), V(desAcc),
                                                       V(groundWind), desAtt, thrust);
             return py::make_tuple(Q(desAtt), thrust, status);
           },
           py::arg("pos"), py::arg("vel"), py::arg("des_pos"), py::arg("des_vel"), py::arg("des_acc"),
           py::arg("ground_wind"))
      // Flight condition solver + offboard attitude loop from a commanded proper acceleration:
      // returns (cmd_ang_vel, cmd_forward_thrust)
      .def("run_accel",
           [](Offboard::TrirotorFixedWingController const &c, Eigen::Vector3d const &cmdProperAcc,
              Eigen::Vector3d const &vel, Eigen::Vector4d const &att, Eigen::Vector3d const &groundWind,
              double dt) {
             Vec3d angVel(0, 0, 0);
             double forwardThrust = 0;
             c.RunAccel(Vf(cmdProperAcc), V(vel), Q(att), V(groundWind), angVel, forwardThrust, float(dt));
             return py::make_tuple(E(angVel), forwardThrust);
           },
           py::arg("cmd_proper_acc"), py::arg("vel"), py::arg("att"), py::arg("ground_wind"), py::arg("dt"))
      // Offboard attitude loop only: returns (cmd_ang_vel, cmd_forward_thrust)
      .def("run_attitude",
           [](Offboard::TrirotorFixedWingController const &c, double thrust, Eigen::Vector4d const &desAtt,
              Eigen::Vector4d const &att, double dt) {
             Vec3d angVel(0, 0, 0);
             double forwardThrust = 0;
             c.RunAttitude(thrust, Q(desAtt), Q(att), angVel, forwardThrust, float(dt));
             return py::make_tuple(E(angVel), forwardThrust);
           },
           py::arg("thrust"), py::arg("des_att"), py::arg("att"), py::arg("dt"))
      // internals of the last run (for logging)
      .def("flight_condition", &Offboard::TrirotorFixedWingController::GetFlightCondition)
      .def("cmd_acc", [](Offboard::TrirotorFixedWingController const &c) { return E(c.GetCmdAcc()); })
      .def("cmd_proper_acc", [](Offboard::TrirotorFixedWingController const &c) { return E(c.GetCmdProperAcc()); })
      .def("cmd_att_yawed", [](Offboard::TrirotorFixedWingController const &c) { return E(c.GetCmdAttYawed()); })
      .def("feedback", [](Offboard::TrirotorFixedWingController const &c) { return E(c.GetFeedback()); })
      .def("cmd_alpha", &Offboard::TrirotorFixedWingController::GetCmdAlpha)
      .def("cmd_thrust", &Offboard::TrirotorFixedWingController::GetCmdThrust)
      .def("solver_T_par", &Offboard::TrirotorFixedWingController::GetFlightConditionSolverTpar)
      .def("solver_T_perp", &Offboard::TrirotorFixedWingController::GetFlightConditionSolverTperp)
      // Full chain incl. offboard attitude loop: returns (cmd_ang_vel, cmd_forward_thrust)
      .def("run",
           [](Offboard::TrirotorFixedWingController const &c, Eigen::Vector3d const &pos,
              Eigen::Vector3d const &vel, Eigen::Vector4d const &att, Eigen::Vector3d const &desPos,
              Eigen::Vector3d const &desVel, Eigen::Vector3d const &desAcc,
              Eigen::Vector3d const &groundWind, double dt) {
             Vec3d angVel(0, 0, 0);
             double forwardThrust = 0;
             c.Run(V(pos), V(vel), Q(att), V(desPos), V(desVel), V(desAcc), V(groundWind), angVel,
                   forwardThrust, float(dt));
             return py::make_tuple(E(angVel), forwardThrust);
           },
           py::arg("pos"), py::arg("vel"), py::arg("att"), py::arg("des_pos"), py::arg("des_vel"),
           py::arg("des_acc"), py::arg("ground_wind"), py::arg("dt"));

  // ---------------------------------------------------------------- PX4 fixed-wing controllers (onboard)
  // The same classes TrirotorLogic runs in FS_TRIMOTOR_ATTITUDE_CONTROL; exposed standalone so an
  // environment can evaluate them (e.g. as a reference/expert) outside the onboard logic.
  py::class_<Onboard::PX4FixedWingAttitudeController>(m, "PX4FixedWingAttitudeController")
      .def(py::init<>())
      .def("set_parameters", &Onboard::PX4FixedWingAttitudeController::SetParameters,
           py::arg("time_const_x"), py::arg("time_const_y"))
      .def("time_constants",
           [](Onboard::PX4FixedWingAttitudeController const &c) { return Ef(c.GetTimeConstants()); })
      .def("desired_angular_velocity",
           [](Onboard::PX4FixedWingAttitudeController const &c, Eigen::Vector4d const &desAtt,
              Eigen::Vector4d const &estAtt, double trueAirspeed, double cmdYawRate) {
             return Ef(c.GetDesiredAngularVelocity(Qf(desAtt), Qf(estAtt), float(trueAirspeed),
                                                   float(cmdYawRate)));
           },
           py::arg("des_att"), py::arg("est_att"), py::arg("true_airspeed"), py::arg("cmd_yaw_rate") = 0.0);

  py::class_<Onboard::PX4FixedWingAngularVelocityController>(m, "PX4FixedWingAngularVelocityController")
      .def(py::init<>())
      .def("set_parameters",
           [](Onboard::PX4FixedWingAngularVelocityController &c, Eigen::Vector3d const &Kp,
              Eigen::Vector3d const &Ki, Eigen::Vector3d const &Kff, Eigen::Vector3d const &integralLimit,
              double trimAirspeed) {
             c.SetParameters(Vf(Kp), Vf(Ki), Vf(Kff), Vf(integralLimit), float(trimAirspeed));
           },
           py::arg("Kp"), py::arg("Ki"), py::arg("Kff"), py::arg("integral_limit"), py::arg("trim_airspeed"))
      .def("set_trim_airspeed", &Onboard::PX4FixedWingAngularVelocityController::SetTrimAirspeed,
           py::arg("trim_airspeed"))
      .def("reset_integral", &Onboard::PX4FixedWingAngularVelocityController::ResetIntegral)
      // normalized pseudo-torques in [-1, 1] (roll, pitch, yaw)
      .def("desired_normalized_torques",
           [](Onboard::PX4FixedWingAngularVelocityController &c, Eigen::Vector3d const &desAngVel,
              Eigen::Vector3d const &estAngVel, double trueAirspeed, double dt) {
             return Ef(c.GetDesiredNormalizedTorques(Vf(desAngVel), Vf(estAngVel), float(trueAirspeed),
                                                     float(dt)));
           },
           py::arg("des_ang_vel"), py::arg("est_ang_vel"), py::arg("true_airspeed"), py::arg("dt"));

  // ---------------------------------------------------------------- mixer
  // The mixer TrirotorLogic runs, set up from the same constants, so an environment can turn thrust and
  // normalized torques into the actuator commands of make_speeds_command.
  using Mixer = Onboard::TrirotorMixer;
  py::class_<Mixer>(m, "TrirotorMixer")
      .def(py::init([](Consts const &c) {
             Mixer mixer;
             mixer.SetParameters(c.dx, c.dy1, c.dy2, c.dz, c.propellerThrustFromSpeedSqr,
                                 c.propellerTorqueFromThrust, c.prop0SpinDir, c.maxThrustPerPropeller,
                                 c.minThrustPerPropeller, c.maxCmdTotalThrust, c.minMotorServoAngle,
                                 c.maxMotorServoAngle, c.tailType, c.tiltServos, c.ailerons, c.tailSurfaces,
                                 c.aeroCoeffs, c.wingParams,
                                 Vec3f(c.aeroRefPointToComOffset_x, c.aeroRefPointToComOffset_y,
                                       c.aeroRefPointToComOffset_z));
             mixer.SetAirspeedModel(c.propellerDiameter, c.propAirspeedModel);
             return mixer;
           }),
           py::arg("consts"))
      // normalized torques in [-1, 1] -> (aileron cmds (R, L), tail cmds), raw servo commands
      .def("px4_control_surface_cmds",
           [](Mixer const &c, Eigen::Vector3d const &normalizedTorque) {
             std::array<float, 2> aileron, tail;
             c.GetPX4ControlSurfaceCmds(Vf(normalizedTorque), aileron.data(), tail.data());
             return py::make_tuple(aileron, tail);
           },
           py::arg("normalized_torque"))
      .def("propeller_speeds_from_thrust",
           [](Mixer const &c, Eigen::Vector3d const &thrusts, Eigen::Vector3d const &motorAngles,
              Eigen::Vector3d const &airspeedBody) {
             float const t[3] = {float(thrusts[0]), float(thrusts[1]), float(thrusts[2])};
             float const a[3] = {float(motorAngles[0]), float(motorAngles[1]), float(motorAngles[2])};
             float speeds[3];
             c.PropellerSpeedsFromThrust(t, speeds, a, Vf(airspeedBody));
             return Eigen::Vector3d(speeds[0], speeds[1], speeds[2]);
           },
           py::arg("thrusts"), py::arg("motor_angles"), py::arg("airspeed_b"))
      .def("motor_servo_cmds_from_angles",
           [](Mixer const &c, Eigen::Vector3d const &angles) {
             float const a[3] = {float(angles[0]), float(angles[1]), float(angles[2])};
             float cmds[3];
             c.MotorServoCmdsFromAngles(a, cmds);
             return Eigen::Vector3d(cmds[0], cmds[1], cmds[2]);
           },
           py::arg("angles"));
}
