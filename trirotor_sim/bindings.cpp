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
#include "Components/Logic/QuadcopterConstants.hpp"
#include "Components/Offboard/LinearAerodynamicModel.hpp"
#include "Components/Offboard/TrirotorController.hpp"
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
      .def_readwrite("attControl_timeConst_z", &Consts::attControl_timeConst_z)
      .def_readwrite("attControl_integralGain", &Consts::attControl_integralGain)
      .def_readwrite("attControl_integralLimit_x", &Consts::attControl_integralLimit_x)
      .def_readwrite("attControl_integralLimit_y", &Consts::attControl_integralLimit_y)
      .def_readwrite("attControl_integralLimit_z", &Consts::attControl_integralLimit_z)
      .def_readwrite("tiltServoDynamics", &Consts::tiltServoDynamics)
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
}
