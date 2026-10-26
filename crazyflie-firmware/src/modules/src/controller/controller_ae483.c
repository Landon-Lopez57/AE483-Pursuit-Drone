#include "controller_ae483.h"
#include "stabilizer_types.h"
#include "power_distribution.h"
#include "log.h"
#include "param.h"
#include "num.h"
#include "math3d.h"

// Sensor measurements
// - tof (from the z ranger on the flow deck)
static uint16_t tof_count = 0;
static float tof_distance = 0.0f;
// - flow (from the optical flow sensor on the flow deck)
static uint16_t flow_count = 0;
static float flow_dpixelx = 0.0f;
static float flow_dpixely = 0.0f;

// Parameters
static bool use_observer = false;
static bool reset_observer = false;
static bool gains_are_set = false;

// State
static float p_x = 0.0f;
static float p_y = 0.0f;
static float p_z = 0.0f;
static float psi = 0.0f;
static float theta = 0.0f;
static float phi = 0.0f;
static float v_x = 0.0f;
static float v_y = 0.0f;
static float v_z = 0.0f;
static float w_x = 0.0f;
static float w_y = 0.0f;
static float w_z = 0.0f;



// Input
static float tau_x = 0.0f;
static float tau_y = 0.0f;
static float tau_z = 0.0f;
static float f_z = 0.0f;

// Desired position
static float p_x_des = 0.0f;
static float p_y_des = 0.0f;
static float p_z_des = 0.0f;

// Motor power command
static uint16_t m_1 = 0;
static uint16_t m_2 = 0;
static uint16_t m_3 = 0;
static uint16_t m_4 = 0;

// For time delay
static float tau_x_cmd = 0.0f;    // tau_x command
static float tau_y_cmd = 0.0f;    // tau_y command
static float w_x_old = 0.0f;      // value of w_x from previous time step
static float w_y_old = 0.0f;      // value of w_y from previous time step
static float J_x = 0.0000292f;          // FIXME: principal moment of inertia about x_B axis
static float J_y = 0.0000319f;          // FIXME: principal moment of inertia about y_B axis
static float dt = 0.002f;         // time step (corresponds to 500 Hz)

// For integral action
static float e_x = 0.0f;    // integrated error in p_x
static float e_y = 0.0f;    // integrated error in p_y
static float e_z = 0.0f;    // integrated error in p_z

// Gains (these are set by flight.py - do not edit the values here)
static float eq_tau_x_cmd = 0.0f;
static float k_tau_x_cmd_p_y = 0.0f;
static float k_tau_x_cmd_phi = 0.0f;
static float k_tau_x_cmd_v_y = 0.0f;
static float k_tau_x_cmd_w_x = 0.0f;
static float k_tau_x_cmd_tau_x = 0.0f;
static float k_tau_x_cmd_e_y = 0.0f;
static float eq_tau_y_cmd = 0.0f;
static float k_tau_y_cmd_p_x = 0.0f;
static float k_tau_y_cmd_theta = 0.0f;
static float k_tau_y_cmd_v_x = 0.0f;
static float k_tau_y_cmd_w_y = 0.0f;
static float k_tau_y_cmd_tau_y = 0.0f;
static float k_tau_y_cmd_e_x = 0.0f;
static float eq_tau_z = 0.0f;
static float k_tau_z_psi = 0.0f;
static float k_tau_z_w_z = 0.0f;
static float eq_f_z = 0.0f;
static float k_f_z_p_z = 0.0f;
static float k_f_z_v_z = 0.0f;
static float k_f_z_e_z = 0.0f;

void ae483UpdateWithTOF(tofMeasurement_t *tof)
{
  tof_distance = tof->distance;
  tof_count++;
}

void ae483UpdateWithFlow(flowMeasurement_t *flow)
{
  flow_dpixelx = flow->dpixelx;
  flow_dpixely = flow->dpixely;
  flow_count++;
}

void ae483UpdateWithDistance(distanceMeasurement_t *meas)
{
  // If you have a loco positioning deck, this function will be called
  // each time a distance measurement is available. You will have to write
  // code to handle these measurements. These data are available:
  //
  //  meas->anchorId  uint8_t   id of anchor with respect to which distance was measured
  //  meas->x         float     x position of this anchor
  //  meas->y         float     y position of this anchor
  //  meas->z         float     z position of this anchor
  //  meas->distance  float     the measured distance
}

void ae483UpdateWithPosition(positionMeasurement_t *meas)
{
  // This function will be called each time you send an external position
  // measurement (x, y, z) from the client, e.g., from a motion capture system.
  // You will have to write code to handle these measurements. These data are
  // available:
  //
  //  meas->x         float     x component of external position measurement
  //  meas->y         float     y component of external position measurement
  //  meas->z         float     z component of external position measurement
}

void ae483UpdateWithPose(poseMeasurement_t *meas)
{
  // This function will be called each time you send an external "pose" measurement
  // (position as x, y, z and orientation as quaternion) from the client, e.g., from
  // a motion capture system. You will have to write code to handle these measurements.
  // These data are available:
  //
  //  meas->x         float     x component of external position measurement
  //  meas->y         float     y component of external position measurement
  //  meas->z         float     z component of external position measurement
  //  meas->quat.x    float     x component of quaternion from external orientation measurement
  //  meas->quat.y    float     y component of quaternion from external orientation measurement
  //  meas->quat.z    float     z component of quaternion from external orientation measurement
  //  meas->quat.w    float     w component of quaternion from external orientation measurement
}

void ae483UpdateWithData(const struct AE483Data* data)
{
  // This function will be called each time AE483-specific data are sent
  // from the client to the drone. You will have to write code to handle
  // these data. For the example AE483Data struct, these data are:
  //
  //  data->x         float
  //  data->y         float
  //  data->z         float
  //
  // Exactly what "x", "y", and "z" mean in this context is up to you.
}


void controllerAE483Init(void)
{
  // Do nothing
}

bool controllerAE483Test(void)
{
  // Do nothing (test is always passed)
  return true;
}

void controllerAE483(control_t *control,
                     const setpoint_t *setpoint,
                     const sensorData_t *sensors,
                     const state_t *state,
                     const stabilizerStep_t stabilizerStep)
{
  // Make sure this function only runs at 500 Hz
  if (!RATE_DO_EXECUTE(ATTITUDE_RATE, stabilizerStep)) {
    return;
  }

  // Get the state (making sure to convert linear velocity from the world frame to the body frame)
  p_x = state->position.x;
  p_y = state->position.y;
  p_z = state->position.z;
  psi = radians(state->attitude.yaw);
  theta = - radians(state->attitude.pitch);
  phi = radians(state->attitude.roll);
  v_x = state->velocity.x*cosf(psi)*cosf(theta) + state->velocity.y*sinf(psi)*cosf(theta) - state->velocity.z*sinf(theta);
  v_y = state->velocity.x*(sinf(phi)*sinf(theta)*cosf(psi) - sinf(psi)*cosf(phi)) + state->velocity.y*(sinf(phi)*sinf(psi)*sinf(theta) + cosf(phi)*cosf(psi)) + state->velocity.z*sinf(phi)*cosf(theta);
  v_z = state->velocity.x*(sinf(phi)*sinf(psi) + sinf(theta)*cosf(phi)*cosf(psi)) + state->velocity.y*(-sinf(phi)*cosf(psi) + sinf(psi)*sinf(theta)*cosf(phi)) + state->velocity.z*cosf(phi)*cosf(theta);
  w_x = radians(sensors->gyro.x);
  w_y = radians(sensors->gyro.y);
  w_z = radians(sensors->gyro.z);

  // Estimate tau_x and tau_y by finite difference
  tau_x = J_x * (w_x - w_x_old) / dt;
  tau_y = J_y * (w_y - w_y_old) / dt;
  w_x_old = w_x;
  w_y_old = w_y;

  // Get the desired position
  p_x_des = setpoint->position.x;
  p_y_des = setpoint->position.y;
  p_z_des = setpoint->position.z;

  if (setpoint->mode.z == modeDisable) {
    // If there is no desired position, then all
    // motor power commands should be zero

    m_1 = 0;
    m_2 = 0;
    m_3 = 0;
    m_4 = 0;

    // Reset the integrated errors
    e_x = 0.0f;
    e_y = 0.0f;
    e_z = 0.0f;
  } else {
    // Otherwise, motor power commands should be
    // chosen by the controller
      // Estimate e_x, e_y, e_z by Forward Euler
    e_x += dt * (p_x - p_x_des);
    e_y += dt * (p_y - p_y_des);
    e_z += dt * (p_z - p_z_des);

    // FIXME (CONTROLLER GOES HERE)
    tau_x_cmd = eq_tau_x_cmd - (k_tau_x_cmd_p_y * (p_y - p_y_des) + k_tau_x_cmd_phi * phi + k_tau_x_cmd_v_y * v_y + k_tau_x_cmd_w_x * w_x + k_tau_x_cmd_tau_x * tau_x + k_tau_x_cmd_e_y * e_y);
    tau_y_cmd = eq_tau_y_cmd - (k_tau_y_cmd_p_x * (p_x - p_x_des) + k_tau_y_cmd_theta * theta + k_tau_y_cmd_v_x * v_x + k_tau_y_cmd_w_y * w_y + k_tau_y_cmd_tau_y * tau_y + k_tau_y_cmd_e_x * e_x);
    tau_z = eq_tau_z - (k_tau_z_psi * psi + k_tau_z_w_z * w_z);
    f_z = eq_f_z - (k_f_z_p_z * (p_z - p_z_des) + k_f_z_v_z * v_z + k_f_z_e_z * e_z);
    
    // FIXME (METHOD OF POWER DISTRIBUTION GOES HERE)
    m_1 = limitUint16( -3132832.1f * tau_x_cmd -3132832.1f * tau_y_cmd -21551724.1f * tau_z + 93985.0f * f_z );
    m_2 = limitUint16( -3132832.1f * tau_x_cmd + 3132832.1f * tau_y_cmd + 21551724.1f * tau_z + 93985.0f * f_z );
    m_3 = limitUint16( 3132832.1f * tau_x_cmd + 3132832.1f * tau_y_cmd -21551724.1f * tau_z + 93985.0f * f_z );
    m_4 = limitUint16( 3132832.1f * tau_x_cmd -3132832.1f * tau_y_cmd + 21551724.1f * tau_z + 93985.0f * f_z );  
  }
  
  // Do not spin the motors unless the gains have been sent by flight.py.
  // Without this, a drone that never received its gains would try to fly
  // with all gains equal to zero, which would end badly.
  if (!gains_are_set) {
    m_1 = 0;
    m_2 = 0;
    m_3 = 0;
    m_4 = 0;
  }

  // Apply motor power commands
  control->m1 = m_1;
  control->m2 = m_2;
  control->m3 = m_3;
  control->m4 = m_4;
}

//              1234567890123456789012345 <-- max total length
//              group   .name
LOG_GROUP_START(ae483log)
LOG_ADD(LOG_UINT16,      num_tof,                &tof_count)
LOG_ADD(LOG_UINT16,      num_flow,               &flow_count)
LOG_ADD(LOG_FLOAT,       p_x,                    &p_x)
LOG_ADD(LOG_FLOAT,       p_y,                    &p_y)
LOG_ADD(LOG_FLOAT,       p_z,                    &p_z)
LOG_ADD(LOG_FLOAT,       psi,                    &psi)
LOG_ADD(LOG_FLOAT,       theta,                  &theta)
LOG_ADD(LOG_FLOAT,       phi,                    &phi)
LOG_ADD(LOG_FLOAT,       v_x,                    &v_x)
LOG_ADD(LOG_FLOAT,       v_y,                    &v_y)
LOG_ADD(LOG_FLOAT,       v_z,                    &v_z)
LOG_ADD(LOG_FLOAT,       w_x,                    &w_x)
LOG_ADD(LOG_FLOAT,       w_y,                    &w_y)
LOG_ADD(LOG_FLOAT,       w_z,                    &w_z)
LOG_ADD(LOG_FLOAT,       p_x_des,                &p_x_des)
LOG_ADD(LOG_FLOAT,       p_y_des,                &p_y_des)
LOG_ADD(LOG_FLOAT,       p_z_des,                &p_z_des)
LOG_ADD(LOG_FLOAT,       tau_x,                  &tau_x)
LOG_ADD(LOG_FLOAT,       tau_y,                  &tau_y)
LOG_ADD(LOG_FLOAT,       tau_z,                  &tau_z)
LOG_ADD(LOG_FLOAT,       f_z,                    &f_z)
LOG_ADD(LOG_UINT16,      m_1,                    &m_1)
LOG_ADD(LOG_UINT16,      m_2,                    &m_2)
LOG_ADD(LOG_UINT16,      m_3,                    &m_3)
LOG_ADD(LOG_UINT16,      m_4,                    &m_4)
LOG_ADD(LOG_FLOAT,       tau_x_cmd,              &tau_x_cmd)
LOG_ADD(LOG_FLOAT,       tau_y_cmd,              &tau_y_cmd)
LOG_ADD(LOG_FLOAT,             e_x,                    &e_x)
LOG_ADD(LOG_FLOAT,             e_y,                    &e_y)
LOG_ADD(LOG_FLOAT,             e_z,                    &e_z)
LOG_GROUP_STOP(ae483log)

//                1234567890123456789012345 <-- max total length
//                group   .name
PARAM_GROUP_START(ae483par)
PARAM_ADD(PARAM_UINT8,     use_observer,            &use_observer)
PARAM_ADD(PARAM_UINT8,     reset_observer,          &reset_observer)
PARAM_ADD(PARAM_UINT8,     gains_are_set,           &gains_are_set)
PARAM_GROUP_STOP(ae483par)


//                1234567890123456789012345 <-- max total length
//                group .name
PARAM_GROUP_START(ae483g)
PARAM_ADD(PARAM_FLOAT,   eq_tau_x_cmd,            &eq_tau_x_cmd)
PARAM_ADD(PARAM_FLOAT,   k_tau_x_cmd_p_y,         &k_tau_x_cmd_p_y)
PARAM_ADD(PARAM_FLOAT,   k_tau_x_cmd_phi,         &k_tau_x_cmd_phi)
PARAM_ADD(PARAM_FLOAT,   k_tau_x_cmd_v_y,         &k_tau_x_cmd_v_y)
PARAM_ADD(PARAM_FLOAT,   k_tau_x_cmd_w_x,         &k_tau_x_cmd_w_x)
PARAM_ADD(PARAM_FLOAT,   k_tau_x_cmd_tau_x,       &k_tau_x_cmd_tau_x)
PARAM_ADD(PARAM_FLOAT,   k_tau_x_cmd_e_y,         &k_tau_x_cmd_e_y)
PARAM_ADD(PARAM_FLOAT,   eq_tau_y_cmd,            &eq_tau_y_cmd)
PARAM_ADD(PARAM_FLOAT,   k_tau_y_cmd_p_x,         &k_tau_y_cmd_p_x)
PARAM_ADD(PARAM_FLOAT,   k_tau_y_cmd_theta,       &k_tau_y_cmd_theta)
PARAM_ADD(PARAM_FLOAT,   k_tau_y_cmd_v_x,         &k_tau_y_cmd_v_x)
PARAM_ADD(PARAM_FLOAT,   k_tau_y_cmd_w_y,         &k_tau_y_cmd_w_y)
PARAM_ADD(PARAM_FLOAT,   k_tau_y_cmd_tau_y,       &k_tau_y_cmd_tau_y)
PARAM_ADD(PARAM_FLOAT,   k_tau_y_cmd_e_x,         &k_tau_y_cmd_e_x)
PARAM_ADD(PARAM_FLOAT,   eq_tau_z,                &eq_tau_z)
PARAM_ADD(PARAM_FLOAT,   k_tau_z_psi,             &k_tau_z_psi)
PARAM_ADD(PARAM_FLOAT,   k_tau_z_w_z,             &k_tau_z_w_z)
PARAM_ADD(PARAM_FLOAT,   eq_f_z,                  &eq_f_z)
PARAM_ADD(PARAM_FLOAT,   k_f_z_p_z,               &k_f_z_p_z)
PARAM_ADD(PARAM_FLOAT,   k_f_z_v_z,               &k_f_z_v_z)
PARAM_ADD(PARAM_FLOAT,   k_f_z_e_z,               &k_f_z_e_z)
PARAM_GROUP_STOP(ae483g)