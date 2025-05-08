'''
Python 3.x library to control an UR robot through its TCP/IP interfaces
Copyright (C) 2017  Martin Huus Bjerge, Rope Robotics ApS, Denmark

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute,
sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL "Rope Robotics ApS" BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Except as contained in this notice, the name of "Rope Robotics ApS" shall not be used
in advertising or otherwise to promote the sale, use or other dealings in this Software
without prior written authorization from "Rope Robotics ApS".
'''
__author__ = "Martin Huus Bjerge"
__copyright__ = "Copyright 2017, Rope Robotics ApS, Denmark"
__license__ = "MIT License"

import URBasic
import numpy as np
from enum import Enum, auto
from scipy.spatial.transform import Rotation as R
from .robotModel import RobotModel
from .waypoint6d import TCP6D, Joint6D, TCP6DDescriptor, Joint6DDescriptor

from copy import deepcopy


class UrScriptExt(URBasic.urScript.UrScript):
  '''
  Interface to remote access UR script commands, and add some extended features as well.
  For more details see the script manual at this site:
  http://www.universal-robots.com/download/

  Beside the implementation of the script interface, this class also inherits from the
  Real Time Client and RTDE interface and thereby also open a connection to these data interfaces.
  The Real Time Client in this version is only used to send program and script commands
  to the robot, not to read data from the robot, all data reading is done via the RTDE interface.

  This class also opens a connection to the UR Dashboard server and enables you to
  e.g. reset error and warnings from the UR controller.
  '''

  def __init__(self, host : str, robotModel : RobotModel, hasForceTorque : bool = False, conf_filename : str | None = None):
    if host is None or not host:
      raise ValueError("Host must be a valid IP address or hostname")
    if robotModel is None:
      raise ValueError("RobotModel must be a valid RobotModel object")
    super(UrScriptExt, self).__init__(host, robotModel, hasForceTorque, conf_filename)
    logger = URBasic.dataLogging.DataLogging()
    name = logger.AddEventLogging(__name__, log2Consol=False)
    self.__logger = logger.__dict__[name]
    self.__logger.info('Init done')

  def close(self):
    self.idle()
    self.robotConnector.close()

  def reset_error(self) -> bool:
    '''
    Check if the UR controller is powered on and ready to run.
    If controller isn't power on it will be power up.
    If there is a safety error, it will be tried rest it once.

    Return Value:
    state (boolean): True of power is on and no safety errors active.
    '''

    if not self.robotConnector.RobotModel.RobotStatus().PowerOn:
      # self.robotConnector.DashboardClient.PowerOn()
      self.robotConnector.DashboardClient.ur_power_on()
      self.robotConnector.DashboardClient.wait_dbs()
      # self.robotConnector.DashboardClient.BrakeRelease()
      self.robotConnector.DashboardClient.ur_brake_release()
      self.robotConnector.DashboardClient.wait_dbs()

    if self.robotConnector.RobotModel.SafetyStatus().StoppedDueToSafety:  # self.get_safety_status()['StoppedDueToSafety']:
      # self.robotConnector.DashboardClient.UnlockProtectiveStop()
      self.robotConnector.DashboardClient.ur_unlock_protective_stop()
      self.robotConnector.DashboardClient.wait_dbs()
      # self.robotConnector.DashboardClient.CloseSafetyPopup()
      self.robotConnector.DashboardClient.ur_close_safety_popup()
      self.robotConnector.DashboardClient.wait_dbs()
      # self.robotConnector.DashboardClient.BrakeRelease()
      self.robotConnector.DashboardClient.ur_brake_release()
      self.robotConnector.DashboardClient.wait_dbs()

      # ADDED: If there was a safety stop -> reupload the realtime control program
      self.init_realtime_control()

    # return self.get_robot_status()['PowerOn'] & (not self.get_safety_status()['StoppedDueToSafety'])
    return self.robotConnector.RobotModel.RobotStatus().PowerOn and not self.robotConnector.RobotModel.SafetyStatus().StoppedDueToSafety

  def get_in(self, port : str, wait=True) -> bool | float:
    '''
    Get input signal level

    Parameters:
    port (HW profile str): Hardware profile tag
    wait (bool): True if wait for next RTDE sample, False, to get the latest sample

    Return Value:
    out (bool or float), The signal level.
    '''
    if 'BCI' == port[:3]:
      return self.get_configurable_digital_in(int(port[4:]), wait)
    elif 'BDI' == port[:3]:
      return self.get_standard_digital_in(int(port[4:]), wait)
    elif 'BAI' == port[:3]:
      return self.get_standard_analog_in(int(port[4:]), wait)

  def set_output(self, port : str, value : bool | float) -> bool:
    '''
    Get output signal level

    Parameters:
    port (HW profile str): Hardware profile tag
    value (bool or float): The output value to be set

    Return Value:
    Status (bool): Status, True if signal set successfully.
    '''
    if self.__locked_func:
      return False
    if 'BCO' == port[:3]:
      self.set_configurable_digital_out(int(port[4:]), value)
    elif 'BDO' == port[:3]:
      self.set_standard_digital_out(int(port[4:]), value)
    elif 'BAO' == port[:3]:
      pass
    elif 'TDO' == port[:3]:
      pass

      # if self.sendData():
      #    return True
      return True  # Vi har sendt det .. vi checker ikke
    else:
      return False

  def init_force_remote(self, task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], f_type=2):
    '''
    The Force Remote function enables changing the force settings dynamically,
    without sending new programs to the robot, and thereby exit and enter force mode again.
    As the new settings are send via RTDE, the force can be updated every 8ms.
    This function initializes the remote force function,
    by sending a program to the robot that can receive new force settings.

    See "force_mode" for more details on force functions

    Parameters:
    task_frame (6D-vector): Initial task frame (can be changed via the update function)
    f_type (int): Initial force type (can be changed via the update function)

    Return Value:
    Status (bool): Status, True if successfully initialized.
    '''
    if self.__locked_func:
      return False

    if not self.robotConnector.RTDE.isRunning():
      self.__logger.error('RTDE need to be running to use force remote')
      return False

    selection_vector = [0, 0, 0, 0, 0, 0]
    wrench = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    limits = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    self.robotConnector.RTDE.setData('input_int_register_0', selection_vector[0])
    self.robotConnector.RTDE.setData('input_int_register_1', selection_vector[1])
    self.robotConnector.RTDE.setData('input_int_register_2', selection_vector[2])
    self.robotConnector.RTDE.setData('input_int_register_3', selection_vector[3])
    self.robotConnector.RTDE.setData('input_int_register_4', selection_vector[4])
    self.robotConnector.RTDE.setData('input_int_register_5', selection_vector[5])

    self.robotConnector.RTDE.setData('input_double_register_0', wrench[0])
    self.robotConnector.RTDE.setData('input_double_register_1', wrench[1])
    self.robotConnector.RTDE.setData('input_double_register_2', wrench[2])
    self.robotConnector.RTDE.setData('input_double_register_3', wrench[3])
    self.robotConnector.RTDE.setData('input_double_register_4', wrench[4])
    self.robotConnector.RTDE.setData('input_double_register_5', wrench[5])

    self.robotConnector.RTDE.setData('input_double_register_6', limits[0])
    self.robotConnector.RTDE.setData('input_double_register_7', limits[1])
    self.robotConnector.RTDE.setData('input_double_register_8', limits[2])
    self.robotConnector.RTDE.setData('input_double_register_9', limits[3])
    self.robotConnector.RTDE.setData('input_double_register_10', limits[4])
    self.robotConnector.RTDE.setData('input_double_register_11', limits[5])

    self.robotConnector.RTDE.setData('input_double_register_12', task_frame[0])
    self.robotConnector.RTDE.setData('input_double_register_13', task_frame[1])
    self.robotConnector.RTDE.setData('input_double_register_14', task_frame[2])
    self.robotConnector.RTDE.setData('input_double_register_15', task_frame[3])
    self.robotConnector.RTDE.setData('input_double_register_16', task_frame[4])
    self.robotConnector.RTDE.setData('input_double_register_17', task_frame[5])

    self.robotConnector.RTDE.setData('input_int_register_6', f_type)
    self.robotConnector.RTDE.sendData()

    prog = '''def force_remote():
  while (True):

      global task_frame =  p[read_input_float_register(12),
                            read_input_float_register(13),
                            read_input_float_register(14),
                            read_input_float_register(15),
                            read_input_float_register(16),
                            read_input_float_register(17)]


      global selection_vector = [ read_input_integer_register(0),
                                  read_input_integer_register(1),
                                  read_input_integer_register(2),
                                  read_input_integer_register(3),
                                  read_input_integer_register(4),
                                  read_input_integer_register(5)]

      global wrench = [ read_input_float_register(0),
                        read_input_float_register(1),
                        read_input_float_register(2),
                        read_input_float_register(3),
                        read_input_float_register(4),
                        read_input_float_register(5)]

      global limits = [ read_input_float_register(6),
                        read_input_float_register(7),
                        read_input_float_register(8),
                        read_input_float_register(9),
                        read_input_float_register(10),
                        read_input_float_register(11)]

      global f_type = read_input_integer_register(6)

      force_mode(task_frame, selection_vector, wrench, f_type , limits)
      sync()
  end
end
'''
    self.robotConnector.RealTimeClient.SendProgram(prog.format(**locals()))
    self.robotConnector.RobotModel.forceRemoteActiveFlag = True

  def set_force_remote(self, task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], selection_vector=[0, 0, 0, 0, 0, 0],
                        wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], limits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1], f_type=2):
    '''
    Update/set remote force, see "init_force_remote" for more details.

    Parameters:
    task frame: A pose vector that defines the force frame relative to the base frame.

    selection vector: A 6d vector that may only contain 0 or 1. 1 means that the robot will be
                      compliant in the corresponding axis of the task frame, 0 means the robot is
                      not compliant along/about that axis.

    wrench: The forces/torques the robot is to apply to its environment. These values
            have different meanings whether they correspond to a compliant axis or not.
            Compliant axis: The robot will adjust its position along/about the axis in order
            to achieve the specified force/torque. Non-compliant axis: The robot follows
            the trajectory of the program but will account for an external force/torque
            of the specified value.

    limits: A 6d vector with float values that are interpreted differently for
            compliant/non-compliant axes:
            Compliant axes: The limit values for compliant axes are the maximum
                            allowed tcp speed along/about the axis.
            Non-compliant axes: The limit values for non-compliant axes are the
                                maximum allowed deviation along/about an axis between the
                                actual tcp position and the one set by the program.

    f_type: An integer specifying how the robot interprets the force frame.
            1: The force frame is transformed in a way such that its y-axis is aligned with a vector
                pointing from the robot tcp towards the origin of the force frame.
            2: The force frame is not transformed.
            3: The force frame is transformed in a way such that its x-axis is the projection of
                the robot tcp velocity vector onto the x-y plane of the force frame.
            All other values of f_type are invalid.

    Return Value:
    Status (bool): Status, True if parameters successfully updated.
    '''
    if self.__locked_func:
      return False
    if not self.robotConnector.RobotModel.forceRemoteActiveFlag:
      self.init_force_remote(task_frame, f_type)

    if self.robotConnector.RTDE.isRunning() and self.robotConnector.RobotModel.forceRemoteActiveFlag:
      self.robotConnector.RTDE.setData('input_int_register_0', selection_vector[0])
      self.robotConnector.RTDE.setData('input_int_register_1', selection_vector[1])
      self.robotConnector.RTDE.setData('input_int_register_2', selection_vector[2])
      self.robotConnector.RTDE.setData('input_int_register_3', selection_vector[3])
      self.robotConnector.RTDE.setData('input_int_register_4', selection_vector[4])
      self.robotConnector.RTDE.setData('input_int_register_5', selection_vector[5])

      self.robotConnector.RTDE.setData('input_double_register_0', wrench[0])
      self.robotConnector.RTDE.setData('input_double_register_1', wrench[1])
      self.robotConnector.RTDE.setData('input_double_register_2', wrench[2])
      self.robotConnector.RTDE.setData('input_double_register_3', wrench[3])
      self.robotConnector.RTDE.setData('input_double_register_4', wrench[4])
      self.robotConnector.RTDE.setData('input_double_register_5', wrench[5])

      self.robotConnector.RTDE.setData('input_double_register_6', limits[0])
      self.robotConnector.RTDE.setData('input_double_register_7', limits[1])
      self.robotConnector.RTDE.setData('input_double_register_8', limits[2])
      self.robotConnector.RTDE.setData('input_double_register_9', limits[3])
      self.robotConnector.RTDE.setData('input_double_register_10', limits[4])
      self.robotConnector.RTDE.setData('input_double_register_11', limits[5])

      self.robotConnector.RTDE.setData('input_double_register_12', task_frame[0])
      self.robotConnector.RTDE.setData('input_double_register_13', task_frame[1])
      self.robotConnector.RTDE.setData('input_double_register_14', task_frame[2])
      self.robotConnector.RTDE.setData('input_double_register_15', task_frame[3])
      self.robotConnector.RTDE.setData('input_double_register_16', task_frame[4])
      self.robotConnector.RTDE.setData('input_double_register_17', task_frame[5])

      self.robotConnector.RTDE.setData('input_int_register_6', f_type)

      self.robotConnector.RTDE.sendData()
      return True

    else:
      if not self.robotConnector.RobotModel.forceRemoteActiveFlag:
        self.__logger.warning('Force Remote not initialized')
      else:
        self.__logger.warning('RTDE is not running')

      return False

  def init_realtime_control(self):
    '''
    The realtime control mode enables continuous updates to a servoj program which is
    initialized by this function. This way no new program has to be sent to the robot
    and the robot can perform a smooth trajectory.
    Sending new servoj commands is done by utilizing RTDE of this library
    
    Parameters:
    sample_time: time of one sample, standard is 8ms as this is the thread-cycle time of UR
    
    Return Value:
    Status (bool): Status, True if successfully initialized.
    '''
    
    if self.__locked_func:
      raise Exception('This functionality is locked - do not use')
      return False
    if not self.robotConnector.RTDE.isRunning():
      self.__logger.error('RTDE needs to be running to use realtime control')
      return False

    # get current tcp pos
    init_pose = self.get_actual_tcp_pose()

    self.robotConnector.RTDE.setData('input_double_register_0', init_pose[0])
    self.robotConnector.RTDE.setData('input_double_register_1', init_pose[1])
    self.robotConnector.RTDE.setData('input_double_register_2', init_pose[2])
    self.robotConnector.RTDE.setData('input_double_register_3', init_pose[3])
    self.robotConnector.RTDE.setData('input_double_register_4', init_pose[4])
    self.robotConnector.RTDE.setData('input_double_register_5', init_pose[5])

    self.robotConnector.RTDE.sendData()

    prog = '''def realtime_control():
  
  
  while (True):
      
      new_pose = p[read_input_float_register(0),
                  read_input_float_register(1),
                  read_input_float_register(2),
                  read_input_float_register(3),
                  read_input_float_register(4),
                  read_input_float_register(5)]
          
      servoj(get_inverse_kin(new_pose), t=.4, gain=150)
          
      sync()
  end
end
'''
    # , t=0.1

    self.robotConnector.RealTimeClient.SendProgram(prog.format(**locals()))
    self.robotConnector.RobotModel.realtimeControlFlag = True

  def set_realtime_pose(self, pose : TCP6D):
    """
    Update/Set realtime_pose after sample_time seconds.

    Parameters
    pose: pose to transition to in sample_time seconds
    """
    if self.__locked_func:
      raise Exception('This functionality is locked - do not use')
      return False
    if not isinstance(pose, URBasic.waypoint6d.TCP6D):
      raise ValueError("pose must be a TCP6D object")
    if not self.robotConnector.RobotModel.realtimeControlFlag:
      print("Realtime control not initialized!")
      self.init_realtime_control()
      print("Realtime control initialized!")

    if self.robotConnector.RTDE.isRunning() and self.robotConnector.RobotModel.realtimeControlFlag:
      self.robotConnector.RTDE.setData('input_double_register_0', pose[0])
      self.robotConnector.RTDE.setData('input_double_register_1', pose[1])
      self.robotConnector.RTDE.setData('input_double_register_2', pose[2])
      self.robotConnector.RTDE.setData('input_double_register_3', pose[3])
      self.robotConnector.RTDE.setData('input_double_register_4', pose[4])
      self.robotConnector.RTDE.setData('input_double_register_5', pose[5])
      self.robotConnector.RTDE.sendData()
      return True
    else:
      if not self.robotConnector.RobotModel.realtimeControlFlag:
        self.__logger.warning('Realtime Remote Control not initialized')
      else:
        self.__logger.warning('RTDE is not running')

    return False

  def move_force_2stop(self, start_tolerance=0.01,
                        stop_tolerance=0.01,
                        wrench_gain=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        timeout=10,
                        task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        selection_vector=[0, 0, 0, 0, 0, 0],
                        wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                        limits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        f_type=2):
    '''
    Move force will set the robot in force mode (see force_mode) and move the TCP until it meets an object making the TCP stand still.

    Parameters:
    start_tolerance (float): sum of all elements in a pose vector defining a robot has started moving (60 samples)

    stop_tolerance (float): sum of all elements in a pose vector defining a standing still robot (60 samples)

    wrench_gain (6D vector): Gain multiplied with wrench each 8ms sample

    timeout (float): Seconds to timeout if tolerance not reached

    task frame: A pose vector that defines the force frame relative to the base frame.

    selection vector: A 6d vector that may only contain 0 or 1. 1 means that the robot will be
                      compliant in the corresponding axis of the task frame, 0 means the robot is
                      not compliant along/about that axis.

    wrench: The forces/torques the robot is to apply to its environment. These values
            have different meanings whether they correspond to a compliant axis or not.
            Compliant axis: The robot will adjust its position along/about the axis in order
            to achieve the specified force/torque. Non-compliant axis: The robot follows
            the trajectory of the program but will account for an external force/torque
            of the specified value.

    limits: A 6d vector with float values that are interpreted differently for
            compliant/non-compliant axes:
            Compliant axes: The limit values for compliant axes are the maximum
                            allowed tcp speed along/about the axis.
            Non-compliant axes: The limit values for non-compliant axes are the
                                maximum allowed deviation along/about an axis between the
                                actual tcp position and the one set by the program.

    f_type: An integer specifying how the robot interprets the force frame.
            1: The force frame is transformed in a way such that its y-axis is aligned with a vector
                pointing from the robot tcp towards the origin of the force frame.
            2: The force frame is not transformed.
            3: The force frame is transformed in a way such that its x-axis is the projection of
                the robot tcp velocity vector onto the x-y plane of the force frame.
            All other values of f_type are invalid.

    Return Value:
    Status (bool): Status, True if signal set successfully.

    '''
    if self.__locked_func:
      raise Exception('This functionality is locked - do not use')
      return False

    timeoutcnt = 125 * timeout
    wrench = np.array(wrench)
    wrench_gain = np.array(wrench_gain)
    self.set_force_remote(task_frame, selection_vector, wrench, limits, f_type)

    dist = np.array(range(60), float)
    dist.fill(0.)
    cnt = 0
    old_pose = self.get_actual_tcp_pose() * np.array(selection_vector)
    while np.sum(dist) < start_tolerance and cnt < timeoutcnt:
      new_pose = self.get_actual_tcp_pose() * np.array(selection_vector)
      wrench = wrench * wrench_gain  # Need a max wrencd check
      self.set_force_remote(task_frame, selection_vector, wrench, limits, f_type)
      dist[np.mod(cnt, 60)] = np.abs(np.sum(new_pose - old_pose))
      old_pose = new_pose
      cnt += 1

    # Check if robot started to move
    if cnt < timeoutcnt:
      dist.fill(stop_tolerance)
      cnt = 0
      while np.sum(dist) > stop_tolerance and cnt < timeoutcnt:
        new_pose = self.get_actual_tcp_pose() * np.array(selection_vector)
        dist[np.mod(cnt, 60)] = np.abs(np.sum(new_pose - old_pose))
        old_pose = new_pose
        cnt += 1

    self.set_force_remote(task_frame, selection_vector, [0, 0, 0, 0, 0, 0], limits, f_type)
    self.end_force_mode()
    if cnt >= timeoutcnt:
      return False
    else:
      return True

  def move_force(self, pose=None,
                a=1.2,
                v=0.25,
                t=0,
                r=0.0,
                movetype='l',
                task_frame=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                selection_vector=[0, 0, 0, 0, 0, 0],
                wrench=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                limits=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                f_type=2,
                wait=True,
                q=None):

    """
    Concatenate several move commands and applies a blending radius
    pose or q is a list of pose or joint-pose, and apply a force in a direction

    Parameters:
    pose: list of target pose (pose can also be specified as joint
          positions, then forward kinematics is used to calculate the corresponding pose see q)

    a:    tool acceleration [m/s^2]

    v:    tool speed [m/s]

    t:    time [S]

    r:    blend radius [m]

    movetype: (str): 'j', 'l', 'p', 'c'

    task frame: A pose vector that defines the force frame relative to the base frame.

    selection vector: A 6d vector that may only contain 0 or 1. 1 means that the robot will be
                      compliant in the corresponding axis of the task frame, 0 means the robot is
                      not compliant along/about that axis.

    wrench: The forces/torques the robot is to apply to its environment. These values
            have different meanings whether they correspond to a compliant axis or not.
            Compliant axis: The robot will adjust its position along/about the axis in order
            to achieve the specified force/torque. Non-compliant axis: The robot follows
            the trajectory of the program but will account for an external force/torque
            of the specified value.

    limits: A 6d vector with float values that are interpreted differently for
            compliant/non-compliant axes:
            Compliant axes: The limit values for compliant axes are the maximum
                            allowed tcp speed along/about the axis.
            Non-compliant axes: The limit values for non-compliant axes are the
                                maximum allowed deviation along/about an axis between the
                                actual tcp position and the one set by the program.

    f_type: An integer specifying how the robot interprets the force frame.
            1: The force frame is transformed in a way such that its y-axis is aligned with a vector
                pointing from the robot tcp towards the origin of the force frame.
            2: The force frame is not transformed.
            3: The force frame is transformed in a way such that its x-axis is the projection of
                the robot tcp velocity vector onto the x-y plane of the force frame.
            All other values of f_type are invalid.

    wait: function return when movement is finished

    q:    list of target joint positions


    Return Value:
    Status (bool): Status, True if signal set successfully.

    """
    if self.__locked_func:
      raise Exception('This functionality is locked - do not use')
      return False
    task_frame = np.array(task_frame)
    if np.size(task_frame.shape) == 2:
      prefix = "p"
      t_val = ''
      if pose is None:
          prefix = ""
          pose = q
      pose = np.array(pose)
      if movetype == 'j' or movetype == 'l':
          tval = 't={t},'.format(**locals())

      prg = 'def move_force():\n'
      for idx in range(np.size(pose, 0)):
          posex = np.round(pose[idx], 4)
          posex = posex.tolist()
          task_framex = np.round(task_frame[idx], 4)
          task_framex = task_framex.tolist()
          if (np.size(pose, 0) - 1) == idx:
              r = 0
          prg += '    force_mode(p{task_framex}, {selection_vector}, {wrench}, {f_type}, {limits})\n'.format(
              **locals())
          prg += '    move{movetype}({prefix}{posex}, a={a}, v={v}, {t_val} r={r})\n'.format(**locals())

      prg += '    stopl({a})\n'.format(**locals())
      prg += '    end_force_mode()\nend\n'
    else:
      prg = '''def move_force():
  force_mode(p{task_frame}, {selection_vector}, {wrench}, {f_type}, {limits})
{movestr}
  end_force_mode()
end
'''
      task_frame = task_frame.tolist()
      movestr = self._move(movetype, pose, a, v, t, r, wait, q)

    self.robotConnector.RealTimeClient.SendProgram(prg.format(**locals()))
    if (wait):
      self.waitRobotIdleOrStopFlag()

  def movej_waypoints(self, waypoints : list[Joint6DDescriptor], wait=True):
    '''
    Movej along multiple waypoints. By configuring a blend radius continuous movements can be enabled.

    Parameters:
    waypoints: List joints dictionaries of Joint6DDescriptor ({'q' : [joints6D], 'a' : a, 'v' : v, 't' : t, 'r' : r})
    '''
    if not isinstance(waypoints, list):
      raise ValueError("waypoints must be a list of Joint6DDescriptor objects")
    for w in waypoints:
      if not isinstance(w, URBasic.waypoint6d.Joint6DDescriptor):
        raise ValueError(f"waypoints must be a list of Joint6DDescriptor objects - at least one object is not a Joint6DDescriptor object (is {type(w)})")
    prg = '''def move_waypoints():
{exec_str}
end
'''
    exec_str = ""
    for waypoint in waypoints:
      movestr = self._move(movetype='j', **(waypoint.getAsDict()))
      exec_str += movestr + "\n"

    programString = prg.format(**locals())

    self.robotConnector.RealTimeClient.SendProgram(programString)
    if (wait):
        self.waitRobotIdleOrStopFlag()

  def movel_waypoints(self, waypoints : list[TCP6DDescriptor], wait=True):
    '''
    Movel along multiple waypoints. By configuring a blend radius continuous movements can be enabled.

    Parameters:
    waypoints: List waypoint dictionaries of TCP6DDescriptor ({'pose' : [pose6D], 'a' : a, 'v' : v, 't' : t, 'r' : r})
    '''
    if not isinstance(waypoints, list):
      raise ValueError("waypoints must be a list of TCP6DDescriptor objects")
    for w in waypoints:
      if not isinstance(w, URBasic.waypoint6d.TCP6DDescriptor):
        raise ValueError(f"waypoints must be a list of TCP6DDescriptor objects - at least one object is not a TCP6DDescriptor object (is {type(w)})")
    prg = '''def move_waypoints():
{exec_str}
end
'''
    exec_str = ""
    for waypoint in waypoints:
      movestr = self._move(movetype='l', **(waypoint.getAsDict()))
      exec_str += movestr + "\n"

    programString = prg.format(**locals())

    self.robotConnector.RealTimeClient.SendProgram(programString)
    if (wait):
      self.waitRobotIdleOrStopFlag()

  def print_actual_tcp_pose(self):
    '''
    print the actual TCP pose
    '''
    self.print_pose(self.get_actual_tcp_pose())

  def print_actual_joint_positions(self):
    '''
    print the actual TCP pose
    '''
    self.print_pose(q=self.get_actual_joint_positions())

  def print_pose(self, pose=None, q=None):
    '''
    print a pose
    '''
    if q is None:
      print('Robot Pose: [{: 06.4f}, {: 06.4f}, {: 06.4f},   {: 06.4f}, {: 06.4f}, {: 06.4f}]'.format(*pose))
    else:
      print('Robot joint positions: [{: 06.4f}, {: 06.4f}, {: 06.4f},   {: 06.4f}, {: 06.4f}, {: 06.4f}]'.format(
            *q))
    
  @staticmethod
  def transformationMatrixFromAxisAngle(x : float, y : float, z : float, rx : float, ry : float, rz : float) -> np.ndarray[np.float64]:
    ''' Create a transformation matrix from a position and rotation vector (axis-angle representation)
    
    Parameters:
      x, y, z: The position of the transformation matrix [m]
      rx, ry, rz: The rotation vector of the transformation matrix [rad]
      
    Returns:
      transformation_matrix: The transformation matrix
    '''
    # Convert rotation vector to rotation matrix
    rotation_vector = np.array([rx, ry, rz])
    rotation_matrix = R.from_rotvec(rotation_vector).as_matrix()
    
    # Homogeneous transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = rotation_matrix
    transformation_matrix[:3, 3] = [x, y, z]
    
    return transformation_matrix
  
  @staticmethod
  def transformationMatrixFromAxisAnglePose(pose : TCP6D) -> np.ndarray[np.float64]:
    ''' Create a transformation matrix from a pose
    
    Parameters:
      pose: The pose of the transformation matrix [m, rad]
      
    Returns:
      transformation_matrix: The transformation matrix
    '''
    return UrScriptExt.transformationMatrixFromAxisAngle(*(pose.toList()))
   
   
  class DirectionAxis(Enum):
    Xp = auto(),
    Xn = auto(),
    Yp = auto(),
    Yn = auto(),
    Zp = auto(),
    Zn = auto()
    
  class DirectionFeature(Enum):
    Base = auto(),
    Tool = auto()
       
  def moveLinearDirectionUntil(self, from_pose : TCP6D, feature : 'UrScriptExt.DirectionFeature',
        axis : 'UrScriptExt.DirectionAxis', dist_mm : float, accel : float = 1.2, velocity : float = 0.25) -> None:
    """
    Move in a direction until a certain distance is reached.
    
    Args:
      from_pose: The pose to calculate the movement from.
      feature: The reference feature to move in, i.e. the pose reference.
      axis: The direction to move in.
      dist_mm: The distance to move in mm.
      accel: The acceleration of the movement.
      velocity: The velocity of the movement.
      
    Raises:
      ValueError: If the axis or feature is invalid.
    """
    if not isinstance(from_pose, URBasic.waypoint6d.TCP6D):
      raise ValueError("from_pose must be a TCP6D object")
    if not isinstance(feature,  URBasic.UrScriptExt.DirectionFeature):
      raise ValueError("feature must be a DirectionFeature object")
    if not isinstance(axis,  URBasic.UrScriptExt.DirectionAxis):
      raise ValueError("axis must be a DirectionAxis object")
    
    # Retrieve current position
    current_pos = deepcopy(from_pose.toList())
    # If feature is Base, move the current_pos in its X/Y/Z direction based on the axis variable
    if feature == self.DirectionFeature.Base:
      match axis:
        case self.DirectionAxis.Xp:
          current_pos[0] += dist_mm / 1000
        case self.DirectionAxis.Xn:
          current_pos[0] -= dist_mm / 1000
        case self.DirectionAxis.Yp:
          current_pos[1] += dist_mm / 1000
        case self.DirectionAxis.Yn:
          current_pos[1] -= dist_mm / 1000
        case self.DirectionAxis.Zp:
          current_pos[2] += dist_mm / 1000
        case self.DirectionAxis.Zn:
          current_pos[2] -= dist_mm / 1000
        case  _ :
          raise ValueError("Invalid axis value")
    # If feature is Tool, move the current_pos based on its current rotation and the axis variable
    elif feature == self.DirectionFeature.Tool:
      # Get vector of the translation
      match axis:
        case self.DirectionAxis.Xp:
          tr = [dist_mm / 1000, 0, 0, 1]
        case self.DirectionAxis.Xn:
          tr = [-dist_mm / 1000, 0, 0, 1]
        case self.DirectionAxis.Yp:
          tr = [0, dist_mm / 1000, 0, 1]
        case self.DirectionAxis.Yn:
          tr = [0, -dist_mm / 1000, 0, 1]
        case self.DirectionAxis.Zp:
          tr = [0, 0, dist_mm / 1000, 1]
        case self.DirectionAxis.Zn:
          tr = [0, 0, -dist_mm / 1000, 1]
        case  _ :
          raise ValueError("Invalid axis value")
      # Transformation matrix from current_pose
      matrix = UrScriptExt.transformationMatrixFromAxisAngle(*current_pos)
      # Apply translation to the matrix
      new_pos = np.dot(matrix, tr)
      # Update current_pos
      current_pos[:3] = new_pos[:3]
    # Unknown
    else:
      raise ValueError("Invalid feature value")
    
    # Move to the new position
    current_pos = TCP6D.createFromMetersRadians(*current_pos)
    self.movel(pose=current_pos, a=accel, v=velocity)
