'''
Python 3.x superclass to control an UR robot along a Robotiq two finger gipper and a Robotiq wrist camera through remote connection.
The robot must be connected to the same network as the application running this library.

The same IP address is used to connect to the three devices.
Ports are automatically setup to reach the different services.
'''
__author__ = "Amand Axel"
__copyright__ = "Copyright 2024, HES-SO Valais/Wallis, Suisse"
__license__ = "MIT License"

import logging

from .robotModel import RobotModel
from .urScriptExt import UrScriptExt
from .devices import RobotiqTwoFingersGripper, RobotiqWristCamera
from .waypoint6d import Joint6D
import URBasic

import cv2
import threading
import time
from traceback import format_exc


class ISCoin():
  
  _instance : 'ISCoin' = None
  
  def __new__(cls, *args, **kwargs) -> 'ISCoin':
    if cls._instance is not None:
      cls._instance.__del__()
    cls._instance = super(ISCoin, cls).__new__(cls)
    return cls._instance

  def __init__(self, host : str, opened_gripper_size_mm : float = 40.0, logging_level : int | str = logging.DEBUG) -> None :
    """ Constructor.
    Set various devices up.

    Args:
      ip: The IP address of the robot.
      logging_level: The logging level to use for the current ISCoin object.
    """
    self._init_ok : bool = False
    self._robot_model : RobotModel = None
    self._script : UrScriptExt = None
    self._gripper : RobotiqTwoFingersGripper = None
    self._camera : RobotiqWristCamera = None
    self._opened_gripper_size_mm : float = opened_gripper_size_mm
    self._stream_thread : threading.Thread = None
    self._stream_thread_killer : threading.Event = threading.Event()
    # Setup logger
      # Override all modules config to use the same format
    logging.basicConfig(format='%(asctime)s,%(msecs)03d | %(name)s | %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s') #, level = logging.DEBUG)
    self._logger = logging.getLogger(self.__class__.__name__ + " - " + host)
    self._logger.setLevel(logging_level)
    # Setup devices
    try:
      self._logger.debug("Initializing devices in constructor")
      self.__reinitDevices(host)
      self._init_ok = True
    except Exception as e:
      self._logger.error(f"ISCoin initialization failed:\n {format_exc()}")
      self.__terminateDevices()
      raise e

  def __del__(self) -> None:
    """ Destructor.
    Close all devices.
    """
    self.__terminateDevices()

  def __terminateDevices(self) -> None:
    """ Close all devices.
    """
    try:
      if self._script:
        self._script.close()
    except Exception as e:
      print(f"Error while closing script: {e}")
    self._script = None
    self._robot_model = None
    self._gripper = None
    self._camera = None
    if self._stream_thread:
      self._stream_thread_killer.set()
      self._stream_thread.join()
      self._stream_thread = None
      self._stream_thread_killer.clear()

  def close(self) -> None:
    """ Close all devices.
    """
    self.__terminateDevices()

  def __reinitDevices(self, ip : str) -> None:
    """ Setup all devices linked to the robot
      
    Args:
      ip: The IP address of the robot.
    Exceptions:
      Exception: If the initialization of a device failed.
    """
    self.__terminateDevices()
    try:
      self._logger.info(f"Reinitializing devices with IP {ip}\n * Robot model")
      self._robot_model = RobotModel()
      self._logger.info(f" * URScriptExt")
      self._script = UrScriptExt(host = ip, robotModel = self._robot_model)
      self._logger.info(f" * RobotiqTwoFingersGripper")
      self._gripper = RobotiqTwoFingersGripper(host = ip, opened_size_mm = self._opened_gripper_size_mm)
      self._logger.info(f" * RobotiqWristCamera")
      self._camera = RobotiqWristCamera(host = ip)
      self._logger.info(" *** Devices initialized")
    except Exception as e:
      self._logger.error(f"Device initialization failed:\n {format_exc()}")
      self.__terminateDevices()
      raise e

  @property
  def isInitOk(self) -> bool:
    """ Check if the initialization was successful.

    Returns:
      True if the initialization was successful, False otherwise.
    """
    return self._init_ok
  
  @property
  def robot_control(self) -> UrScriptExt | None:
    """ Get the robot model handler.
    
    Returns:
      The robot model handler if the initialization was successful, None otherwise.
    """
    if self.isInitOk:
      return self._script
    else:
      self._logger.error("Cannot get robot controller as ISCoin initialization failed")
    return None
  
  @property
  def gripper(self) -> RobotiqTwoFingersGripper | None:
    """ Get the Robotiq two finger gripper handler.
    
    Returns:
      The Robotiq two finger gripper handler if the initialization was successful, None otherwise.
    """
    if self.isInitOk:
      return self._gripper
    else:
      self._logger.error("Cannot get gripper as ISCoin initialization failed")
    return None
  
  @property
  def camera(self) -> RobotiqWristCamera | None:
    """ Get the Robotiq wrist camera handler.
    
    Returns:
      The Robotiq wrist camera handler if the initialization was successful, None otherwise.
    """
    if self.isInitOk:
      return self._camera
    else:
      self._logger.error("Cannot get camera as ISCoin initialization failed")
    return None
  
  def displayCameraOCV(self, image_type : RobotiqWristCamera.ImageType = RobotiqWristCamera.ImageType.COLOR) -> None:
    """ Display the last captured frame of the camera using OpenCV.
    BLOCKING FUNCTION.

    Args:
      image_type: The type of image to display based on the RobotiqWristCamera.ImageType enum.
    """
    if self.isInitOk:
      cv2.imshow('Camera Frame', self._camera.getImageAsOpenCvNpArray(type = image_type))
      cv2.waitKey(0)
      cv2.destroyAllWindows()
    else:
      self._logger.error("Cannot display camera frame as ISCoin initialization failed")

  def displayCameraStreamOCV(self, image_type : RobotiqWristCamera.ImageType = RobotiqWristCamera.ImageType.COLOR, streamKiller : threading.Event | None = None) -> None:
    """ Display the camera stream using OpenCV.
    BLOCKING FUNCTION. Press 'q' on the opened window to close it.

    Args:
      image_type: The type of image to display based on the RobotiqWristCamera.ImageType enum.
    """
    if streamKiller is None:
      streamKiller = threading.Event()
    if self.isInitOk:
      while streamKiller.is_set() == False:
        try:
          cv2.imshow('Camera Frame', self._camera.getImageAsOpenCvNpArray(type = image_type))
          if cv2.waitKey(1) & 0xFF == ord('q'):
            break
          try:
            if cv2.getWindowProperty('Camera Frame', 0) < 0:
              break
          except:
            break
        except:
          pass
        time.sleep(0.05)
      cv2.destroyAllWindows()
    else:
      self._logger.error("Cannot display camera stream as ISCoin initialization failed")
      
  def displayCameraStreamOCVParallel(self, image_type : RobotiqWristCamera.ImageType = RobotiqWristCamera.ImageType.COLOR) -> None:
    """ Display the camera stream using OpenCV in a parallel thread.
    NON-BLOCKING FUNCTION. Press 'q' on the opened window to close it.

    Args:
      image_type: The type of image to display based on the RobotiqWristCamera.ImageType enum.
    """
    if self.isInitOk:
      self._logger.debug('Init ok - starting OCV stream')
      if self._stream_thread:
        self._logger.debug(' * Thread already exists - stopping it')
        self._stream_thread_killer.set()
        self._stream_thread.join()
        self._stream_thread_killer.clear()
      self._stream_thread = threading.Thread(target=self.displayCameraStreamOCV, args=(image_type, self._stream_thread_killer))
      self._logger.debug(' * Starting thread')
      self._stream_thread.start()
    else:
      self._logger.error("Cannot display camera stream as ISCoin initialization failed")
