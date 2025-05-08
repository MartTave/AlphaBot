'''
A 6D list holder for waypoints.

To be used with the ISCoin specific functions only for now.
'''
__author__ = "Amand Axel"
__copyright__ = "Copyright 2024, HES-SO Valais/Wallis, Suisse"
__license__ = "MIT License"

from math import radians
import URBasic

class GenericWaypoint6D:
  """ Generic class to store a 6D waypoint
  """

  def __init__(self, *args) -> None:
    """ Constructor.
    """
    if len(args) == 1 and isinstance(args[0], list):
      self.__initWp6D(args[0])
    elif len(args) == 6:
      self.__initValues(*args)
    else:
      raise ValueError("Invalid arguments for GenericWaypoint6D constructor")

  def __initWp6D(self, wp : list[float]) -> None:
    """ Constructor.
    
    Args:
      GenericWaypoint6D: The 6D waypoint to store.
    """
    self._wp = wp
  
  def __initValues(self, x : float, y : float, z : float, rx : float, ry : float, rz : float) -> None:
    """ Constructor.
    
    Args:
      x: The x coordinate of the waypoint.
      y: The y coordinate of the waypoint.
      z: The z coordinate of the waypoint.
      rx: The rotation around the x axis of the waypoint.
      ry: The rotation around the y axis of the waypoint.
      rz: The rotation around the z axis of the waypoint.
    """
    self._wp = [float(x), float(y), float(z), float(rx), float(ry), float(rz)]

  def __getitem__(self, index : int) -> float:
    """ Get an element of the waypoint.
    
    Args:
      index: The index of the element to get.
    
    Returns:
      The element at the given index.
    """
    return self._wp[index]
  
  def __setitem__(self, index : int, value : float) -> None:
    """ Set an element of the waypoint.
    
    Args:
      index: The index of the element to set.
      value: The value to set at the given index.
    """
    self._wp[index] = value
  
  def __str__(self) -> str:
    """ Get the string representation of the waypoint.
    
    Returns:
      The string representation of the waypoint.
    """
    return f"GenericWaypoint6D({self._wp})"
  
  def __repr__(self) -> str:
    """ Get the string representation of the waypoint.
    """
    return self.__str__()
  
  def toList(self) -> list[float]:
    """ Get the waypoint as a list.
    
    Returns:
      The waypoint as a list.
    """
    return self._wp
  

class TCP6D(GenericWaypoint6D):
  """ Class to store a 6D TCP waypoint.
  """

  def __init__(self, *args) -> None:
    """ Constructor.
    Should never be called outside of this module.
    """
    if not hasattr(self, '_initialized'):
      raise RuntimeError("Use a dedicated class static method to create instances of this class.")
    super().__init__(*args)
    
  @property
  def x(self) -> float:
    """ Get the x coordinate of the TCP waypoint.
    
    Returns:
      The x coordinate of the TCP waypoint.
    """
    return self._wp[0]
  
  @property
  def y(self) -> float:
    """ Get the y coordinate of the TCP waypoint.
    
    Returns:
      The y coordinate of the TCP waypoint.
    """
    return self._wp[1]
  
  @property
  def z(self) -> float:
    """ Get the z coordinate of the TCP waypoint.
    
    Returns:
      The z coordinate of the TCP waypoint.
    """
    return self._wp[2]
  
  @property
  def rx(self) -> float:
    """ Get the rotation around the x axis of the TCP waypoint.
    
    Returns:
      The rotation around the x axis of the TCP waypoint.
    """
    return self._wp[3]
  
  @property
  def ry(self) -> float:
    """ Get the rotation around the y axis of the TCP waypoint.
    
    Returns:
      The rotation around the y axis of the TCP waypoint.
    """
    return self._wp[4]
  
  @property
  def rz(self) -> float:
    """ Get the rotation around the z axis of the TCP waypoint.
    
    Returns:
      The rotation around the z axis of the TCP waypoint.
    """
    return self._wp[5]
  
  @staticmethod
  def createFromMetersRadians(x : float, y : float, z : float, rx : float, ry : float, rz : float) -> 'TCP6D':
    """ Create a TCP waypoint from values in meters and radians.
    
    Args:
      x: The x coordinate of the TCP waypoint [m]
      y: The y coordinate of the TCP waypoint [m]
      z: The z coordinate of the TCP waypoint [m]
      rx: The rotation around the x axis of the TCP waypoint [rad]
      ry: The rotation around the y axis of the TCP waypoint [rad]
      rz: The rotation around the z axis of the TCP waypoint [rad]
    
    Returns:
      A new TCP waypoint with the given values.
    """
    instance = TCP6D.__new__(TCP6D)
    instance._initialized = True
    instance.__init__(x, y, z, rx, ry, rz)
    return instance
  
  @staticmethod
  def createFromMillietersRadians(x : float, y : float, z : float, rx : float, ry : float, rz : float) -> 'TCP6D':
    """ Create a TCP waypoint from values in mm and radians.
    
    Args:
      x: The x coordinate of the TCP waypoint [mm]
      y: The y coordinate of the TCP waypoint [mm]
      z: The z coordinate of the TCP waypoint [mm]
      rx: The rotation around the x axis of the TCP waypoint [rad]
      ry: The rotation around the y axis of the TCP waypoint [rad]
      rz: The rotation around the z axis of the TCP waypoint [rad]
      
    Returns:
      A new TCP waypoint with the given values.
    """
    instance = TCP6D.__new__(TCP6D)
    instance._initialized = True
    instance.__init__(x / 1000.0, y / 1000.0, z / 1000.0, rx, ry, rz)
    return instance
  
  def __add__(self, other : 'TCP6D') -> 'TCP6D':
    """ Add two waypoints together.
    
    Args:
      other: The other waypoint to add.
    
    Returns:
      A new waypoint which is the sum of the two waypoints.
    """
    # Add the three first values and multiply the three last together
    return TCP6D.createFromMetersRadians(*[self._wp[i] + other._wp[i] if i < 3 else self._wp[i] * other._wp[i] for i in range(6)])
  
  def __str__(self) -> str:
    """ Get the string representation of the TCP waypoint.
    
    Returns:
      The string representation of the TCP waypoint.
    """
    return f"TCP6D({self._wp})"
  
  def __repr__(self) -> str:
    """ Get the string representation of the TCP waypoint.
    """
    return self.__str__()
  
class TCP6DDescriptor():
  ''' A class to store a 6D TCP waypoint descriptor.
  Used with functions like movel_waypoints.
  '''
  
  def __init__(self, tcp : TCP6D, a : float = 1.2, v : float = 0.25, t : float = 0, r : float = 0):
    ''' Constructor.
    
    Args:
      tcp: The TCP waypoint to store.
      a: The acceleration of the waypoints [m/s^2].
      v: The velocity of the waypoints [m/s].
      t: The time of the waypoints [s].
      r: The radius of the waypoints [m]
    '''
    self.tcp = tcp
    self.a = a
    self.v = v
    self.t = t
    self.r = r
    
  def getAsDict(self) -> dict:
    ''' Get the descriptor as a dictionary.
    
    Returns:
      The descriptor as a dictionary.
    '''
    return {"pose": self.tcp.toList(), "a": self.a, "v": self.v, "t": self.t, "r": self.r}
  
  @staticmethod
  def createFromTCPList(tcp_list : list[TCP6D], a : float = 1.2, v : float = 0.25, t : float = 0, r : float = 0) -> list['TCP6DDescriptor']:
    ''' Create a list of TCP6DDescriptor from a list of TCP6D.
    
    Args:
      tcp_list: The list of TCP6D to create the descriptors from.
      a: The acceleration of the waypoints.
      v: The velocity of the waypoints.
      t: The time of the waypoints.
      r: The radius of the waypoints.
    
    Returns:
      A list of TCP6DDescriptor.
    '''
    if not isinstance(tcp_list, list):
      raise ValueError("tcp_list must be a list")
      return []
    for t in tcp_list:
      if not isinstance(t, URBasic.waypoint6d.TCP6D):
        raise ValueError("tcp_list must be a list of TCP6D - at least one element is not a TCP6D")
        return []
    return [TCP6DDescriptor(t, a, v, t, r) for t in tcp_list]

class Joint6D(GenericWaypoint6D):
  """ Class to store a 6D joint waypoint.
  """

  def __init__(self, *args) -> None:
    """ Constructor.
    Should never be called outside of this module.
    """
    if not hasattr(self, '_initialized'):
      raise RuntimeError("Use a dedicated class static method to create instances of this class.")
    super().__init__(*args)
    
  @property
  def j1(self) -> float:
    """ Get the first joint value of the joint waypoint.
    
    Returns:
      The first joint value of the joint waypoint.
    """
    return self._wp[0]
  
  @property
  def j2(self) -> float:
    """ Get the second joint value of the joint waypoint.
    
    Returns:
      The second joint value of the joint waypoint.
    """
    return self._wp[1]
  
  @property
  def j3(self) -> float:
    """ Get the third joint value of the joint waypoint.
    
    Returns:
      The third joint value of the joint waypoint.
    """
    return self._wp[2]
  
  @property
  def j4(self) -> float:
    """ Get the fourth joint value of the joint waypoint.
    
    Returns:
      The fourth joint value of the joint waypoint.
    """
    return self._wp[3]
  
  @property
  def j5(self) -> float:
    """ Get the fifth joint value of the joint waypoint.
    
    Returns:
      The fifth joint value of the joint waypoint.
    """
    return self._wp[4]
  
  @property
  def j6(self) -> float:
    """ Get the sixth joint value of the joint waypoint.
    
    Returns:
      The sixth joint value of the joint waypoint.
    """
    return self._wp[5]
  
  @staticmethod
  def createFromRadians(j1 : float, j2 : float, j3 : float, j4 : float, j5 : float, j6 : float) -> 'Joint6D':
    """ Create a joint waypoint from values.
    
    Args:
      j1: The first joint value of the joint waypoint [rad]
      j2: The second joint value of the joint waypoint [rad]
      j3: The third joint value of the joint waypoint [rad]
      j4: The fourth joint value of the joint waypoint [rad]
      j5: The fifth joint value of the joint waypoint [rad]
      j6: The sixth joint value of the joint waypoint [rad]
    
    Returns:
      A new joint waypoint with the given values.
    """
    instance = Joint6D.__new__(Joint6D)
    instance._initialized = True
    instance.__init__(j1, j2, j3, j4, j5, j6)
    return instance
  
  @staticmethod
  def createFromRadList(jList : list[float]) -> 'Joint6D':
    """ Create a joint waypoint from values.
    
    Args:
      j1: The first joint value of the joint waypoint [rad]
      j2: The second joint value of the joint waypoint [rad]
      j3: The third joint value of the joint waypoint [rad]
      j4: The fourth joint value of the joint waypoint [rad]
      j5: The fifth joint value of the joint waypoint [rad]
      j6: The sixth joint value of the joint waypoint [rad]
      jList: a list containing the six joint waypoints, from j1 to j6 (so [j1,j2,j3,j4,j5,j6]) [rad,rad,rad,rad,rad,rad]
    
    Returns:
      A new joint waypoint with the given values.
    """
    instance = Joint6D.__new__(Joint6D)
    instance._initialized = True
    instance.__init__(jList[0], jList[1], jList[2], jList[3], jList[4], jList[5])
    return instance

  @staticmethod
  def createFromDegList(jList : list[float]) -> 'Joint6D':
    """ Create a joint waypoint from values.
    
    Args:
      j1: The first joint value of the joint waypoint [deg]
      j2: The second joint value of the joint waypoint [deg]
      j3: The third joint value of the joint waypoint [deg]
      j4: The fourth joint value of the joint waypoint [deg]
      j5: The fifth joint value of the joint waypoint [deg]
      j6: The sixth joint value of the joint waypoint [deg]
      jList: a list containing the six joint waypoints, from j1 to j6 (so [j1,j2,j3,j4,j5,j6]) [deg,deg,deg,deg,deg,deg]
    
    Returns:
      A new joint waypoint with the given values.
    """
    instance = Joint6D.__new__(Joint6D)
    instance._initialized = True
    instance.__init__(radians(jList[0]), radians(jList[1]), radians(jList[2]), radians(jList[3]), radians(jList[4]), radians(jList[5]))
    return instance

  @staticmethod
  def createFromDegrees(j1 : float, j2 : float, j3 : float, j4 : float, j5 : float, j6 : float) -> 'Joint6D':
    """ Create a joint waypoint from values.
    
    Args:
      j1: The first joint value of the joint waypoint [deg]
      j2: The second joint value of the joint waypoint [deg]
      j3: The third joint value of the joint waypoint [deg]
      j4: The fourth joint value of the joint waypoint [deg]
      j5: The fifth joint value of the joint waypoint [deg]
      j6: The sixth joint value of the joint waypoint [deg]
    
    Returns:
      A new joint waypoint with the given values.
    """
    instance = Joint6D.__new__(Joint6D)
    instance._initialized = True
    instance.__init__(radians(j1), radians(j2), radians(j3), radians(j4), radians(j5), radians(j6))
    return instance
  
  def __add__(self, other : 'Joint6D') -> 'Joint6D':
    """ Add two waypoints together.
    
    Args:
      other: The other waypoint to add.
    
    Returns:
      A new waypoint which is the sum of the two waypoints.
    """
    return Joint6D.createFromRadians(*[self._wp[i] + other._wp[i] for i in range(6)])
  
  def __sub__(self, other : 'Joint6D') -> 'Joint6D':
    """ Subtract two waypoints together.
    
    Args:
      other: The other waypoint to subtract.
    
    Returns:
      A new waypoint which is the difference of the two waypoints.
    """
    return Joint6D.createFromRadians(*[self._wp[i] - other._wp[i] for i in range(6)])
  
  def __mul__(self, scalar : float) -> 'Joint6D':
    """ Multiply a waypoint by a scalar.
    
    Args:
      scalar: The scalar to multiply the waypoint by.
    
    Returns:
      A new waypoint which is the waypoint multiplied by the scalar.
    """
    return Joint6D.createFromRadians(*[self._wp[i] * scalar for i in range(6)])
  
  def __truediv__(self, scalar : float) -> 'Joint6D':
    """ Divide a waypoint by a scalar.
    
    Args:
      scalar: The scalar to divide the waypoint by.
    
    Returns:
      A new waypoint which is the waypoint divided by the scalar.
    """
    return Joint6D.createFromRadians(*[self._wp[i] / scalar for i in range(6)])
  
  def __str__(self) -> str:
    """ Get the string representation of the joint waypoint.
    
    Returns:
      The string representation of the joint waypoint.
    """
    return f"Joint6D({self._wp})"
  
  def __repr__(self) -> str:
    """ Get the string representation of the joint waypoint.
    """
    return self.__str__()

class Joint6DDescriptor():
  ''' A class to store a 6D joint waypoint descriptor.
  Used with functions like movej_waypoints.
  '''
  
  def __init__(self, joints : Joint6D, a = 1.4, v : float = 2.05, t : float = 0, r : float = 0):
    ''' Constructor.
    
    Args:
      joints: The joint waypoint to store.
      a: The acceleration of the waypoints [rad/s^2].
      v: The velocity of the waypoints [rad/s].
      t: The time of the waypoints [s].
      r: The radius of the waypoints [m].
    '''
    self.joints = joints
    self.a = a
    self.v = v
    self.t = t
    self.r = r

  def getAsDict(self) -> dict:
    ''' Get the descriptor as a dictionary.
    '''
    return {"q": self.joints.toList(), "a": self.a, "v": self.v, "t": self.t, "r": self.r}
  
  @staticmethod
  def createFromJointsList(joints : list[Joint6D], a : float = 1.4, v : float = 2.05, t : float = 0, r : float = 0) -> list['Joint6DDescriptor']:
    ''' Create a list of Joint6DDescriptor from a list of Joint6D.
    
    Args:
      joints: The list of Joint6D to create the descriptors from.
      a: The acceleration of the waypoints.
      v: The velocity of the waypoints.
      t: The time of the waypoints.
      r: The radius of the waypoints.
    
    Returns:
      A list of Joint6DDescriptor.
    '''
    if not isinstance(joints, list):
      raise ValueError("joints must be a list")
      return []
    for j in joints:
      if not isinstance(j, URBasic.waypoint6d.Joint6D):
        raise ValueError("joints must be a list of Joint6D - at least one element is not a Joint6D")
        return []
    return [Joint6DDescriptor(j, a, v, t, r) for j in joints]
    