import socket
from time import sleep

class RobotiqTwoFingersGripper:
  '''
  Class to control the Robotiq Two Finger Gripper
  '''

  '''
  Socket definitions for the robot to acces the gripper through URScript
  '''
  __SOCKET_PORT = 63352

  def __init__(self, host : str, opened_size_mm : float) -> None:
    '''Constructor.
    Setup the socket to connect to the gripper.'''
    self.__opened_size_mm = opened_size_mm
    try:
      self.__sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)            
      self.__sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)         
      self.__sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
      self.__sock.settimeout(2.0)
      self.__sockTimeout = 2.0
      self.__sock.connect((host, RobotiqTwoFingersGripper.__SOCKET_PORT))
    except Exception as e:
      self.__sock = None
      raise e
  
  def __del__(self) -> None:
    try:
      self.__sock.close()
      self.__sock = None
    except Exception as e:
      print(f'Error deleting gripper: {e}')

  def __sendVar(self, var : str, value : str) -> bool:
    '''Set a variable on the gripper through the socket'''
    self.__clearSocket()
    self.__sock.sendall(f"SET {var} {value}\n".encode(encoding='ascii'))
    resp = self.__getUntilDelim(b'k')
    return 'ack' in resp
  
  def __clearSocket(self) -> None:
    '''Clear the socket from any data'''
    self.__sock.settimeout(0.02)
    while True:
      try:
        rec = self.__sock.recv(2048)
        if not rec or len(rec) < 2048:
          break
      except socket.timeout:
        break
    self.__sock.settimeout(self.__sockTimeout)

  def __getUntilDelim(self, delim : bytes = b'\n') -> str:
    '''Get data from the socket until a delimiter is found'''
    resp = b''
    try:
      while True:
        rb = self.__sock.recv(1)
        # If no delimiter => invalid data
        if not rb:
          return ''
        resp += rb
        if rb == delim:
          break
      # print(f'Read socket: {resp}')
    except socket.timeout:
      return ''
    return str(resp, 'ascii')

  def __getVar(self, var) -> str:
    '''Get a variable from the gripper through the socket'''
    self.__clearSocket()
    self.__sock.sendall(f"GET {var}\n".encode(encoding='ascii'))
    return self.__getUntilDelim()[:-1] # remove the newline at the end

  def activate(self) -> bool:
    '''
    Activate the gripper
    Required once before running any operation
    '''
    if self.__sendVar('ACT', 1):
      return self.__sendVar('GTO', 1)
    return False

  def deactivate(self) -> bool:
    '''
    Deactivate the gripper
    '''
    return self.__sendVar('ACT', 0)
  
  def isActivated(self) -> bool:
    '''
    Check if the gripper is activated
    '''
    return self.getACT() == 1 and self.getSTA() == 3
  
  def waitUntilActive(self) -> None:
    '''
    Wait until the gripper is active
    '''
    while not self.isActivated():
      sleep(0.02)
  
  def isMoving(self) -> bool:
    '''
    Check if the gripper is moving
    '''
    return self.getOBJ() == 0
  
  def waitUntilStopped(self) -> None:
    '''
    Wait until the gripper is stopped
    '''
    while self.isMoving():
      sleep(0.02)

  def move(self, pos : int, speed : int = 255, force : int = 50) -> bool:
    '''
    Move the gripper to a position

    @param pos: Position to move to (0-255, 0 being open (50mm), 255 being closed (0mm)). Around 0.2mm per unit.
    @param speed: Speed of the gripper (0-255, 0 being slowest)
    @param force: Force of the gripper (0-255, 0 being weakest)
    '''
    if not self.isActivated():
      return False
    if self.__sendVar('FOR', int(force)):
      if self.__sendVar('SPE', int(speed)):
        return self.__sendVar('POS', int(pos))
    return False
  
  def open(self, speed : int = 255, force : int = 50) -> bool:
    '''
    Open the gripper

    @param speed: Speed of the gripper (0-255, 0 being slowest)
    @param force: Force of the gripper (0-255, 0 being weakest)
    '''
    return self.move(0, speed, force)
  
  def close(self, speed : int = 255, force : int = 50) -> bool:
    '''
    Close the gripper

    @param speed: Speed of the gripper (0-255, 0 being slowest)
    @param force: Force of the gripper (0-255, 0 being weakest)
    '''
    return self.move(255, speed, force)
  
  def hasDetectedObject(self) -> bool:
    '''
    Check if the gripper has an object.
    This is not reliable for small objects, use closeAndCheckSize / openAndCheckSize instead.
    '''
    gobj = self.getOBJ()
    return gobj == 1 or gobj == 2
  
  def closeAndCheckSize(self, expected_size_mm : float, plus_margin_mm : float,
      minus_margin_mm : float, speed : int = 255, force : int = 50) -> bool:
    """
    Close the gripper and check if the size is within the expected range.
    May be useful for object detection since the gOBJ status is not always reliable.
    
    Args:
      expected_size_mm: The expected size of the object in mm.
      plus_margin_mm: The positive margin to allow.
      minus_margin_mm: The negative margin to allow.
      speed: The speed of the gripper.
      force: The force of the gripper.
      
    Returns:
      True if the size is within the expected range, False otherwise.
    """
    if not self.close(speed, force):
      return False
    sleep(0.3)
    self.waitUntilStopped()
    pos = self.getEstimatedPositionMm()
    return pos >= expected_size_mm - abs(minus_margin_mm) and pos <= expected_size_mm + abs(plus_margin_mm)
  
  def openAndCheckSize(self, expected_size_mm : float, plus_margin_mm : float,
      minus_margin_mm : float, speed : int = 255, force : int = 50) -> bool:
    """
    Open the gripper and check if the size is within the expected range.
    May be useful for object detection since the gOBJ status is not always reliable.

    Args:
      expected_size_mm: The expected size of the object in mm.
      plus_margin_mm: The positive margin to allow.
      minus_margin_mm: The negative margin to allow.
      speed: The speed of the gripper.
      force: The force of the gripper.

    Returns:
      True if the size is within the expected range, False otherwise.
    """
    if not self.open(speed, force):
      return False
    sleep(0.3)
    self.waitUntilStopped()
    pos = self.getEstimatedPositionMm()
    return pos >= expected_size_mm - abs(minus_margin_mm) and pos <= expected_size_mm + abs(plus_margin_mm)
  
  def getPosition(self) -> int:
    '''
    Get the position of the gripper (0-255, 0 being open, 255 being closed)
    '''
    resp = self.__getVar('POS').split() # responds "POS xyzzzzz"
    return int(resp[1], 10)
  
  def getEstimatedPositionMm(self) -> float:
    '''
    Get the estimated position of the gripper in mm
    '''
    pos = (250 - self.getPosition()) * (self.__opened_size_mm / 247)
    if pos > self.__opened_size_mm:
      pos = self.__opened_size_mm
    elif pos < 0:
      pos = 0.0
    return pos
  
  def getCurrent(self) -> int:
    '''
    Get the current of the gripper
    Real value is arnd. 10 * register [mA] and can be read only while is in motion
    '''
    resp = self.__getVar('CUR').split() # responds "CUR xyzzzzz"
    try:
      resp = int(resp[1], 10)
    except ValueError: # in case of unknown value, e.g. when not moving
      resp = 0
    return resp
  
  def getEstimatedCurrentMA(self) -> float:
    '''
    Get the estimated current of the gripper in mA
    '''
    return self.getCurrent() * 10.0
    

  class Status:
    '''
    Status representing the gripper
    '''

    def __init__(self, gOBJ : int, gSTA : int, gTO : int, gACT : int) -> None:
      self.gOBJ = gOBJ
      self.gSTA = gSTA
      self.gTO = gTO
      self.gACT = gACT
    
    def __str__(self) -> str:
      return ('Status:\n'
        f' - {self._statusToGobjstr(self.gOBJ)}\n'
        f' - {self._statusToGstastr(self.gSTA)}\n'
        f' - {self._statusToGtostr(self.gTO)}\n'
        f' - {self._statusToGactstr(self.gACT)}'
      )

    def __repr__(self) -> str:
      return self.__str__()
      
    def _statusToGobjstr(self, gobj : int) -> str:
      '''
      Convert the status of the gripper to a string isolating gOBJ
      '''
      objstr = 'gOBJ: '
      match gobj:
        case 0:
          objstr += 'In motion towards requested position.'
        case 1:
          objstr += 'Stopped due to a contact while opening.'
        case 2:
          objstr += 'Stopped due to a contact while closing.'
        case 3:
          objstr += 'Arrived to requested position. No object detected.'
        case _:
          objstr += 'INVALID gOBJ !'
      return objstr
    
    def _statusToGstastr(self, gsta : int) -> str:
      '''
      Convert the status of the gripper to a string isolating gSTA
      '''
      stastr = 'gSTA: '
      match gsta:
        case 0:
          stastr += 'Gripper is in reset ( or automatic release ) state.'
        case 1:
          stastr += 'Activation in progress.'
        case 2:
          stastr += 'Not used.'
        case 3:
          stastr += 'Activation is completed.'
        case _:
          stastr += 'INVALID gSTA !'
      return stastr
    
    def _statusToGtostr(self, gto : int) -> str:
      '''
      Convert the status of the gripper to a string isolating gTO
      '''
      gtostr = 'gTO: '
      match gto:
        case 0:
          gtostr += 'Stopped (or peforming activation / automatic release).'
        case 1:
          gtostr += 'Go to Position Request mode.'
        case _:
          gtostr += 'INVALID gTO !'
      return gtostr
    
    def _statusToGactstr(self, gact : int) -> str:
      '''
      Convert the status of the gripper to a string isolating gACT
      '''
      gactstr = 'gACT: '
      match gact:
        case 0:
          gactstr += 'Gripper stopped.'
        case 1:
          gactstr += 'Gripper active.'
        case _:
          gactstr += 'INVALID gACT !'
      return gactstr

  def getStatus(self) -> Status:
    '''
    Get the status of the gripper
    '''
    return RobotiqTwoFingersGripper.Status(self.getOBJ(), self.getSTA(), self.getGTO(), self.getACT())
  
  def getOBJ(self) -> int:
    '''
    Get the object status of the gripper
    '''
    resp = self.__getVar('OBJ').split() # responds "OBJ x"
    return int(resp[1], 16)
  
  def getSTA(self) -> int:
    '''
    Get the status of the gripper
    '''
    resp = self.__getVar('STA').split() # responds "STA x"
    return int(resp[1], 16)
  
  def getGTO(self) -> int:
    '''
    Get the go-to status of the gripper
    '''
    resp = self.__getVar('GTO').split() # responds "GTO x"
    return int(resp[1], 16)
  
  def getACT(self) -> int:
    '''
    Get the activation status of the gripper
    '''
    resp = self.__getVar('ACT').split() # responds "ACT x"
    return int(resp[1], 16)
  

  class Fault:
    '''
    Gripper fault representation
    '''

    def __init__(self, kflt : int, gflt : int) -> None:
      self.gFLT = gflt
      self.kFLT = kflt

    def __str__(self) -> str:
      return ('Fault:\n'
        f' - {self._faultToGfltstr(self.gFLT)}\n'
        f' - kFLT: {self.kFLT} (see optional controller manual)'
      )
    
    def __repr__(self) -> str:
      return self.__str__()
    
    def _faultToGfltstr(self, gflt : int) -> str:
      '''
      Convert the fault of the gripper to a string isolating gFLT
      '''
      fltstr = 'gFLT: '
      match gflt:
        case 0:
          fltstr += 'No Fault (solid blue LED).'
        case 5:
          fltstr += 'Action delayed; the activation (re-activation) must be completed prior to perform the action (solid blue LED).'
        case 7:
          fltstr += 'The activation bit must be set prior to performing the action (solid blue LED).'
        case 8:
          fltstr += 'Maximum operating temperature exceeded (≥ 85 °C internally); let cool down (below 80 °C) (solid red LED).'
        case 9:
          fltstr += 'No communication during at least 1 second.'
        case 10:
          fltstr += 'Under minimum operating voltag (red/blue LEDs blinking - need reset).'
        case 11:
          fltstr += 'Automatic release in progress (red/blue LEDs blinking - need reset).'
        case 12:
          fltstr += 'Internal fault, contact support@robotiq.com (red/blue LEDs blinking - need reset).'
        case 13:
          fltstr += 'Activation fault, verify that no interference or other error occurred (red/blue LEDs blinking - need reset).'
        case 14:
          fltstr += 'Overcurrent triggered (red/blue LEDs blinking - need reset).'
        case 15:
          fltstr += 'Automatic release completed (red/blue LEDs blinking - need reset).'
        case _:
          fltstr += 'INVALID gFLT !'
      return fltstr

  def getFault(self) -> Fault:
    '''
    Get the fault status of the gripper
    '''
    resp = self.__getVar('FLT').split() # responds "FLT xx"
    return RobotiqTwoFingersGripper.Fault(int(resp[1][0], 16), int(resp[1][1], 16))
