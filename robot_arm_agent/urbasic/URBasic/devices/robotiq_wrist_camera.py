import numpy as np
from PIL import Image
import io
import requests
from enum import Enum, auto
import cv2
from .camera_settings import CameraSettings

# http://10.30.5.158:4242/current.jpg?type=color
# http://10.30.5.158:4242/current.jpg?type=edges
# http://10.30.5.158:4242/current.jpg?type=magnitude
# http://10.30.5.158:4242/current.jpg?type=annotations

# http://10.30.5.158:4242/captureimage <= when program is running, forces the robot to take picture before watching it
# http://10.30.5.158:4242/current.jpg?annotations=<on|off>

class RobotiqWristCamera:
    '''Class to handle the Robotiq wrist camera.
    '''

    '''The port of the camera.'''
    __CAM_PORT = 4242

    def __init__(self, host):
      '''Constructor.
      Setup URL to connect to the camera.
      '''
      self._url = "http://" + host + ":" + str(RobotiqWristCamera.__CAM_PORT)
      self._imguri = "/current.jpg"

    class ImageType(Enum):
      '''Enum to define the type of image to get from the camera.
      '''
      COLOR = auto()
      EDGES = auto()
      MAGNITUDE = auto()

      def __init__(self, value) -> None:
        self._value = value

      def typeToUrl(self) -> str:
        '''Get the URL parameter for the image type.
        '''
        match self._value:
          case self.COLOR.value:
            return "?type=color"
          case self.EDGES.value:
            return "?type=edges"
          case self.MAGNITUDE.value:
            return "?type=magnitude"
          case _:
            return "?type=color"

    def getImageAsNpArray(self, type : ImageType = ImageType.COLOR) -> np.ndarray:
      '''Get the image from the camera as a numpy array.'''
      response = requests.get(self._url + self._imguri + type.typeToUrl()).content
      return np.asarray(bytearray(response), dtype="uint8")

    def getImageAsOpenCvNpArray(self, type : ImageType = ImageType.COLOR) -> np.ndarray:
      '''Get the image from the camera as an OpenCV ready numpy array.'''
      return cv2.imdecode(self.getImageAsNpArray(type), -1)

    def getImageAsImage(self, type : ImageType = ImageType.COLOR) -> Image:
      '''Get the image from the camera as a PIL Image.'''
      data = self.getImageAsNpArray(type)
      return Image.open(io.BytesIO(data))
    
    def __get_vision_server(self, uri) -> tuple[int, dict]:
      '''GET request to the vision server.'''
      if uri[0] != '/':
        uri = '/' + uri
      response = requests.get(self._url + uri)
      return response.status_code, response.json()
    
    def getCurrentCameraSettings(self) -> CameraSettings | None:
      '''Get the current camera settings.'''
      status, resp = self.__get_vision_server('getcamerasettings')
      if status != 200:
        return None
      return CameraSettings.createFromDict(resp['cameraSettings'])
    
    def resetCameraSettings(self) -> bool:
      '''Reset the camera settings.'''
      status, _ = self.__get_vision_server('setdefaultcamerasettings')
      return status == 200
    
    def setCameraSettings(self, settings : CameraSettings) -> bool:
      '''Set the camera settings.'''
      response = requests.post(self._url + '/setcamerasettings?setManualValuesIfModeChanged=true', data = settings.asPost(), headers={'Content-Type': 'text/plain'})
      return response.status_code == 200

    def __str__(self) -> str:
      return "RobotiqWristCamera on URL " + self._url

    def __repr__(self) -> str:
      return self.__str__()
