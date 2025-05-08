'''
Classes to hijack the camera settings when using it remotely.
'''
from enum import Enum
import json

class LightingSettings:
  '''Settings for the lighting of the camera.'''
  
  class LightingMode(Enum):
    '''Mode of the lighting of the camera'''
    AUTO = 0
    OFF = 1
    MANUAL = 2
    
  class LightingValue:
    '''Value for the lighting of the camera in manual mode. From 0 to 175.'''
    def __init__(self, value : int = 43):
      self.value = value
      
    def __str__(self):
      return f'LightingValue({self.__dict__})'
    
    def __repr__(self):
      return f'LightingValue({self.__dict__})'
    
    @property
    def value(self) -> int:
      return self.__value
    
    @value.setter
    def value(self, value : int):
      if value < 0 or value > 175:
        raise ValueError(f"LightingValue must be in the range 0 to 175 - got {value}")
      self.__value = value
      
    @property
    def valuePerc(self) -> int:
      return (self.__value / 175) * 100
    
    @valuePerc.setter
    def valuePerc(self, value : int):
      if value < 0 or value > 100:
        raise ValueError(f"LightingValue must be in the range 0 to 100 - got {value}")
      self.__value = int((value / 100) * 175)
    
  def __init__(self, mode: LightingMode = LightingMode.OFF, value: LightingValue = LightingValue()):
    self.__mode : LightingSettings.LightingMode = mode
    self.__value : LightingSettings.LightingValue = value
    
  def setAutoMode(self):
    '''Set the mode to auto.'''
    self.__mode = LightingSettings.LightingMode.AUTO
  
  def setOffMode(self):
    '''Set the mode to off.'''
    self.__mode = LightingSettings.LightingMode.OFF
    
  def setManualMode(self, valuePerc : int):
    '''Set the value for manual mode.
        Value must be in the range 0 to 100.'''
    if valuePerc < 0 or valuePerc > 100:
      raise ValueError(f"LightingValue must be in the range 0 to 100 - got {valuePerc}")
    self.__mode = LightingSettings.LightingMode.MANUAL
    self.__value.valuePerc = valuePerc
    
  @property
  def mode(self) -> int:
    return self.__mode.value
  
  @property
  def value(self) -> int:
    return self.__value.value

  def asPost(self) -> dict:
    '''Return the settings as a dict to be used in a POST request.'''
    return {'lightingMode': self.__mode.value, 'lightingValue': self.__value.value}
  
  def __str__(self):
    return f'LightingSettings({self.__dict__})'
  
  def __repr__(self):
    return f'LightingSettings({self.__dict__})'

    
class ExposureSettings():
  '''Settings for the exposure of the camera.'''
  
  class ExposureSensitivity():
    '''Sensitivity of the camera in auto mode. From -7 to 7.'''
    def __init__(self, sensitivity : int = 1):
      self.sensitivity = sensitivity
      
    def __str__(self):
      return f'ExposureSensitivity({self.__dict__})'
    
    def __repr__(self):
      return f'ExposureSensitivity({self.__dict__})'

    @property
    def sensitivity(self) -> int:
      return self.__sensitivity

    @sensitivity.setter
    def sensitivity(self, sensitivity : int):
      if sensitivity < -7 or sensitivity > 7:
        raise ValueError(f"ExposureSensitivity must be in the range -7 to 7 - got {sensitivity}")
      self.__sensitivity = sensitivity
      
  class ExposureTimeUs():
    '''Exposure time of the camera in manual mode. From 1 to 66604.'''
    def __init__(self, value : int = 1):
      self.value = value
      
    def __str__(self):
      return f'ExposureTimeUs({self.__dict__})'
    
    def __repr__(self):
      return f'ExposureTimeUs({self.__dict__})'
      
    @property
    def value(self) -> int:
      return self.__value
    
    @value.setter
    def value(self, value : int):
      if value < 1 or value > 66604:
        raise ValueError(f"Exposure must be in the range 1 to 66604 - got {value}")
      self.__value = value
      
  class Gain():
    '''Gain of the camera in manual mode. From 1 to 63.'''
    def __init__(self, value : int = 1):
      self.value = value
      
    def __str__(self):
      return f'Gain({self.__dict__})'
    
    def __repr__(self):
      return f'Gain({self.__dict__})'
      
    @property
    def value(self) -> int:
      return self.__value
    
    @value.setter
    def value(self, value : int):
      if value < 1 or value > 63:
        raise ValueError(f"Gain must be in the range 1 to 63 - got {value}")
      self.__value = value * 16
  
  def __init__(self, sensitivity: ExposureSensitivity = ExposureSensitivity(),
      exposure_time_use: ExposureTimeUs = ExposureTimeUs(), gain : Gain = Gain(), auto : bool = True):
    self.__sensitivity : ExposureSettings.ExposureSensitivity = sensitivity
    self.__exposure_time_us : ExposureSettings.ExposureTimeUs = exposure_time_use
    self.__gain : ExposureSettings.Gain = gain
    self.__auto_mode : bool = auto
    
  def setAutoMode(self, sensitivity : int):
    '''Set the sensitivity for auto mode.
       Sensitivity must be in the range -7 to 7.'''
    self.__sensitivity.sensitivity = sensitivity
    self.__auto_mode = True
    
  def setManualMode(self, exposure_time_us : int, gain : int):
    '''Set the exposure time and gain for manual mode.
       Exposure time must be in the range 1 to 66604.
       Gain must be in the range 1 to 63.'''
    self.__exposure_time_us.value = exposure_time_us
    self.__gain.value = gain
    self.__auto_mode = False
    
  def asPost(self) -> dict:
    '''Return the settings as a dict to be used in a POST request.'''
    return {'exposure': self.__exposure_time_us.value, 'exposureAuto': self.__auto_mode, 'exposureSensitivity': self.__sensitivity.sensitivity, 'gain': self.__gain.value}
  
  def __str__(self):
    return f'ExposureSettings({self.__dict__})'
  
  def __repr__(self):
    return f'ExposureSettings({self.__dict__})'
  
  @property
  def auto(self) -> bool:
    return self.__auto_mode
  
  @property
  def sensitivity(self) -> int:
    return self.__sensitivity.sensitivity
  
  @property
  def exposure_time_us(self) -> int:
    return self.__exposure_time_us.value
  
  @property
  def gain(self) -> int:
    return self.__gain.value
  
  

class FocusSettings:
  '''Settings for the focus of the camera.'''
  
  class FocusValue:
    '''Value for the focus of the camera in manual mode. From 0 to 1023.'''
    def __init__(self, value : int = 0):
      self.value = value
      
    def __str__(self):
      return f'FocusValue({self.__dict__})'
      
    def __repr__(self):
      return f'FocusValue({self.__dict__})'
      
    @property
    def value(self) -> int:
      return 1023 - self.__value
    
    @value.setter
    def value(self, value : int):
      if value < 0 or value > 1023:
        raise ValueError(f"FocusValue must be in the range 0 to 1023 - got {value}")
      self.__value = 1023 - value
  
  def __init__(self, auto : bool = True, focus : FocusValue = FocusValue()):
    self.__auto_mode : bool = auto
    self.__focus : FocusSettings.FocusValue = focus
    
  def setAutoMode(self):
    '''Set the focus to auto.'''
    self.__auto_mode = True
    
  def setManualMode(self, focus : FocusValue):
    '''Set the focus to manual.
       Focus must be in the range 0 to 1023.'''
    self.__auto_mode = False
    self.__focus = focus
    
  def asPost(self) -> dict:
    '''Return the settings as a dict to be used in a POST request.'''
    return {'focus': self.__focus.value, 'focusAuto': self.__auto_mode}
  
  def __str__(self):
    return f'FocusSettings({self.__dict__})'
  
  def __repr__(self):
    return f'FocusSettings({self.__dict__})'
  
  @property
  def auto(self) -> bool:
    return self.__auto_mode
  
  @property
  def value(self) -> FocusValue:
    return self.__focus.value
  
class WhiteBalanceSettings:
  '''Settings for the white balance of the camera.'''
  
  class WhiteBalanceColor:
    '''Color representation for the white balance of the camera. From 0 to 4095.'''
    def __init__(self, r : int = 2048, g : int = 2048, b : int = 2048):
      self.red = r
      self.green = g
      self.blue = b
      
    def __str__(self):
      return f'WhiteBalanceColor({self.__dict__})'
    
    def __repr__(self):
      return f'WhiteBalanceColor({self.__dict__})'
      
    @property
    def red(self) -> int:
      return self.__red
    
    @red.setter
    def red(self, r : int):
      if r < 0 or r > 4095:
        raise ValueError(f"WhiteBalanceColor must be in the range 0 to 4095 - got {r}")
      self.__red = r
      
    @property
    def green(self) -> int:
      return self.__green
    
    @green.setter
    def green(self, g : int):
      if g < 0 or g > 4095:
        raise ValueError(f"WhiteBalanceColor must be in the range 0 to 4095 - got {g}")
      self.__green = g
      
    @property
    def blue(self) -> int:
      return self.__blue
    
    @blue.setter
    def blue(self, b : int):
      if b < 0 or b > 4095:
        raise ValueError(f"WhiteBalanceColor must be in the range 0 to 4095 - got {b}")
      self.__blue = b
      
  def __init__(self, auto : bool = True, color : WhiteBalanceColor = WhiteBalanceColor()):
    self.__auto_mode : bool = auto
    self.__color : WhiteBalanceSettings.WhiteBalanceColor = color
    
  def setAutoMode(self):
    '''Set the white balance to auto.'''
    self.__auto_mode = True
    
  def setManualMode(self, color : WhiteBalanceColor):
    '''Set the white balance to manual.
       Color must be in the range 0 to 4095.'''
    self.__auto_mode = False
    self.__color = color
    
  def setManualMode(self, r : int, g : int, b : int):
    '''Set the white balance to manual.
       Colors must be in the range 0 to 4095.'''
    self.__auto_mode = False
    self.__color.red = r
    self.__color.green = g
    self.__color.blue = b
    
  def asPost(self) -> dict:
    '''Return the settings as a dict to be used in a POST request.'''
    return {'whiteBalanceAuto': self.__auto_mode, 'whiteBalanceBlue': self.__color.blue, 'whiteBalanceGreen': self.__color.green, 'whiteBalanceRed': self.__color.red}
  
  def __str__(self):
    return f'WhiteBalanceSettings({self.__dict__})'
  
  def __repr__(self):
    return f'WhiteBalanceSettings({self.__dict__})'
  
  @property
  def auto(self) -> bool:
    return self.__auto_mode
  
  @property
  def blue(self) -> int:
    return self.__color.blue
  
  @property
  def green(self) -> int:
    return self.__color.green
  
  @property
  def red(self) -> int:
    return self.__color.red

class CameraSettings:
  '''Settings for the camera.'''
  
  def __init__(self,
      whiteBalanceSettings : WhiteBalanceSettings = WhiteBalanceSettings(),
      focusSettings : FocusSettings = FocusSettings(),
      lightingSettings : LightingSettings = LightingSettings(),
      exposureSettings : ExposureSettings = ExposureSettings(),
    ):
    self.__brightness : int = 0
    self.__contrast : int = 0
    self.__contrastBrightnessAuto : bool = True
    self.__focusMax : int = 1023
    self.__focusMin : int = 150
    self.__focusSensitivity : int = 10
    self.__focusStep : int = 24
    self.__focusStrategy : int = 0
    self.__focusWindowEnabled : int = 3
    self.__focusWindowX1 : int = 192
    self.__focusWindowX2 : int = 448
    self.__focusWindowY1 : int = 144
    self.__focusWindowY2 : int = 336
    self.__gamma : int = 0
    self.__gammaAuto : bool = True
    self.__hue : int = 0
    self.__hueAuto : bool = True
    self.__pan : int = -1
    self.__saturation : int = 0
    self.__saturationAuto : bool = True
    self.__sharpness : int = 0
    self.__sharpnessAuto : bool = True
    self.__tilt : int = -1
    self.__zoom : int = -1
    self.__whiteBalanceTemp : int = 5
    self.whiteBalanceSettings : WhiteBalanceSettings = whiteBalanceSettings
    self.focusSettings : FocusSettings = focusSettings
    self.lightingSettings : LightingSettings = lightingSettings
    self.exposureSettings : ExposureSettings = exposureSettings

  def __str__(self) -> str:
    return f'CameraSettings({self.__dict__})'
  
  def __repr__(self) -> str:
    return f'CameraSettings({self.__dict__})'
  
  @staticmethod
  def createFromDict(data : dict):
    '''Create a CameraSettings object from a dict.
    '''
    whiteBalanceSettings = WhiteBalanceSettings(auto = bool(data['whiteBalanceAuto']), color = WhiteBalanceSettings.WhiteBalanceColor(r = int(data['whiteBalanceRed']), g = int(data['whiteBalanceGreen']), b = int(data['whiteBalanceBlue'])))
    focusSettings = FocusSettings(auto = bool(data['focusAuto']), focus = FocusSettings.FocusValue(value = int(data['focus'])))
    lightingSettings = LightingSettings(mode = LightingSettings.LightingMode(LightingSettings.LightingMode(int(data['lightingMode']))), value = LightingSettings.LightingValue(value = int(data['lightingValue'])))
    exposureSettings = ExposureSettings(sensitivity = ExposureSettings.ExposureSensitivity(sensitivity = int(data['exposureSensitivity'])), exposure_time_use = ExposureSettings.ExposureTimeUs(value = int(data['exposure'])), gain = ExposureSettings.Gain(value = int(data['gain'])), auto = bool(data['exposureAuto']))
    cs = CameraSettings(whiteBalanceSettings = whiteBalanceSettings, focusSettings = focusSettings, lightingSettings = lightingSettings, exposureSettings = exposureSettings)
    cs.__brightness = int(data['brightness'])
    cs.__contrast = int(data['contrast'])
    cs.__contrastBrightnessAuto = bool(data['contrastBrightnessAuto'])
    cs.__focusMax = int(data['focusMax'])
    cs.__focusMin = int(data['focusMin'])
    cs.__focusSensitivity = int(data['focusSensitivity'])
    cs.__focusStep = int(data['focusStep'])
    cs.__focusStrategy = int(data['focusStrategy'])
    cs.__focusWindowEnabled = int(data['focusWindowEnabled'])
    cs.__focusWindowX1 = int(data['focusWindowX1'])
    cs.__focusWindowX2 = int(data['focusWindowX2'])
    cs.__focusWindowY1 = int(data['focusWindowY1'])
    cs.__focusWindowY2 = int(data['focusWindowY2'])
    cs.__gamma = int(data['gamma'])
    cs.__gammaAuto = bool(data['gammaAuto'])
    cs.__hue = int(data['hue'])
    cs.__hueAuto = bool(data['hueAuto'])
    cs.__pan = int(data['pan'])
    cs.__saturation = int(data['saturation'])
    cs.__saturationAuto = bool(data['saturationAuto'])
    cs.__sharpness = int(data['sharpness'])
    cs.__sharpnessAuto = bool(data['sharpnessAuto'])
    cs.__tilt = int(data['tilt'])
    cs.__zoom = int(data['zoom'])
    cs.__whiteBalanceTemp = int(data['whiteBalanceTemp'])
    return cs
  
  def asPost(self) -> str:
    '''Return the settings as a dict to be used in a POST request.
       Values which should not be modified are reset to their default values.'''
    # Put the dict inside a dict key 'cameraSettings'
    vals = {
      'brightness': 0,
      'contrast': 0,
      'contrastBrightnessAuto': True,
      'focusMax': 1023,
      'focusMin': 150,
      'focusSensitivity': 10,
      'focusStep': 24,
      'focusStrategy': 0,
      'focusWindowEnabled': 3,
      'focusWindowX1': 192,
      'focusWindowX2': 448,
      'focusWindowY1': 144,
      'focusWindowY2': 336,
      'gamma': 0,
      'gammaAuto': True,
      'hue': 0,
      'hueAuto': True,
      'pan': -1,
      'saturation': 0,
      'saturationAuto': True,
      'sharpness': 0,
      'sharpnessAuto': True,
      'tilt': -1,
      'zoom': -1,
      'whiteBalanceAuto': self.whiteBalanceSettings.auto,
      'whiteBalanceBlue': self.whiteBalanceSettings.blue,
      'whiteBalanceGreen': self.whiteBalanceSettings.green,
      'whiteBalanceRed': self.whiteBalanceSettings.red,
      'whiteBalanceTemp': 5,
      'focusAuto': self.focusSettings.auto,
      'focus': self.focusSettings.value,
      'lightingMode': self.lightingSettings.mode,
      'lightingValue': self.lightingSettings.value,
      'exposure': self.exposureSettings.exposure_time_us,
      'exposureAuto': self.exposureSettings.auto,
      'exposureSensitivity': self.exposureSettings.sensitivity,
      'gain': self.exposureSettings.gain,
    }
    return json.dumps({'cameraSettings': vals}).replace(' ', '')
  
  @property
  def brightness(self) -> int:
    return self.__brightness
  
  @property
  def contrast(self) -> int:
    return self.__contrast
  
  @property
  def contrastBrightnessAuto(self) -> bool:
    return self.__contrastBrightnessAuto
  
  @property
  def focusMax(self) -> int:
    return self.__focusMax
  
  @property
  def focusMin(self) -> int:
    return self.__focusMin
  
  @property
  def focusSensitivity(self) -> int:
    return self.__focusSensitivity
  
  @property
  def focusStep(self) -> int:
    return self.__focusStep
  
  @property
  def focusStrategy(self) -> int:
    return self.__focusStrategy
  
  @property
  def focusWindowEnabled(self) -> int:
    return self.__focusWindowEnabled
  
  @property
  def focusWindowX1(self) -> int:
    return self.__focusWindowX1
  
  @property
  def focusWindowX2(self) -> int:
    return self.__focusWindowX2
  
  @property
  def focusWindowY1(self) -> int:
    return self.__focusWindowY1
  
  @property
  def focusWindowY2(self) -> int:
    return self.__focusWindowY2
  
  @property
  def gamma(self) -> int:
    return self.__gamma
  
  @property
  def gammaAuto(self) -> bool:
    return self.__gammaAuto
  
  @property
  def hue(self) -> int:
    return self.__hue
  
  @property
  def hueAuto(self) -> bool:
    return self.__hueAuto
  
  @property
  def pan(self) -> int:
    return self.__pan
  
  @property
  def saturation(self) -> int:
    return self.__saturation
  
  @property
  def saturationAuto(self) -> bool:
    return self.__saturationAuto
  
  @property
  def sharpness(self) -> int:
    return self.__sharpness
  
  @property
  def sharpnessAuto(self) -> bool:
    return self.__sharpnessAuto
  
  @property
  def tilt(self) -> int:
    return self.__tilt
  
  @property
  def zoom(self) -> int:
    return self.__zoom
  
  @property
  def whiteBalanceAuto(self) -> bool:
    return self.whiteBalanceSettings.auto
  
  @property
  def whiteBalanceBlue(self) -> int:
    return self.whiteBalanceSettings.blue
  
  @property
  def whiteBalanceGreen(self) -> int:
    return self.whiteBalanceSettings.green
  
  @property
  def whiteBalanceRed(self) -> int:
    return self.whiteBalanceSettings.red
  
  @property
  def whiteBalanceTemp(self) -> int:
    return self.__whiteBalanceTemp
  
  @property
  def focusAuto(self) -> bool:
    return self.focusSettings.auto
  
  @property
  def focus(self) -> FocusSettings.FocusValue:
    return self.focusSettings.value
  
  @property
  def lightingMode(self) -> int:
    return self.lightingSettings.mode
  
  @property
  def lightingValue(self) -> int:
    return self.lightingSettings.value
  
  @property
  def exposure(self) -> int:
    return self.exposureSettings.exposure_time_us
  
  @property
  def exposureAuto(self) -> bool:
    return self.exposureSettings.auto
  
  @property
  def exposureSensitivity(self) -> int:
    return self.exposureSettings.sensitivity
  
  @property
  def gain(self) -> int:
    return self.exposureSettings.gain
