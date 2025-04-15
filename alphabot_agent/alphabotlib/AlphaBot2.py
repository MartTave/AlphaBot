import threading
import RPi.GPIO as GPIO
import time
import cv2
from alphabot_agent.alphabotlib.TRSensors import TRSensor
import numpy as np
from functools import reduce
import logging


logger = logging.getLogger(__name__)


def flatten(li):
    return reduce(
        lambda x, y: [*x, y] if not isinstance(y, list) else x + flatten(y),
        li,
        [],
    )


class AlphaBot2(object):
    def __init__(self, ain1=12, ain2=13, ena=6, bin1=20, bin2=21, enb=26):
        self.GPIOSetup(ain1, ain2, bin1, bin2, ena, enb)

        self.forwardCorrection = -2

        self.turn_speed = 15
        self.turn_braking_time = 13

        self.forward_speed = 30
        self.forward_braking_time = 50

        self.forwardEquation = lambda x: 2.916192 * x + 130.98477
        self.turnEquation = lambda x: 4.792480 * x + 118.629884

        self.TR = TRSensor()

    def GPIOSetup(self, ain1, ain2, bin1, bin2, ena, enb):
        self.AIN1 = ain1
        self.AIN2 = ain2
        self.BIN1 = bin1
        self.BIN2 = bin2
        self.ENA = ena
        self.ENB = enb
        self.PA = 50
        self.PB = 50
        self.PIC_WIDTH = 640
        self.PIC_HEIGHT = 480
        self.CTR = 7
        self.LEFT = 10
        self.RIGHT = 9
        self.DOWN = 11
        self.UP = 8
        self.BUZ = 4
        self.DR = 16
        self.DL = 19

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.setup(self.BUZ, GPIO.OUT)
        GPIO.setup(self.CTR, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.LEFT, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.RIGHT, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.UP, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.DOWN, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.AIN1, GPIO.OUT)
        GPIO.setup(self.AIN2, GPIO.OUT)
        GPIO.setup(self.BIN1, GPIO.OUT)
        GPIO.setup(self.BIN2, GPIO.OUT)
        GPIO.setup(self.ENA, GPIO.OUT)
        GPIO.setup(self.ENB, GPIO.OUT)
        GPIO.setup(self.DR, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.DL, GPIO.IN, GPIO.PUD_UP)
        self.PWMA = GPIO.PWM(self.ENA, 500)
        self.PWMB = GPIO.PWM(self.ENB, 500)
        self.PWMA.start(self.PA)
        self.PWMB.start(self.PB)
        self.stop()

    def takePic(self):
        # Open camera
        cap = cv2.VideoCapture(0)

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.PIC_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.PIC_HEIGHT)

        # Get actual width/height returned
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Actual resolution: {actual_width}x{actual_height}")

        time.sleep(0.5)

        ret, frame = cap.read()
        if ret:
            return frame.reshape((self.PIC_HEIGHT, self.PIC_WIDTH, 3))
        raise Exception("Failed to capture image")

    def beep_on(self):
        GPIO.output(self.BUZ, GPIO.HIGH)

    def beep_off(self):
        GPIO.output(self.BUZ, GPIO.LOW)

    def calibrateTRSensors(self):
        self.left(15)
        time.sleep(0.5)
        self.TR.calibrate()
        self.stop()
        print("Calibrated, values:")
        print("Min: ", self.TR.calibratedMin)
        print("Max: ", self.TR.calibratedMax)

    def fullCalibration(self, turn_speed=15, forward_speed=30):
        self.forward_speed = forward_speed
        self.turn_speed = turn_speed
        logger.info(
            "Press joystick center to start calibration. We will start by the sensors"
        )
        self.waitForJoystickCenter()
        self.calibrateTRSensors()

        logger.info("Sensor calibration done ! Doing correction calibration")
        self.calibrateForwardCorrection()
        logger.info("Sensors calibration done ! Doing forward calibration...")
        self.calibrateForward(forward_speed)
        logger.info("Forward calibration done ! Doing turn calibration...")
        self.calibrateTurn(turn_speed)
        logger.info("Turn calibration done ! Doing correction calibration...")
        logger.info("Calibration done !")

    def calibrateTurn(self, speed=15):
        lineTreshold = 150
        whiteTreshold = 850

        self.turn_speed = speed

        angles = [90, 180, 270, 360, 450, 540]
        measurements = []

        preciseSpeed = 11

        def runUntilLine(numerOfLine=1, timeout=5000):
            armed = False
            counter = 0
            lineCounter = 0
            while True:
                if counter >= timeout:
                    self.stop()
                    break
                counter += 1
                res = self.TR.readCalibrated()
                if res[2] < lineTreshold and (
                    res[1] > whiteTreshold and res[3] > whiteTreshold
                ):
                    if armed:
                        lineCounter += 1
                        armed = False
                if not armed and res[2] > whiteTreshold:
                    armed = True
                if lineCounter >= numerOfLine:
                    break

        def measureTimeToNextLine(numberOfLine=1):
            self.left(speed)
            start = time.time()
            runUntilLine(numberOfLine)
            stop = time.time()
            self.stop()
            return stop - start

        def turnToLine():
            self.left(preciseSpeed)
            runUntilLine()
            self.stop()

        def turnBackToLine():
            self.right(preciseSpeed)
            runUntilLine()
            self.stop()

        print("Waiting for joystick press to start the turn calibration")
        self.waitForJoystickCenter()
        time.sleep(0.5)

        turnToLine()

        time.sleep(1)

        for a in angles:
            nbrOfLine = a // 90
            measurements.append(measureTimeToNextLine(nbrOfLine) * 1000)

            time.sleep(0.5)
            turnBackToLine()
            if a != angles[-1]:
                print("Waiting for joystick for next turn")
                self.waitForJoystickCenter()
                time.sleep(0.5)

        a, b = np.polyfit(angles, measurements, 1)

        logger.debug("A is : " + str(a))
        logger.debug("B is : " + str(b))

        error = 0
        for x, y in zip(angles, measurements):
            error += abs(a * x + b - y)
        error /= len(angles)
        logger.info("Error is : " + str(error))

        self.turnEquation = lambda x: a * x + b

    def waitForJoystickCenter(self):
        while True:
            if GPIO.input(self.CTR) == 0:
                self.beep_on()
                while GPIO.input(self.CTR) == 0:
                    time.sleep(0.05)
                self.beep_off()
                break

    def calibrateForward(self, speed=30):
        lineTreshold = 100
        whiteTreshold = 900

        self.forward_speed = speed

        papers = [[30, 40, 70, 120], [100, 150]]

        measurements = []

        preciseSpeed = 9

        def runUntilLine(timeout=1500):
            armed = False
            counter = 0
            while True:
                if counter >= timeout:
                    self.stop()
                    logger.warning(
                        "Line not detected until timeout ! Calibration failed..."
                    )
                counter += 1
                res = self.TR.readCalibrated()
                if res[2] < lineTreshold and (
                    res[1] < lineTreshold or res[3] < lineTreshold
                ):
                    if armed:
                        break
                if res[2] > whiteTreshold and (
                    res[1] > whiteTreshold or res[3] > whiteTreshold
                ):
                    armed = True
            pass

        def measureTimeToNextLine():
            self.PA = speed
            self.PB = speed + self.forwardCorrection
            self.forward()
            start = time.time()
            runUntilLine()
            stop = time.time()
            self.stop()
            return stop - start

        def goToStartLine():
            self.PA = preciseSpeed
            self.PB = preciseSpeed
            self.forward()
            runUntilLine()
            self.stop()

        def goBackToLine():
            self.PA = preciseSpeed
            self.PB = preciseSpeed
            self.backward()
            runUntilLine()
            self.stop()

        for p in papers:
            logger.info("Waiting for joystick press to start the forward calibration")
            self.waitForJoystickCenter()
            time.sleep(0.5)
            goToStartLine()
            logger.debug("At start line. starting !")
            time.sleep(0.5)
            for d in p:
                timeTaken = measureTimeToNextLine()
                logger.debug("Measurement taken !")
                measurements.append(timeTaken * 1000)
                time.sleep(0.5)
                goBackToLine()
                time.sleep(2)
        flattened = flatten(papers)

        a, b = np.polyfit(flattened, measurements, 1)

        logger.debug("A is : " + str(a))
        logger.debug("B is : " + str(b))

        error = 0
        for x, y in zip(flattened, measurements):
            error += abs(a * x + b - y)
        error /= len(flattened)
        logger.info("Error is : " + str(error))

        self.forwardEquation = lambda x: a * x + b

    def calibrateForwardCorrection(self):
        logger.info(
            "Press joystick center to go forward, left or right to correct. Down to stop"
        )
        while True:
            for n in [self.CTR, self.RIGHT, self.LEFT, self.DOWN]:
                if GPIO.input(n) == 0:
                    self.beep_on()
                    while GPIO.input(n) == 0:
                        time.sleep(0.01)
                    self.beep_off()
                    if n == self.DOWN:
                        return
                    if n == self.RIGHT:
                        self.forwardCorrection -= 0.2
                        pass
                    if n == self.LEFT:
                        self.forwardCorrection += 0.2
                        pass
                    if n == self.CTR:
                        self.PA = self.forward_speed
                        self.PB: float = self.forward_speed + self.forwardCorrection
                        self.forward()
                        time.sleep(1)
                        self.stop()

    def turn(self, angle=90):
        self.PWMA.ChangeDutyCycle(self.turn_speed)
        self.PWMB.ChangeDutyCycle(self.turn_speed)

        if self.turnEquation:
            duration = self.turnEquation(abs(angle) - self.turn_braking_time) / 1000
        else:
            duration = 0.1 * abs(angle)
            logger.warning(
                "No calibration found for turning ! Angle will be approximate"
            )

        if angle > 0:
            GPIO.output(self.AIN1, GPIO.LOW)
            GPIO.output(self.AIN2, GPIO.HIGH)
            GPIO.output(self.BIN1, GPIO.HIGH)
            GPIO.output(self.BIN2, GPIO.LOW)
        else:
            GPIO.output(self.AIN1, GPIO.HIGH)
            GPIO.output(self.AIN2, GPIO.LOW)
            GPIO.output(self.BIN1, GPIO.LOW)
            GPIO.output(self.BIN2, GPIO.HIGH)

        time.sleep(duration)

        self.stop()

        pass

    def safeForward(self, mm=100):
        if self.forwardEquation:
            duration = self.forwardEquation(mm - self.forward_braking_time) / 1000
        else:
            duration = self.forward_speed * mm * 150
            duration += self.motor_startup_forward
            logger.warning(
                "No forward calibration done ! Duration will be aproximative at best !"
            )

        self.PWMA.ChangeDutyCycle(self.forward_speed)
        self.PWMB.ChangeDutyCycle(self.forward_speed + self.forwardCorrection)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.HIGH)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.HIGH)


        def run_for_time(duration):
            start_time = time.time()
            while time.time() - start_time < duration:
                DR_status = GPIO.input(self.DR)
                DL_status = GPIO.input(self.DL)
                if (DL_status == 0) or (DR_status == 0):
                    break
            self.stop()

        thread = threading.Thread(target=run_for_time, args=(duration,))
        thread.start()

    def forward(self):
        self.PWMA.ChangeDutyCycle(self.PA)
        self.PWMB.ChangeDutyCycle(self.PB)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.HIGH)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.HIGH)

    def stop(self):
        self.PWMA.ChangeDutyCycle(0)
        self.PWMB.ChangeDutyCycle(0)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.LOW)

    def backward(self):
        self.PWMA.ChangeDutyCycle(self.PA)
        self.PWMB.ChangeDutyCycle(self.PB)
        GPIO.output(self.AIN1, GPIO.HIGH)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.HIGH)
        GPIO.output(self.BIN2, GPIO.LOW)

    def left(self, speed=30):
        self.PWMA.ChangeDutyCycle(speed)
        self.PWMB.ChangeDutyCycle(speed)
        GPIO.output(self.AIN1, GPIO.HIGH)
        GPIO.output(self.AIN2, GPIO.LOW)
        GPIO.output(self.BIN1, GPIO.LOW)
        GPIO.output(self.BIN2, GPIO.HIGH)

    def right(self, speed=30):
        self.PWMA.ChangeDutyCycle(speed)
        self.PWMB.ChangeDutyCycle(speed)
        GPIO.output(self.AIN1, GPIO.LOW)
        GPIO.output(self.AIN2, GPIO.HIGH)
        GPIO.output(self.BIN1, GPIO.HIGH)
        GPIO.output(self.BIN2, GPIO.LOW)

    def setPWMA(self, value):
        self.PA = value
        self.PWMA.ChangeDutyCycle(self.PA)

    def setPWMB(self, value):
        self.PB = value
        self.PWMB.ChangeDutyCycle(self.PB)

    def setMotor(self, left, right):
        if (right >= 0) and (right <= 100):
            GPIO.output(self.AIN1, GPIO.HIGH)
            GPIO.output(self.AIN2, GPIO.LOW)
            self.PWMA.ChangeDutyCycle(right)
        elif (right < 0) and (right >= -100):
            GPIO.output(self.AIN1, GPIO.LOW)
            GPIO.output(self.AIN2, GPIO.HIGH)
            self.PWMA.ChangeDutyCycle(0 - right)
        if (left >= 0) and (left <= 100):
            GPIO.output(self.BIN1, GPIO.HIGH)
            GPIO.output(self.BIN2, GPIO.LOW)
            self.PWMB.ChangeDutyCycle(left)
        elif (left < 0) and (left >= -100):
            GPIO.output(self.BIN1, GPIO.LOW)
            GPIO.output(self.BIN2, GPIO.HIGH)
            self.PWMB.ChangeDutyCycle(0 - left)

if __name__ == "__main__":
    Ab = AlphaBot2()
    Ab.forward()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
