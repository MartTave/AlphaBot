import json
from ssl import ALERT_DESCRIPTION_CERTIFICATE_REVOKED
import threading
import RPi.GPIO as GPIO
import time
from numpy._core.shape_base import block
import cv2
import os
import requests
from alphabot_agent.alphabotlib.TRSensors import TRSensor
import numpy as np
import asyncio
import aiohttp
from functools import reduce
import logging
import math
from enum import Enum
from alphabot_agent.alphabotlib.Pathfinding import Pathfinding

logger = logging.getLogger(__name__)



def flatten(li):
    return reduce(
        lambda x, y: [*x, y] if not isinstance(y, list) else x + flatten(y),
        li,
        [],
    )


class AlphaBot2(object):
    def __init__(self, ain1=12, ain2=13, ena=6, bin1=20, bin2=21, enb=26):

        botn = os.environ.get("XMPP_USERNAME")

        if botn is None:
            raise Exception("Could not get robot name through env variable")

        self.xmpp_username = botn
        self.target = None
        self.other_target = None

        self.botN = botn.split("-")[-1]

        self.otherN = "1" if self.botN == "2" else "2"

        self.configPath = "./alphabot_agent/config.json"

        self.loadConfig()

        self.GPIOSetup(ain1, ain2, bin1, bin2, ena, enb)

        self.updateFromConfig()

        self.labyrinth = None

        self.calibrationFilePath = "./alphabot_agent/calibs/"

        fileLow = np.load(self.calibrationFilePath + "Logitec_ceiling_854-480_0.npz")
        self.mtxLow = fileLow["mtx"]
        self.distLow = fileLow["dist"]

        fileHigh = np.load(self.calibrationFilePath + "Logitec_ceiling_1920-1080_0.npz")
        self.mtxHigh = fileHigh["mtx"]
        self.distHigh = fileHigh["dist"]

        self.api_url = "http://prosody:3000/api/messages"
        self.api_token = os.environ.get("API_TOKEN", "your_secret_token")
        self.session = None

        # Create HTTP session with keepalive
        timeout = aiohttp.ClientTimeout(total=None)  # No timeout
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=60, max=1000",
            },
        )

        self.TR = TRSensor()

    class BotState(Enum):
        IDLE = "idle"
        EXECUTING = "executing"

    @property
    def config(self):
        return self.config_file[self.botN]

    @property
    def otherConfig(self):
        return self.config_file[self.otherN]

    def loadConfig(self):
        with open(self.configPath) as file:
            self.config_file = json.load(file)

    def updateFromConfig(self):
        self.loadConfig()
        self.robot_aruco_id = self.config.get("arucoId", -1)
        self.target_aruco_id = self.config.get("targetArucoId", -1)

        self.other_aruco_id = self.otherConfig.get("arucoId", -1)
        self.other_target_aruco_id = self.otherConfig.get("targetArucoId", -1)

        self.forwardCorrection = self.config.get("forwardCorrection", 0)

        self.turn_speed = self.config.get("turnSpeed", 15)
        self.turn_braking_time = self.config.get("turnBrakingTime", 13)

        self.forward_speed = self.config.get("forwardSpeed", 30)
        self.forward_braking_time = self.config.get("forwardBrakingTime", 50)

        self.forwardEquation = lambda x: self.config.get("forwardA", 2.916) * x + self.config.get("forwardB", 130.984)

        self.turnEquation = lambda x:self.config.get("turnA", 4.792) * x + self.config.get("turnB", 118.63)

    def saveConfig(self):

        with open(self.configPath, "w") as file:
            file.write(json.dumps(self.config_file))

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
        self.config["turnSpeed"] = speed


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

        self.config["turnA"] = a
        self.config["turnB"] = b
        self.saveConfig()

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

        self.config["forwardSpeed"] = speed


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

        self.config["forwardA"] = a
        self.config["forwardB"] = b

        self.saveConfig()

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
                        self.config["forwardCorrection"] = self.forwardCorrection
                        self.saveConfig()
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

    def safeForward(self, mm=100, blocking=False):
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
        if blocking:
            thread.join()

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


    def cropImage(self, img, angle, x_pos, y_pos):
        """
        Apply a rotation and a crop on an image.

        Parameters:
            img: Input image
            angle: Rotation angle in degrees
            x_pos: x limits for the crop [min, max]
            y_pos: y limits for the crop [min, max]

        Returns:
            Rotated and cropped version of the input image
        """
        # Get image dimensions
        h, w = img.shape[:2]

        # Calculate the center of the image
        center = (w // 2, h // 2)

        # Create rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

        # Apply the affine transformation
        rotated_zoomed = cv2.warpAffine(img, rotation_matrix, (w, h))

        # returns the cropped rotated image
        x_min, x_max = x_pos
        y_min, y_max = y_pos
        end_img = rotated_zoomed[x_min:x_max, y_min:y_max]

        cv2.imwrite("maze.jpg", end_img)

        return(end_img)

        ##### test
        # parameters for the temporary camera
        # rotation = -3.6
        # x_pos = [152,725]
        # y_pos = [68,1824]
        # img = cv2.imread('./pic0.jpg')

        # using the function and showing result
        # img = cropImage(img=img, angle=rotation, x_pos=x_pos, y_pos=y_pos)
        # cv2.imshow('Zoomed and Rotated', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        #####

    def _getLines(self, img, rho, theta, threshold, min_line_length, max_line_gap, angle_thresh):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # -30 to 0
        image_lower_hsv = np.array([160, 91, 108])
        image_upper_hsv = np.array([179, 255, 255])
        imageMask1 = cv2.inRange(hsv, image_lower_hsv, image_upper_hsv)
        # 0 to 30
        image_lower_hsv = np.array([0, 148, 108])
        image_upper_hsv = np.array([20, 255, 255])
        imageMask2 = cv2.inRange(hsv, image_lower_hsv, image_upper_hsv)
        imageMask2 = cv2.inRange(hsv, image_lower_hsv, image_upper_hsv)

        # combine masks
        mask = cv2.bitwise_or(imageMask1, imageMask2)

        ## Slice the red
        red = cv2.bitwise_and(img, img, mask=mask)

        # Convert the masked red image to grayscale first
        red_gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY)

        # Threshold it to get cleaner lines
        _, red_binary = cv2.threshold(red_gray, 1, 255, cv2.THRESH_BINARY)

        # Hough transform on the purified edges
        lines = cv2.HoughLinesP(
            red_binary, rho, theta, threshold, np.array([]),
            min_line_length, max_line_gap
        )

        # Draw detected lines (in blue) on a blank canvas
        line_image = np.zeros_like(img)
        approved_lines = []

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = math.atan2(y2 - y1, x2 - x1) * 180 / np.pi

                if abs(angle) < angle_thresh or abs(angle) > (180 - angle_thresh) or (90 - angle_thresh) < abs(angle) < (90 + angle_thresh):
                    approved_lines.append(line[0])

        return approved_lines


    # checks if a line [x1,y1 to x2,y2] intersects with any line of forbidden_lines
    def _is_line_interrupted(self, forbidden_lines, x1, y1, x2, y2):
        # gets the linear equation [y = ax + b] that goes through p1 [x1,y1] and p2 [x2,y2] and outputs it as (a,b)
        def eq_from_p(p1, p2):
            p1x = p1[0]
            p1y = p1[1]
            p2x = p2[0]
            p2y = p2[1]
            if p2x != p1x:
                a = (p2y - p1y) / (p2x - p1x)
            else:
                a = 100000

            b = p2y - a * p2x

            return (a,b)


        # gets the intersection point between two linear equations [y = ax + b] and outputs it with a boolean telling if it exists [bool, x, y]
        def inter_from_eqs(eq1, eq2):
            a1, b1 = eq1
            a2, b2 = eq2
            if a1 == a2:
                return (b1 == b2, a1, b1)

            x = (b2 - b1) / (a1 - a2)
            y = a1 * x + b1
            return (True, x, y)


        # checks if two lines intersect
        def intersects(l1, l2):
            p1 = l1[:2]
            p2 = l1[2:]
            p3 = l2[:2]
            p4 = l2[2:]

            eq1 = eq_from_p(p1, p2)
            eq2 = eq_from_p(p3, p4)
            status, x, y = inter_from_eqs(eq1, eq2)
            x = round(x,0)
            y = round(y,0)
            if not status: return status

            bx = [p1[0], p2[0]]
            bx.sort()
            bx1, bx2 = bx

            bx = [p3[0], p4[0]]
            bx.sort()
            bx3, bx4 = bx

            by = [p1[1], p2[1]]
            by.sort()
            by1, by2 = by

            by = [p3[1], p4[1]]
            by.sort()
            by3, by4 = by

            return bx1 <= x <= bx2 and bx3 <= x <= bx4 and by1 <= y <= by2 and by3 <= y <= by4

        # ==============================================================================================================================
        # assures points format to avoid rounding problems
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        # checks if the given lines intersects with any of the labyrinth lines
        for i in forbidden_lines:
            if intersects(i, [x1,y1,x2,y2]):
                return True
        return False


    def _check_left(self, forbidden_lines, x, y, sec_w, sec_h):
        n_inter = 0
        n_inter += self._is_line_interrupted(forbidden_lines, x, y, x-sec_w, y)
        n_inter += self._is_line_interrupted(forbidden_lines, x, y-0.25*sec_h, x-sec_w, y-0.25*sec_h)
        n_inter += self._is_line_interrupted(forbidden_lines, x, y+0.25*sec_h, x-sec_w, y+0.25*sec_h)
        return n_inter >= 2

    def _check_right(self, forbidden_lines, x, y, sec_w, sec_h):
        n_inter = 0
        n_inter += self._is_line_interrupted(forbidden_lines, x, y, x+sec_w, y)
        n_inter += self._is_line_interrupted(forbidden_lines, x, y-0.25*sec_h, x+sec_w, y-0.25*sec_h)
        n_inter += self._is_line_interrupted(forbidden_lines, x, y+0.25*sec_h, x+sec_w, y+0.25*sec_h)
        return n_inter >= 2

    def _check_bottom(self, forbidden_lines, x, y, sec_w, sec_h):
        n_inter = 0
        n_inter += self._is_line_interrupted(forbidden_lines, x, y, x, y+sec_h)
        n_inter += self._is_line_interrupted(forbidden_lines, x-0.25*sec_w, y, x-0.25*sec_w, y+sec_h)
        n_inter += self._is_line_interrupted(forbidden_lines, x+0.25*sec_w, y, x+0.25*sec_w, y+sec_h)
        return n_inter >= 2

    def _check_top(self, forbidden_lines, x, y, sec_w, sec_h):
        n_inter = 0
        n_inter += self._is_line_interrupted(forbidden_lines, x, y, x, y-sec_h)
        n_inter += self._is_line_interrupted(forbidden_lines, x-0.25*sec_w, y, x-0.25*sec_w, y-sec_h)
        n_inter += self._is_line_interrupted(forbidden_lines, x+0.25*sec_w, y, x+0.25*sec_w, y-sec_h)
        return n_inter >= 2


    def processImage(self, img, quality):
        rotation = -1
        factor = 1
        if quality == "full quality":
            factor = 1
        elif quality == "low quality":
            factor = 2.25

        x_pos = [int(305 / factor),int(868 / factor)]
        y_pos = [int(37 / factor),int(1838 / factor)]

        grid_top = int(40 / factor)
        grid_down = int(480 / factor)
        grid_left = int(112 / factor)
        grid_right = int(1700 / factor)

        grid_width = 11
        grid_height = 3


        cell_width = (grid_right - grid_left) / grid_width
        cell_height = (grid_down - grid_top) / grid_height

        # Crop and rotate the image
        cropped = self.cropImage(img, rotation, x_pos, y_pos)

        # Draw circles on the cropped image at grid_top, grid_left, grid_right, and grid_down
        cv2.circle(cropped, (grid_left, grid_top), 10, (0, 255, 0), -1)  # Top-left
        cv2.circle(cropped, (grid_right, grid_top), 10, (0, 255, 0), -1)  # Top-right
        cv2.circle(cropped, (grid_left, grid_down), 10, (0, 255, 0), -1)  # Bottom-left
        cv2.circle(cropped, (grid_right, grid_down), 10, (0, 255, 0), -1)  # Bottom-right

        cv2.imwrite("./alphabot_agent/frame_aha.png", cropped)

        def posToGrid(pos):
            grid_x = int((pos[0]-grid_left)/cell_width)
            grid_y = int((pos[1]-grid_top)/cell_height)
            logger.info(f"Pos is : grid_x : {grid_x}, grid_y : {grid_y} grid_width : {grid_width} grid_height : {grid_height}")
            n = grid_x + grid_width * grid_y
            return n

        def find_labyrinth(img):
            if self.labyrinth is None:
                section_width = (grid_right - grid_left) / grid_width
                section_height = (grid_down - grid_top) / grid_height
                lines = self._getLines(img, 1, np.pi / 360, 50, 80, 10, 10)

                section_tab = []
                for i in range(grid_width):
                    for j in range(grid_height):
                        x = int(grid_left + (i + 0.5) * section_width)
                        y = int(grid_top + (j + 0.5) * section_height)

                        l = 'l' if self._check_left(lines, x, y, section_width, section_height) or i == 0 else ''
                        r = 'r' if self._check_right(lines, x, y, section_width, section_height) or i == grid_width-1 else ''
                        b = 'b' if self._check_bottom(lines, x, y, section_width, section_height) or j == grid_height-1 else ''
                        t = 't' if self._check_top(lines, x, y, section_width, section_height) or j == 0 else ''
                        section = l + r + b + t
                        section_tab.append(section)
                logger.info(section_tab)
                self.labyrinth = np.reshape(section_tab, (grid_width, grid_height)).T
            return self.labyrinth

        def where_aruco(img, aruco_id):

            # detection of all arucos
            dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            detectorParams = cv2.aruco.DetectorParameters()
            detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
            marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(img)

            if marker_ids is not None and len(marker_ids) == 0:
                logger.warning("No arcuo marker found in image")
                return (-1, -1, -1)

            # looking for asked aruco
            pos = np.where(marker_ids.flatten() == [aruco_id])

            # if no asked aruco
            if len(pos[0]) == 0:
                logger.error(f"Didn't find the aruco {aruco_id} in the picture.")
                return (-1,-1,-1)

            # else
            # boar method to get the center (averaging the x,y coordinates of the corners)
            moy_x = 0
            moy_y = 0
            for i in marker_corners[pos[0][0]][0]:
                moy_x += i[0]
                moy_y += i[1]

            moy_x /= 4
            moy_y /= 4

            cell = posToGrid([moy_x, moy_y])
            # placing the points
            marker_length = 1.0
            obj_points = np.array([
                [-marker_length/2,  marker_length/2, 0],  # Top-left
                [ marker_length/2,  marker_length/2, 0],  # Top-right
                [ marker_length/2, -marker_length/2, 0],  # Bottom-right
                [-marker_length/2, -marker_length/2, 0]   # Bottom-left
            ], dtype=np.float32)


            camera_matrix = self.mtxHigh if quality == "full quality" else self.mtxLow
            dist_coeffs = self.distHigh if quality == "full quality" else self.distLow

            # Solve for pose
            success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners[pos[0][0]][0], camera_matrix, dist_coeffs)

            # Convert rotation vector to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Extract Euler angles (ZYX convention)
            sy = np.sqrt(rotation_matrix[0,0] * rotation_matrix[0,0] + rotation_matrix[1,0] * rotation_matrix[1,0])
            singular = sy < 1e-6

            if not singular:
                yaw = np.arctan2(rotation_matrix[1,0], rotation_matrix[0,0])
            else:
                yaw = np.arctan2(-rotation_matrix[0,1], rotation_matrix[1,1])

            yaw_deg = np.degrees(yaw)


            return cell, yaw_deg

        def detect_targets(img):
            if self.target is None:
                self.target = where_aruco(img, self.target_aruco_id)[0]
                self.other_target = where_aruco(img, self.other_target_aruco_id)[0]

        def detect_positions(img):
            logger.error(f"self.robot_aruco_id {self.robot_aruco_id}")
            logger.error(f"self.other_aruco_id {self.other_aruco_id}")

            robot = where_aruco(img, self.robot_aruco_id)
            other_robot = where_aruco(img, self.other_aruco_id)
            return robot, other_robot


        detect_targets(cropped)
        labyrinth = find_labyrinth(cropped)
        robot, other_robot = detect_positions(cropped)

        logger.info(f"Start is : {robot[0]} and other is : {other_robot[0]}")

        return self.runMaze(robot, self.target, other_robot, self.other_target, (grid_left, grid_top), (grid_right, grid_down))



    def posToSubGrid(self, pos,grid_top,grid_left,section_width,section_height):
        section_width = int(section_width/2)
        section_height = int(section_height/2)
        grid_x = int((pos[0]-grid_left)/section_width)
        grid_y = int((pos[1]-grid_top)/section_height)
        return grid_x, grid_y


    def runMaze(self, robot, target, other_robot, other_target, top_left, bottom_right):
        pathfinder = Pathfinding()
        if self.labyrinth is None:
            logger.error("Can't run maze wihtout loading it first !")
            return
        path_robo1 = pathfinder.get_path_from_maze(self.labyrinth, robot[0], target)
        path_robo2 = pathfinder.get_path_from_maze(self.labyrinth, other_robot[0], other_target)

        logger.error(f"other robot: {other_robot}")

        pathfinder.draw_maze(self.labyrinth, path_robo1, path_robo2,  "./alphabot_agent/without_col.png")
        curr_path, other_path = pathfinder.avoid_collision(path_robo1, path_robo2)


        json_commands = pathfinder.get_json_from_path(curr_path, robot[1])

        b64_image = pathfinder.draw_on_pic(curr_path, other_path, top_left, bottom_right)

        requests.post(
            "http://prosody:3000/api/maze/plan",
            json = {
                "image": b64_image
            },
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Connection": "keep-alive",
            },
            timeout=None,
        )
        arrived = False
        toExecute = []
        if len(json_commands["commands"]) == 2:
            logger.warning("Reducing the number of command, removing the last to stop before target")
            toExecute = json_commands["commands"][:-1]
            arrived = True
        else:
            toExecute = json_commands["commands"][:2]

        for i in toExecute:
            # logger.info(i["command"])
            current_command = i["command"]
            # notify state change
            # when executing command
            self.notify_state_change(self.BotState.EXECUTING, current_command)

            if current_command == "rotate":
                rotation = int(float(i["args"][0]))
                self.turn(rotation)
            elif current_command == "forward":
                frwrd = int(float(i["args"][0]))
                self.safeForward(200 * frwrd, blocking=True)

            # notify state change
            # when going back to idle
            self.notify_state_change(self.BotState.IDLE, "")
        return arrived

    def notify_state_change(self, state, label):
        if self.session:
            try:
                state_update = {
                    "agent_jid": self.xmpp_username,
                    "type": "state_update",
                    "state": state.value,  # Use the string value of the enum
                    "label": label,
                    "timestamp": int(time.time()),
                }

                try:
                    response = requests.post(
                        self.api_url,
                        json=state_update,
                        headers={
                            "Authorization": f"Bearer {self.api_token}",
                            "Connection": "keep-alive",
                        },
                        timeout=None,  # No timeout
                    )

                    if response.status_code == 200:
                        logger.info(f"State update sent: {state.value}")
                    else:
                        logger.error(f"Failed to send state update. Status: {response.status_code}")
                except Exception as e:
                    logger.error(f"Failed to send request: {e}")

            except Exception as e:
                logger.error(f"Failed to send state update: {e}")

if __name__ == "__main__":
    Ab = AlphaBot2()
    Ab.forward()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()
