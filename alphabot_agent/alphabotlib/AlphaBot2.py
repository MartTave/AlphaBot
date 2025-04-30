import threading
import RPi.GPIO as GPIO
import time
import cv2
import Image
from alphabot_agent.alphabotlib.TRSensors import TRSensor
import numpy as np
from functools import reduce
import logging
from Pathfinding import Pathfinding
import math

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
        return(rotated_zoomed[x_min:x_max, y_min:y_max])

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

        lower = np.array([93, 68, 99])
        upper = np.array([179, 255,255])
        mask = cv2.inRange(hsv, lower, upper)

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


    def find_labyrinth(self, img, grid_top, grid_down, grid_left, grid_right, grid_width, grid_height):
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

        return np.reshape(section_tab, (grid_width, grid_height)).T


    
    def where_arucos(self, img, aru_id):
        # detection of all arucos
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        detectorParams = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, detectorParams)
        marker_corners, marker_ids, rejected_candidates = detector.detectMarkers(img)

        # looking for asked aruco
        pos = np.where(marker_ids.flatten() == [aru_id])
        
        # if no asked aruco
        if len(pos[0]) == 0:
            print(f"Didn't find the aruco {aru_id} in the picture.")
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

        # placing the points
        marker_length = 1.0
        obj_points = np.array([
            [-marker_length/2,  marker_length/2, 0],  # Top-left
            [ marker_length/2,  marker_length/2, 0],  # Top-right
            [ marker_length/2, -marker_length/2, 0],  # Bottom-right
            [-marker_length/2, -marker_length/2, 0]   # Bottom-left
        ], dtype=np.float32)


        # Camera matrix and distortion coefficients (a regler une fois la camera calibrée)
        camera_matrix = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 1]]) 
        dist_coeffs = np.zeros((4, 1)) 

        # Solve for pose
        success, rvec, tvec = cv2.solvePnP(obj_points, marker_corners[0][0], camera_matrix, dist_coeffs)

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


        return moy_x,moy_y,yaw_deg


    def runMaze(self, maze, start_r1, stop_r1, angle_r1 = 0, start_r2 = 0, stop_r2 = 4, angle_r2 = 0, bot = 1):
        pathfinder = Pathfinding()
        path_robo1 = pathfinder.get_path_from_maze(maze, start_r1, stop_r1)
        path_robo2 = pathfinder.get_path_from_maze(maze, start_r2, stop_r2)

        print("Robots crossing at:")
        pathfinder.problem_detect(path_robo1, path_robo2)
        print("to be handled later")

        with Image.open("cropped.jpg") as im:
            pathfinder.draw_on_pic(im, path_robo1, path_robo2)

        json_commands = {}
        if bot == 1:
            json_commands = pathfinder.get_json_from_maze(maze, start_r1, stop_r1, False, angle_r1)
        else:
            json_commands = pathfinder.get_json_from_maze(maze, start_r2, stop_r2, False, angle_r2)

        for idx, i in enumerate(json_commands["commands"]):
            if i["command"] == "rotate":
                rotation = float(i["args"][0])
                self.turn(rotation)
            elif i["command"] == "forward":
                frwrd = float(i["args"][0])
                self.safeForward(200 * frwrd)
            
            if idx != 0:
                if i["command"] == "rotate":
                    break
        self.stop()

    def runBot(self, img):
        rotation = -3.6
        x_pos = [152,725]
        y_pos = [68,1824]
        img = self.cropImage(img, rotation, x_pos, y_pos)
        cv2.imwrite("cropped.jpg", img)

        # grid parameters for temporary camera
        grid_top = 45
        grid_down = 475
        grid_left = 65
        grid_right = 1680
        grid_width = 11
        grid_height = 3
        
        # finding the labyrinth
        tab = self.find_labyrinth(img,grid_top,grid_down,grid_left,grid_right,grid_width,grid_height)




if __name__ == "__main__":
    Ab = AlphaBot2()
    Ab.forward()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        GPIO.cleanup()