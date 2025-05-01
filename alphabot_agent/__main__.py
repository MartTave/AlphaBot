import asyncio
import logging
import os
import time
import RPi.GPIO as GPIO
import base64
import numpy as np
import cv2
from enum import Enum

import aiohttp
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message

from alphabot_agent.alphabotlib.AlphaBot2 import AlphaBot2

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AlphaBotAgent")

last_photo = None

# Enable SPADE and XMPP specific logging
for log_name in ["spade", "aioxmpp", "xmpp"]:
    log = logging.getLogger(log_name)
    log.setLevel(logging.INFO)
    log.propagate = True


class BotState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"


class AlphaBotAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_url = "http://prosody:3000/api/messages"
        self.api_token = os.environ.get("API_TOKEN", "your_secret_token")
        self.session = None
        self._state = None
        self.robot = AlphaBot2()

    @property
    def state(self):
        return self._state

    async def set_state(self, new_state, command):
        """Async method to set state and notify about the change."""
        if self._state != new_state:  # Only update if state actually changes
            self._state = new_state
            if self.session:
                print(
                    f"Calling notify_state_change for new state: {new_state} with command {command}"
                )
                await self.notify_state_change(command)

    async def notify_state_change(self, label):
        try:
            state_update = {
                "agent_jid": self.jid[0],
                "type": "state_update",
                "state": self.state.value,
                "label": label,
                "timestamp": int(asyncio.get_event_loop().time()),
            }
            print(state_update)

            # Use keepalive connection
            async with self.session.post(
                self.api_url,
                json=state_update,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Connection": "keep-alive",  # Add keepalive header
                },
                timeout=aiohttp.ClientTimeout(total=None),  # No timeout
            ) as response:
                if response.status == 200:
                    logger.info(f"State update sent: {self.state.value}")
                else:
                    logger.error(
                        f"Failed to send state update. Status: {response.status}"
                    )

        except Exception as e:
            logger.error(f"Failed to send state update: {e}")

    async def setup(self):
        # Create HTTP session with keepalive
        timeout = aiohttp.ClientTimeout(total=None)  # No timeout
        self.session = aiohttp.ClientSession(
            timeout=timeout,
            headers={
                "Connection": "keep-alive",
                "Keep-Alive": "timeout=60, max=1000",
            },
        )

        # Add a periodic heartbeat behavior
        heartbeat_behavior = self.HeartbeatBehavior()
        self.add_behaviour(heartbeat_behavior)

        # Add command listener behavior
        # command_behavior = self.XMPPCommandListener()
        # self.add_behaviour(command_behavior)

        waitForStartBehavior = self.waitForStartBehavior()
        self.add_behaviour(waitForStartBehavior)

        # Set initial state after setup
        await self.set_state(BotState.IDLE, "")

    # Add a new heartbeat behavior
    class HeartbeatBehavior(CyclicBehaviour):
        async def run(self):
            if self.agent.state:  # Only send heartbeat if we have a state
                await self.agent.notify_state_change(
                    ""
                )  # Send current state as heartbeat
            await asyncio.sleep(30)

    class waitForStartBehavior(OneShotBehaviour):
        async def run(self):

            def waitForUpJoystick():
                UP = 8
                BUZ = 4
                GPIO.setmode(GPIO.BCM)
                GPIO.setwarnings(False)
                GPIO.setup(UP, GPIO.IN, GPIO.PUD_UP)
                GPIO.setup(BUZ, GPIO.OUT)


                def beep_on():
                    GPIO.output(BUZ, GPIO.HIGH)
                    pass

                def beep_off():
                    GPIO.output(BUZ, GPIO.LOW)
                    pass

                while True:
                    time.sleep(0.05)
                    if GPIO.input(UP) == 0:
                        beep_on()
                        while GPIO.input(UP) == 0:
                            time.sleep(0.05)
                        beep_off()
                        break

            # waitForUpJoystick()
            self.agent.add_behaviour(self.agent.AskPhotoBehaviour())
            # We can start the maze resolution !

    class ProcessImageBehaviour(OneShotBehaviour):
        def __init__(self, img):
            super().__init__()
            self.img = img

        async def run(self):
            rotation = -3.6
            x_pos = [int(152 / 2.25),int(725 / 2.25)]
            y_pos = [int(68 / 2.25),int(1824 / 2.25)]

            grid_top = int(45 / 2.25)
            grid_down = int(475 / 2.25)
            grid_left = int(65 / 2.25)
            grid_right = int(1680 / 2.25)


            grid_width = 11
            grid_height = 3

            # Crop and rotate the image
            cropped = self.agent.robot.cropImage(self.img, rotation, x_pos, y_pos)

            cv2.circle(cropped, (grid_left, grid_top), 3, [255, 0, 0])
            cv2.circle(cropped, (grid_right, grid_down), 3, [255, 0, 0])


            cv2.imwrite("./alphabot_agent/frame.png", cropped)

            # This will update the labyrinth var inside the robot class
            self.agent.robot.find_labyrinth(cropped, grid_top, grid_down, grid_left, grid_right, grid_width, grid_height)
            posx, posy, angle = self.agent.robot.where_arucos(cropped, 12)

            logger.info(f"Arcuo pos : {posx} : {posy}")

            cell_width = (grid_right - grid_left) / grid_width
            cell_height = (grid_down - grid_top) / grid_height

            grid_x, grid_y = self.agent.robot.posToGrid([posx, posy], grid_top, grid_left, cell_width, cell_height)

            n = grid_x + grid_width * grid_y

            logger.info(f"Alors, {grid_x} et {grid_width} et {grid_y}")

            logger.info("Start is : " + str(n))

            logger.info(f"Angle is : {angle}")

            self.agent.robot.runMaze(n, 3, angle)

            self.agent.add_behaviour(self.agent.AskPhotoBehaviour())



    class AskPhotoBehaviour(OneShotBehaviour):
        async def run(self):
            global last_photo
            msg = Message(to="camera_agent@prosody")
            msg.set_metadata("performative", "inform")
            msg.body = "Requesting top-view picture"
            if last_photo is not None and time.time() - last_photo < 500:
                delay = (500 - (time.time() - last_photo))
                logger.info("500ms not elapsed since last request, sleeping for : " + str(delay))
                time.sleep(delay / 1000)
            await self.send(msg)
            msg = await self.receive(timeout=5000)
            logger.info("Got picture !!!")
            last_photo = time.time()
            logger.info(msg)
            if msg:
                img_data = base64.b64decode(msg.body)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.agent.add_behaviour(self.agent.ProcessImageBehaviour(img))
            else:
                logger.error("Could not get photo in return of camera_agent")


    class XMPPCommandListener(CyclicBehaviour):

        async def run(self):
            msg = await self.receive(timeout=100)
            if msg:
                logger.info(
                    f"[Behavior] Received command ({msg.sender}): {msg.body}"
                )

                command = msg.body
                if not command.startswith("command:"):
                    return
                command = command.replace("command:", "")
                # Set state to EXECUTING and notify immediately
                await self.agent.set_state(BotState.EXECUTING, command)

                # Process the command
                await self.process_command(command)

                # Set state back to IDLE after processing
                await self.agent.set_state(BotState.IDLE, "")

                # Send a confirmation response
                reply = Message(to=str(msg.sender))
                reply.set_metadata("performative", "inform")
                reply.body = f"Executed command: {msg.body}"
                await self.send(reply)

        async def process_command(self, command):
            command = command.strip().lower()
            args = command.split()[1:]
            command = command.split()[0]

            if command == "forward":
                distance = args[0]
                distance = int(distance)
                logger.info(
                    f"[Behavior] Moving forward safely for {distance} mm"
                )
                self.agent.bot.safeForward(mm=distance)

            elif command == "turn":
                angle = args[0]
                angle = int(angle)
                logger.info("[Behavior] Turning...")
                self.agent.bot.turn(angle=angle)
            elif command == "full_calibration":
                logger.info("Starting full calibration")
                self.agent.bot.fullCalibration()
            elif command == "calibrate_turn":
                logger.info("Calibrating turn...")
                self.agent.bot.calibrateTurn()
            elif command == "calibrate_sensors":
                logger.info("[Behavior] Calibrating sensors...")
                self.agent.bot.calibrateTRSensors()
            elif command == "calibrate_forward":
                logger.info("[Behavior] Calibrating forward...")
                self.agent.bot.calibrateForward()
            elif command == "calibrate_forward_correction":
                logger.info("Calibrating forward correction")
                self.agent.bot.calibrateForwardCorrection()

            elif command.startswith("motor "):
                try:
                    _, left, right = command.split()
                    left_speed = int(left)
                    right_speed = int(right)
                    logger.info(
                        f"[Behavior] Setting motor speeds to {left_speed} (left) and {right_speed} (right)..."
                    )
                    self.agent.bot.setMotor(left_speed, right_speed)
                    await asyncio.sleep(2)
                    self.agent.bot.stop()
                except (ValueError, IndexError):
                    logger.error(
                        "[Behavior] Invalid motor command format. Use 'motor <left_speed> <right_speed>'"
                    )

            elif command == "stop":
                logger.info("[Behavior] Stopping...")
                self.agent.bot.stop()

            else:
                logger.warning(f"[Behavior] Unknown command: {command}")

    async def stop(self):
        if self.session:
            await self.session.close()
        await super().stop()


async def main():
    xmpp_domain = os.environ.get("XMPP_SERVER", "prosody")
    xmpp_username = os.environ.get("XMPP_USERNAME", "alpha-pi-zero-agent")
    xmpp_jid = f"{xmpp_username}@{xmpp_domain}"
    xmpp_password = os.environ.get("XMPP_PASSWORD", "top_secret")
    try:
        agent = AlphaBotAgent(
            jid=xmpp_jid, password=xmpp_password, verify_security=False
        )

        await agent.start(auto_register=True)

        try:
            while agent.is_alive():
                await asyncio.sleep(100)
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
            await agent.stop()
    except Exception as e:
        logger.error(f"Error starting agent: {str(e)}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Critical error in main loop: {str(e)}", exc_info=True)
