import asyncio
import logging
import os
import time
import RPi.GPIO as GPIO
import base64
import json
import numpy as np
import cv2
import io
from enum import Enum
import datetime
from PIL import Image
import aiohttp
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message

from alphabot_agent.alphabotlib.AlphaBot2 import AlphaBot2

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("AlphaBotAgent")

last_photo = None

# Enable SPADE and XMPP specific logging
for log_name in ["spade", "aioxmpp", "xmpp"]:
    log = logging.getLogger(log_name)
    log.setLevel(logging.ERROR)
    log.propagate = True


class BotState(Enum):
    IDLE = "idle"
    EXECUTING = "executing"


class AlphaBotAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.state = None

    async def setup(self):
        self.robot = AlphaBot2()

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
        # heartbeat_behavior = self.HeartbeatBehavior()
        # self.add_behaviour(heartbeat_behavior)

        # Add command listener behavior
        command_behavior = self.XMPPCommandListener()
        self.add_behaviour(command_behavior)

        # Set initial state after setup
        # await self.set_state(BotState.IDLE, "")

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
        def __init__(self, img, quality):
            super().__init__()
            self.img = img
            self.quality = quality

        async def run(self):

            self.agent.robot.processImage(self.img, self.quality)

            self.agent.add_behaviour(self.agent.AskPhotoBehaviour())

    class AskPhotoBehaviour(OneShotBehaviour):
        def __init__(self):
            super().__init__()
            self.response = None
            self.image_chunks = {}
            self.total_chunks = 0
            self.total_size = 0
            self.received_chunks = 0
            self.complete_image = None

        async def receive_image_chunks(self, timeout=30):
            """Receive and reassemble chunked image data"""
            start_time = datetime.datetime.now()
            timeout_delta = datetime.timedelta(seconds=timeout)

            # Wait for the metadata message first
            metadata_msg = await self.receive(timeout=5)
            if not metadata_msg:
                print("No metadata received within timeout period")
                return False

            if metadata_msg.get_metadata("content_type") != "image_metadata":
                print(f"Unexpected message type: {metadata_msg.get_metadata('content_type')}")
                return False

            # Parse metadata from message metadata instead of body
            try:
                # Get metadata from message metadata fields
                self.total_chunks = int(metadata_msg.get_metadata("total_chunks"))
                self.total_size = int(metadata_msg.get_metadata("total_size"))

                # Print the descriptive message from the body
                print(metadata_msg.body)

            except Exception as e:
                print(f"Error parsing metadata: {e}")
                return False

            # Receive all chunks
            while self.received_chunks < self.total_chunks:
                # Check timeout
                if datetime.datetime.now() - start_time > timeout_delta:
                    print(f"Timeout receiving chunks. Got {self.received_chunks} of {self.total_chunks}")
                    return False

                chunk_msg = await self.receive(timeout=5)
                if not chunk_msg:
                    print("No chunk received within timeout")
                    continue

                if chunk_msg.get_metadata("content_type") != "image_chunk":
                    print(f"Unexpected message type: {chunk_msg.get_metadata('content_type')}")
                    continue

                # Store the chunk
                chunk_index = int(chunk_msg.get_metadata("chunk_index"))
                self.image_chunks[chunk_index] = chunk_msg.body
                self.received_chunks += 1

                if self.received_chunks % 5 == 0:  # Log progress every 5 chunks
                    print(f"Received {self.received_chunks}/{self.total_chunks} chunks")

            # Reassemble the complete image
            if len(self.image_chunks) == self.total_chunks:
                reassembled = ""
                for i in range(self.total_chunks):
                    reassembled += self.image_chunks[i]

                if len(reassembled) == self.total_size:
                    self.complete_image = reassembled
                    print("Image reassembled successfully")
                    return True
                else:
                    print(f"Reassembled size ({len(reassembled)}) doesn't match expected size ({self.total_size})")
                    return False
            else:
                print(f"Missing chunks. Got {len(self.image_chunks)}/{self.total_chunks}")
                return False


        async def run(self):
            global last_photo
            msg = Message(to="camera_agent@prosody")
            msg.set_metadata("performative", "inform")
            if last_photo is None:
                quality = "full quality"
            else:
                quality = "low quality"

            msg.body = quality
            if last_photo is not None and time.time() - last_photo < 500:
                delay = (500 - (time.time() - last_photo))
                logger.info("500ms not elapsed since last request, sleeping for : " + str(delay))
                time.sleep(delay / 1000)
            await self.send(msg)
            success = await self.receive_image_chunks()
            if not success or self.complete_image is None:
                logger.error("Could not get full picture...")
                return
            msg = self.complete_image
            last_photo = time.time()
            if msg:
                img_data = base64.b64decode(msg)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                self.agent.add_behaviour(self.agent.ProcessImageBehaviour(img, quality))
            else:
                logger.error("Could not get photo in return of camera_agent")

    class XMPPCommandListener(OneShotBehaviour):

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
                # await self.agent.set_state(BotState.EXECUTING, command)

                # Process the command
                try:
                    await self.process_command(command)
                except Exception as e:
                    logger.error(e)

                # Set state back to IDLE after processing
                # await self.agent.set_state(BotState.IDLE, "")

                # Send a confirmation response
                reply = Message(to=str(msg.sender))
                reply.set_metadata("performative", "inform")
                reply.body = f"Executed command: {msg.body}"
                await self.send(reply)
            logger.info("loop done, restarting !")
            self.agent.add_behaviour(self.agent.XMPPCommandListener())

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
                self.agent.robot.safeForward(mm=distance)

            elif command == "turn":
                angle = args[0]
                angle = int(angle)
                logger.info("[Behavior] Turning...")
                self.agent.robot.turn(angle=angle)
            elif command == "full_calibration":
                logger.info("Starting full calibration")
                self.agent.robot.fullCalibration()
            elif command == "calibrate_turn":
                logger.info("Calibrating turn...")
                self.agent.robot.calibrateTurn()
            elif command == "calibrate_sensors":
                logger.info("[Behavior] Calibrating sensors...")
                self.agent.robot.calibrateTRSensors()
            elif command == "calibrate_forward":
                logger.info("[Behavior] Calibrating forward...")
                self.agent.robot.calibrateForward()
            elif command == "calibrate_forward_correction":
                logger.info("Calibrating forward correction")
                self.agent.robot.calibrateForwardCorrection()

            elif command == "solve":
                self.agent.add_behaviour(self.agent.AskPhotoBehaviour())
            elif command.startswith("motor "):
                try:
                    _, left, right = command.split()
                    left_speed = int(left)
                    right_speed = int(right)
                    logger.info(
                        f"[Behavior] Setting motor speeds to {left_speed} (left) and {right_speed} (right)..."
                    )
                    self.agent.robot.setMotor(left_speed, right_speed)
                    await asyncio.sleep(2)
                    self.agent.robot.stop()
                except (ValueError, IndexError):
                    logger.error(
                        "[Behavior] Invalid motor command format. Use 'motor <left_speed> <right_speed>'"
                    )

            elif command == "stop":
                logger.info("[Behavior] Stopping...")
                self.agent.robot.stop()

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
