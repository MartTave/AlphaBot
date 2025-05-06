import asyncio
import base64
import aiofiles
import logging
import numpy as np
import json
import cv2
import os
import datetime
from spade.agent import Agent  # Fixed import
from spade.behaviour import OneShotBehaviour, PeriodicBehaviour, CyclicBehaviour
from spade.message import Message

from camera_sync.camera import Camera

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

class CalibrationAgent(Agent):  # Changed from 'agent' to 'Agent'
    def __init__(self, jid, password):
        super().__init__(jid, password)

        self.camera_full = Camera("Logitec_ceiling", -1, "./src/calibrations", focus=0, resolution=(1920, 1080))
        self.camera_low = Camera("Logitec_ceiling", -1 , "./src/calibrations", focus=0, resolution=(640, 480))

    async def setup(self):
        self.add_behaviour(self.WaitForRequestBehavior())

    class WaitForRequestBehavior(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=9999)
            if msg:
                sender = str(msg.sender)
                body = str(msg.body)
                if body.startswith("get_calibration"):
                    quality = body.split(" ")[-1]
                    cam = None
                    if quality == "full":
                        cam = self.agent.camera_full
                    elif quality == "low":
                        cam = self.agent.camera_low
                    b = CalibrationAgent.SendCalibrationBehavior(requester_jid=sender, camera=cam)
                    self.agent.add_behaviour(b)
                elif body == "start_calibration":
                    quality = body.split(" ")[-1]
                    cam = None
                    if quality == "full":
                        cam = self.agent.camera_full
                    elif quality == "low":
                        cam = self.agent.camera_low
                    start_at = datetime.datetime.now() + datetime.timedelta(seconds=2)
                    b = CalibrationAgent.CalibrateBehavior(period=1, start_at=start_at,camera=cam, quality=quality)
                    self.agent.add_behaviour(b)

    class SendCalibrationBehavior(OneShotBehaviour):
        def __init__(self, requester_jid, camera: Camera):
            super().__init__()
            self.requester_jid = requester_jid
            self.camera = camera

        async def run(self):
            msg = Message(to=self.requester_jid)
            msg.set_metadata("performative", "inform")
            response = {}

            if not self.camera.calibrated:
                logger.warning("Camera not calibrated")
                response["success"] = False
                msg.body = json.dumps(response)
            else:
                response["success"] = True
                response["mtx"] = self.camera.mtx.tolist()
                response["dist"] = self.camera.dist.tolist()
                msg.body = json.dumps(response)

            await self.send(msg)

    class CalibrateBehavior(PeriodicBehaviour):
        def __init__(self, period, start_at, camera: Camera, camera_agent_jid: str = "camera_agent@prosody", quality="full"):
            super().__init__(period=period, start_at=start_at)
            self.camera = camera
            self.camera_agent_jid = camera_agent_jid
            self.recipient_jid = camera_agent_jid  # Added missing recipient_jid
            self.quality = quality

            self.image_chunks = {}
            self.total_chunks = 0
            self.total_size = 0
            self.received_chunks = 0
            self.complete_image = None

        async def on_start(self):
            self.counter = 0
            self.pics = []

        async def setup(self):
            start_at = datetime.datetime.now() + datetime.timedelta(seconds=3)
            b = self.InformBehav(period=1, start_at=start_at)
            self.add_behaviour(b)

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
            if self.counter == 9:
                logger.info("Enough picture, calbrating")
                mtx, dist = Camera.calibrate(self.pics)
                self.camera.mtx = mtx
                self.camera.dist = dist
                self.camera.calibrated = True
                self.camera.saveCalibration()
                self.kill()
            else:
                logger.info("Requesting photo from camera agent")
                msg = Message(to=self.recipient_jid)
                msg.set_metadata("performative", "inform")
                msg.body = f"{self.quality} quality"

                await self.send(msg)
                success = await self.receive_image_chunks()
                msg = await self.receive(timeout=5000)
                if msg:
                    img_data = base64.b64decode(msg.body)
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.pics.append(img)
                    self.counter += 1
                else:
                    logger.warning("No response from camera agent")
