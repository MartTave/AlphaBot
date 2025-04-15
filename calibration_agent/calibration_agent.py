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

        self.camera = Camera("Logitec_ceiling", -1, "./src/calibrations", focus=0)

    async def setup(self):
        self.add_behaviour(self.WaitForRequestBehavior())

    class WaitForRequestBehavior(CyclicBehaviour):
        async def run(self):
            msg = await self.receive(timeout=9999)
            if msg:
                sender = str(msg.sender)
                body = str(msg.body)
                if body == "get_calibration":
                    b = CalibrationAgent.SendCalibrationBehavior(requester_jid=sender, camera=self.agent.camera)
                    self.agent.add_behaviour(b)
                elif body == "start_calibration":
                    start_at = datetime.datetime.now() + datetime.timedelta(seconds=2)
                    b = CalibrationAgent.CalibrateBehavior(period=1, start_at=start_at,camera=self.agent.camera)
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
        def __init__(self, period, start_at, camera: Camera, camera_agent_jid: str = "camera_agent@prosody"):
            super().__init__(period=period, start_at=start_at)
            self.camera = camera
            self.camera_agent_jid = camera_agent_jid
            self.recipient_jid = camera_agent_jid  # Added missing recipient_jid

        async def on_start(self):
            self.counter = 0
            self.pics = []

        async def setup(self):
            start_at = datetime.datetime.now() + datetime.timedelta(seconds=3)
            b = self.InformBehav(period=1, start_at=start_at)
            self.add_behaviour(b)

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
                msg.body = "Requesting photo"
                
                await self.send(msg)
                msg = await self.receive(timeout=5000)
                if msg:
                    img_data = base64.b64decode(msg.body)
                    nparr = np.frombuffer(img_data, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    self.pics.append(img)
                    self.counter += 1
                else:
                    logger.warning("No response from camera agent")