import asyncio
import base64

import aiofiles
import cv2
from spade import agent, behaviour
from spade.message import Message


class CameraAgent(agent.Agent):
    def __init__(self, jid, password):
        super().__init__(jid, password)

    class SendPhotoBehaviour(behaviour.OneShotBehaviour):
        def __init__(self, requester_jid):
            super().__init__()
            self.requester_jid = requester_jid

        async def run(self):
            print("Capturing image...")
            camera = cv2.VideoCapture(2)

            await asyncio.sleep(2)

            ret, frame = camera.read()

            if not ret:
                print("Failed to capture image.")
                return

            filename = "photo.jpg"
            cv2.imwrite(filename, frame)

            async with aiofiles.open(filename, "rb") as img_file:
                img_data = await img_file.read()
                encoded_img = base64.b64encode(img_data).decode("utf-8")

            msg = Message(to=self.requester_jid)
            msg.set_metadata("performative", "inform")
            msg.body = encoded_img

            await self.send(msg)
            print("Photo sent.")

    class WaitForRequestBehaviour(behaviour.CyclicBehaviour):
        async def run(self):
            print("Waiting for request...")
            msg = await self.receive(timeout=9999)
            if msg:
                print("Received camera image request.")
                requester_jid = str(msg.sender)
                self.agent.add_behaviour(
                    self.agent.SendPhotoBehaviour(requester_jid)
                )

    async def setup(self):
        print(f"{self.jid} is ready.")
        # Instead of immediately sending a photo, wait for requests
        self.add_behaviour(self.WaitForRequestBehaviour())
