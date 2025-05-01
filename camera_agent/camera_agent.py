import asyncio
import base64
import re
import datetime
from time import time

import aiofiles
import cv2
from aiohttp import web
from spade import agent, behaviour
from spade.message import Message


class CameraAgent(agent.Agent):
    def __init__(self, jid, password, http_port=3001):
        super().__init__(jid, password)
        # picture requests logging
        # as per default 500ms timeout
        self.timeout = 500  # ms
        self.ban_timeout = 10000  # ms
        self.requests: dict = {}
        self.http_port = http_port
        self.app = web.Application()
        self.app.add_routes(
            [
                web.post("/ban", self.handle_ban_request),
                web.get("/status", self.handle_status),
            ]
        )
        self.runner = None
        self.site = None

        # create event for ban concurrency
        # issues as request is ongoing process
        self.processing_complete = asyncio.Event()
        self.processing_complete.set()

    async def handle_ban_request(self, request):
        """Handle incoming ban requests via HTTP."""
        try:
            # Parse the request body
            data = await request.json()
            if not data or "agent" not in data:
                return web.json_response(
                    {"error": "Invalid request format"}, status=400
                )

            target_jid = data["agent"]
            now = time()

            reset_ban_timeout = lambda last: (
                lambda now: int(round((now - last) * 1000)) >= self.ban_timeout
            )
            # Apply the ban if there is
            # no registred behaviour
            await self.processing_complete.wait()
            self.requests[target_jid] = reset_ban_timeout(now)

            print(
                f"Agent {target_jid} has been banned for {self.ban_timeout}ms"
            )

            return web.json_response(
                {
                    "status": "success",
                    "message": f"Agent {target_jid} has been banned",
                    "ban_timeout": self.ban_timeout,
                }
            )

        except Exception as e:
            print(f"Error processing ban request: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_status(self, request):
        """Return status information about the camera agent."""
        return web.json_response(
            {
                "status": "online",
                "jid": str(self.jid),
                "banned_agents": len(self.requests),
            }
        )

    async def start_http_server(self):
        """Start the HTTP server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "0.0.0.0", self.http_port)
        await self.site.start()
        print(f"HTTP server started on port {self.http_port}")

    async def stop_http_server(self):
        """Stop the HTTP server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        print("HTTP server stopped")

    class SendPhotoBehaviour(behaviour.OneShotBehaviour):
        def __init__(self, requester_jid, quality="normal"):
            super().__init__()
            self.raw_requester_jid = requester_jid
            self.requester_jid = re.sub(r"(.*@.*)\/.*", r"\1", requester_jid)
            self.reset_timeout = lambda last: (
                lambda now: int(round((now - last) * 1000))
                >= self.agent.timeout
            )
            # Chunk size for splitting large images (100KB per chunk)
            self.chunk_size = 100 * 1024
            self.quality = quality

        async def run(self):
            self.agent.processing_complete.clear()
            now = time()

            # check if last request exceeded
            # predefined timeout
            if not self.agent.requests.get(self.requester_jid, lambda _: True)(
                now
            ):
                print(
                    f"Request from {self.requester_jid} under timeout. No response.."
                )

                msg = Message(to=self.raw_requester_jid)
                msg.set_metadata("performative", "info")
                msg.body = "Request cancelled due to ban"

                await self.send(msg)
                return

            print("Capturing image...")
            camera = self.agent.capture

            # Flush the camera buffer to get a real-time image
            for _ in range(5):  # Discard 5 frames to flush the buffer
                camera.grab()

            await asyncio.sleep(0.5)  # Small delay to stabilize

            if self.quality == "full":

                self.agent.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.agent.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            else:
                self.agent.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
                self.agent.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            ret, frame = camera.read()

            if not ret:
                print("Failed to capture image.")
                return

            # Apply some compression to keep good quality but smaller size
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            filename = "photo.jpg"
            cv2.imwrite(filename, frame, encode_params)

            # Get image dimensions for metadata
            height, width, channels = frame.shape
            image_format = "JPEG"
            quality = 85
            timestamp = datetime.datetime.now().isoformat()

            async with aiofiles.open(filename, "rb") as img_file:
                img_data = await img_file.read()
                encoded_img = base64.b64encode(img_data).decode("utf-8")

            # Check again if agent was banned during processing
            if not self.agent.requests.get(self.requester_jid, lambda _: True)(
                now
            ):
                print(
                    f"Agent {self.requester_jid} was banned during processing. Dropping response."
                )
                msg = Message(to=self.raw_requester_jid)
                msg.set_metadata("performative", "info")
                msg.body = "Request cancelled due to ban"
                await self.send(msg)
                return

            # Split the encoded image into chunks
            total_chunks = (len(encoded_img) + self.chunk_size - 1) // self.chunk_size
            total_size = len(encoded_img)
            print(f"Splitting image into {total_chunks} chunks")

            # Create a more descriptive metadata payload
            metadata_payload = f"Sending {total_chunks} chunks, total size {total_size} bytes, resolution {width}x{height}"

            # First, send metadata about the image
            metadata_msg = Message(to=self.raw_requester_jid)
            metadata_msg.set_metadata("performative", "inform")
            metadata_msg.set_metadata("content_type", "image_metadata")
            metadata_msg.set_metadata("total_chunks", str(total_chunks))
            metadata_msg.set_metadata("total_size", str(total_size))
            metadata_msg.body = metadata_payload
            await self.send(metadata_msg)
            await asyncio.sleep(0.1)  # Small delay between messages

            # Send each chunk
            for i in range(total_chunks):
                start_idx = i * self.chunk_size
                end_idx = min((i + 1) * self.chunk_size, len(encoded_img))
                chunk = encoded_img[start_idx:end_idx]

                chunk_msg = Message(to=self.raw_requester_jid)
                chunk_msg.set_metadata("performative", "inform")
                chunk_msg.set_metadata("content_type", "image_chunk")
                chunk_msg.set_metadata("chunk_index", str(i))
                chunk_msg.set_metadata("total_chunks", str(total_chunks))
                chunk_msg.body = chunk

                await self.send(chunk_msg)
                await asyncio.sleep(0.1)  # Small delay between chunks to avoid flooding

            print(f"Sent image in {total_chunks} chunks")

            self.agent.requests[self.requester_jid] = self.reset_timeout(now)
            self.agent.processing_complete.set()

    class WaitForRequestBehaviour(behaviour.CyclicBehaviour):
        def __init__(self):
            super().__init__()

        async def run(self):
            print("Waiting for request...")
            msg = await self.receive(timeout=9999)
            if msg:
                print("Received camera image request.")
                requester_jid = str(msg.sender)
                quality = str(msg.body)
                self.agent.add_behaviour(
                    self.agent.SendPhotoBehaviour(requester_jid, quality=quality)
                )

    async def setup(self):
        print(f"{self.jid} is ready.")
        # Start the HTTP server
        await self.start_http_server()

        self.capture = cv2.VideoCapture(0)
        # Set buffer size to 1 (minimum)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 854)
        # self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        # This is here to force the init the camera feed.
        self.capture.read()

        # Keep the XMPP behaviors for photo requests
        self.add_behaviour(self.WaitForRequestBehaviour())

    async def stop(self):
        # Stop the HTTP server first
        await self.stop_http_server()
        # Then stop the agent
        await super().stop()
