import asyncio
import base64
import datetime
import os

import aiofiles
from spade import agent, behaviour
from spade.message import Message


class ReceiverAgent(agent.Agent):
    class RequestPhotoBehaviour(behaviour.OneShotBehaviour):
        async def run(self):
            # Create a message to request a photo
            msg = Message(
                to="camera_agent@prosody"
            )  # Replace with the actual JID of the camera agent
            msg.set_metadata("performative", "request")
            msg.body = "Requesting photo"

            # Send the message
            await self.send(msg)
            await self.send(msg)
            await asyncio.sleep(3.0)
            await self.send(msg)

            print("Request for photo sent.")

    class ReceivePhotoBehaviour(behaviour.CyclicBehaviour):
        async def run(self):
            print("Waiting for photo message...")
            msg = await self.receive(timeout=9999)
            if msg:
                print("Received photo message.")
                img_data = base64.b64decode(msg.body)

                # Create directory if it doesn't exist
                photos_dir = os.path.join(os.getcwd(), "received_photos")
                os.makedirs(photos_dir, exist_ok=True)

                # Generate filename with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"photo_{timestamp}.jpg"
                filepath = os.path.join(photos_dir, filename)

                # Save the received image
                async with aiofiles.open(filepath, "wb") as img_file:
                    await img_file.write(img_data)

                print(f"Photo saved as '{filepath}'.")

    async def setup(self):
        print(f"{self.jid} is ready.")
        self.add_behaviour(self.RequestPhotoBehaviour())
        self.add_behaviour(self.ReceivePhotoBehaviour())


async def main():
    xmpp_server = os.getenv("XMPP_SERVER", "localhost")
    xmpp_username = os.getenv("CAMERA_USERNAME", "receiver_agent")
    xmpp_password = os.getenv("CAMERA_PASSWORD", "top_secret")

    receiver = ReceiverAgent(f"{xmpp_username}@{xmpp_server}", xmpp_password)

    await receiver.start(auto_register=True)

    if not receiver.is_alive():
        print("Receiver agent couldn't connect. Check Prosody configuration.")
        await receiver.stop()
        return

    print("Receiver agent connected successfully. Running...")

    try:
        while receiver.is_alive():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down agent...")
    finally:
        # Clean up: stop the agent
        await receiver.stop()


if __name__ == "__main__":
    asyncio.run(main())
