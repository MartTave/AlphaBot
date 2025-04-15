import base64
import aiofiles
import asyncio
from spade import agent, behaviour
from spade.message import Message

class ReceiverAgent(agent.Agent):
    class RequestPhotoBehaviour(behaviour.OneShotBehaviour):
        async def run(self):
            # Create a message to request a photo
            msg = Message(to="camera_agent@prosody")  # Replace with the actual JID of the camera agent
            msg.set_metadata("performative", "request")
            msg.body = "Requesting photo"
            
            # Send the message
            await self.send(msg)
            print("Request for photo sent.")

    class ReceivePhotoBehaviour(behaviour.CyclicBehaviour):
        async def run(self):
            print("Waiting for photo message...")
            msg = await self.receive(timeout=9999)
            if msg:
                print("Received photo message.")
                img_data = base64.b64decode(msg.body)

                # Save the received image
                async with aiofiles.open("received_photo.jpg", "wb") as img_file:
                    await img_file.write(img_data)

                print("Photo saved as 'received_photo.jpg'.")

    async def setup(self):
        print(f"{self.jid} is ready.")
        self.add_behaviour(self.RequestPhotoBehaviour())
        self.add_behaviour(self.ReceivePhotoBehaviour())

async def main():
    xmpp_server = "localhost"
    xmpp_username = "receiver_agent"
    xmpp_password = "top_secret"
    
    receiver_jid = f"{xmpp_username}@{xmpp_server}"
    receiver_password = xmpp_password
    
    print(f"Connecting with JID: {receiver_jid}")

    receiver = ReceiverAgent(receiver_jid, receiver_password)

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
