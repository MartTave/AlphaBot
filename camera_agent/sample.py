import asyncio

from src.camera_agent import CameraAgent
from src.receiver_agent import ReceiverAgent


async def main():
    sender_jid = "sender@localhost/spade"
    sender_password = "password"
    receiver_jid = "receiver@localhost/spade"
    receiver_password = "password"

    # Instantiate agents
    receiver = ReceiverAgent(receiver_jid, receiver_password)
    sender = CameraAgent(sender_jid, sender_password)
    sender.receiver_jid = receiver_jid

    # Start both agents concurrently
    await asyncio.gather(
        receiver.start(auto_register=True),
        sender.start(auto_register=True)
    )

    # Confirm both agents are running
    if not (receiver.is_alive() and sender.is_alive()):
        print("One or both agents couldn't connect. Check Prosody configuration.")
        await asyncio.gather(receiver.stop(), sender.stop())
        return

    print("Both agents connected successfully. Waiting for message exchange...")

    try:
        # Run indefinitely while agents are alive
        while receiver.is_alive() and sender.is_alive():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down agents...")
    finally:
        # Clean up: stop both agents concurrently
        await asyncio.gather(sender.stop(), receiver.stop())


if __name__ == "__main__":
    asyncio.run(main())
