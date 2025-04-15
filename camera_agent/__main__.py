import asyncio
import os

from src.camera_agent import CameraAgent


async def main():
    xmpp_server = os.environ.get("XMPP_SERVER", "prosody")
    xmpp_username = os.environ.get("XMPP_USERNAME", "camera_agent")
    xmpp_password = os.environ.get("XMPP_PASSWORD", "top_secret")
    
    sender_jid = f"{xmpp_username}@{xmpp_server}"
    sender_password = xmpp_password
    
    print(f"Connecting with JID: {sender_jid}")

    sender = CameraAgent(sender_jid, sender_password)

    await sender.start(auto_register=True)

    if not sender.is_alive():
        print("Camera agent couldn't connect. Check Prosody configuration.")
        await sender.stop()
        return

    print("Camera agent connected successfully. Running...")

    try:
        while sender.is_alive():
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down agent...")
    finally:
        # Clean up: stop the agent
        await sender.stop()



if __name__ == "__main__":
    asyncio.run(main())
