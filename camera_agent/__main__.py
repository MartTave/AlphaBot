import asyncio
import os

from src.camera_agent import CameraAgent


async def main():
    xmpp_server = os.environ.get("XMPP_SERVER", "prosody")
    xmpp_username = os.environ.get("XMPP_USERNAME", "camera_agent")
    xmpp_password = os.environ.get("XMPP_PASSWORD", "top_secret")
    http_port = int(os.environ.get("HTTP_PORT", "3001"))

    sender_jid = f"{xmpp_username}@{xmpp_server}"
    sender_password = xmpp_password

    print(f"Connecting with JID: {sender_jid}")
    print(f"HTTP server will run on port: {http_port}")

    sender = CameraAgent(sender_jid, sender_password, http_port)

    await sender.start(auto_register=True)

    if not sender.is_alive():
        print("Camera agent couldn't connect. Check Prosody configuration.")
        await sender.stop()
        return

    print("Camera agent connected successfully. Running...")
    print(f"Ban requests can be sent to: http://localhost:{http_port}/ban")

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
