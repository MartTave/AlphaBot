import asyncio
import json
import logging
import os
import ssl

import aiosasl
import aioxmpp.security_layer
from spade.container import Container

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from alphabot_controller import AlphabotController
from calibration_sender import CalibrationSender
from camera_receiver import ReceiverAgent


def create_ssl_context():
    """Create SSL context with our certificate"""
    ctx = ssl.create_default_context()
    ctx.load_verify_locations("/app/certs/prosody.crt")
    return ctx


# Configure global security settings for SPADE
Container.security_layer = aioxmpp.security_layer.SecurityLayer(
    ssl_context_factory=create_ssl_context,
    certificate_verifier_factory=aioxmpp.security_layer.PKIXCertificateVerifier,
    tls_required=True,
    sasl_providers=[
        aiosasl.PLAIN(
            credential_provider=lambda _: (
                os.getenv("XMPP_JID"),
                os.getenv("XMPP_PASSWORD"),
            )
        )
    ],
)


async def run_alphabot_controller(instructions=[]):
    xmpp_jid = os.getenv("XMPP_JID")
    xmpp_password = os.getenv("XMPP_PASSWORD")
    robot_recipient = os.getenv("ROBOT_RECIPIENT")

    final_instructions = []

    for instr in instructions:
        final_instructions.append(instr["command"] + " ")
        for a in instr["args"]:
            string = a
            if a != instr["args"][-1]:
                string += " "
            final_instructions[-1] += string

    logger.info(f"Starting AlphabotController with JID: {xmpp_jid}")

    for i, instr in enumerate(final_instructions):
        logger.info(f"Instruction {i + 1}: {instr}")

    alphabot_controller = AlphabotController(
        jid=xmpp_jid, password=xmpp_password
    )
    await alphabot_controller.start(auto_register=True)

    send_instructions_behaviour = alphabot_controller.SendInstructionsBehaviour(
        robot_recipient, final_instructions
    )
    alphabot_controller.add_behaviour(send_instructions_behaviour)

    return alphabot_controller


async def run_camera_receiver():
    xmpp_jid = os.getenv("XMPP_JID")
    xmpp_password = os.getenv("XMPP_PASSWORD")

    logger.info(f"Starting CameraReceiver with JID: {xmpp_jid}")

    receiver = ReceiverAgent(xmpp_jid, xmpp_password)
    await receiver.start(auto_register=True)

    if not receiver.is_alive():
        logger.error("Camera receiver agent couldn't connect.")
        await receiver.stop()
        return None

    logger.info("Camera receiver agent started successfully.")
    return receiver


async def startCalibration():
    xmpp_jid = os.getenv("XMPP_JID")
    xmpp_password = os.getenv("XMPP_PASSWORD")

    calib_sender = CalibrationSender(xmpp_jid, xmpp_password)
    await calib_sender.start(auto_register=True)
    if not calib_sender.is_alive():
        logger.error("Calibration sender agent couldn't connect.")
        await calib_sender.stop()
        return None
    logger.info("Calibration sender agent started successfully.")

    return calib_sender


async def main(command_file="./commands/command.json"):
    os.makedirs("received_photos", exist_ok=True)

    with open(command_file, "r") as file:
        data = json.load(file)

    commands = data["commands"]

    try:
        alphabot_controller = await run_alphabot_controller(commands)
        camera_receiver = await run_camera_receiver()

        if not camera_receiver:
            logger.error(
                "Failed to start camera receiver. Stopping alphabot controller."
            )
            await alphabot_controller.stop()
            return

        logger.info("Both agents running. Press Ctrl+C to stop.")

        while any(
            behavior.is_running for behavior in alphabot_controller.behaviours
        ):
            await asyncio.sleep(1)

        logger.info("Alphabot controller has completed all instructions.")

        while camera_receiver.is_alive():
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    finally:
        if "alphabot_controller" in locals():
            await alphabot_controller.stop()
        if "camera_receiver" in locals():
            await camera_receiver.stop()
        logger.info("All agents stopped.")


if __name__ == "__main__":
    asyncio.run(main())
