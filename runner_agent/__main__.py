import asyncio
import json
import logging
import os
import ssl

import aiosasl
import aioxmpp.security_layer
from spade.container import Container
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message
from .image_tester import ImageTester

XMPP_JID = os.getenv("XMPP_JID", "runner") + "@prosody"

TARGET = os.getenv("RUNNER_TARGET")

if TARGET is None:
    raise Exception("RUNNER_TARGET environment variable is not set")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from .alphabot_controller import AlphabotController
from .calibration_sender import CalibrationSender
from .camera_receiver import ReceiverAgent


class ScanCommandSender(Agent):
    class SendScanCommandBehaviour(OneShotBehaviour):
        def __init__(self, recipient_jid):
            super().__init__()
            self.recipient_jid = recipient_jid
            self.message_body = "command:scan"
            self.result = None

        async def run(self):
            msg = Message(to=self.recipient_jid)
            msg.set_metadata("performative", "inform")
            msg.body = self.message_body
            logger.info(f"Sending 'scan' command to {self.recipient_jid}...")
            await self.send(msg)

            # Wait for a response with a 5-second timeout
            response = await self.receive(5000)
            if response:
                logger.info(f"Received response from gate handler: {response.body}")
                self.result = response.body
            else:
                logger.warning("No response received from gate handler within timeout period")

    async def setup(self):
        pass  # We'll add behaviors dynamically


async def run_scan_command_sender():
    """Initialize and run the scan command sender agent"""
    xmpp_jid = XMPP_JID
    xmpp_password = os.getenv("XMPP_PASSWORD")

    logger.info(f"Starting ScanCommandSender with JID: {xmpp_jid}")

    # Create and start the agent
    scan_sender = ScanCommandSender(xmpp_jid, xmpp_password)
    await scan_sender.start(auto_register=True)

    # Check if agent started successfully
    if not scan_sender.is_alive():
        logger.error("Scan command sender agent couldn't connect.")
        await scan_sender.stop()
        return None

    logger.info("Scan command sender agent started successfully.")

    # Add and run the scan behavior
    scan_behavior = scan_sender.SendScanCommandBehaviour("gate_handler@prosody")
    scan_sender.add_behaviour(scan_behavior)

    # Wait for the behavior to complete
    while scan_behavior.is_running:
        await asyncio.sleep(0.5)

    # Get the result
    result = scan_behavior.result

    # Stop the agent
    await scan_sender.stop()

    return result


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
                XMPP_JID,
                os.getenv("XMPP_PASSWORD"),
            )
        )
    ],
)


async def run_alphabot_controller(recipient, instructions):
    xmpp_jid = XMPP_JID
    xmpp_password = os.getenv("XMPP_PASSWORD")

    final_instructions = []

    for instr in instructions:
        final_instructions.append(instr["command"] + " ")
        if "args" in instr:
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
        recipient, final_instructions
    )
    alphabot_controller.add_behaviour(send_instructions_behaviour)

    return alphabot_controller


async def run_camera_receiver():
    xmpp_jid = XMPP_JID
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
    xmpp_jid = XMPP_JID
    xmpp_password = os.getenv("XMPP_PASSWORD")

    calib_sender = CalibrationSender(xmpp_jid, xmpp_password)
    await calib_sender.start(auto_register=True)
    if not calib_sender.is_alive():
        logger.error("Calibration sender agent couldn't connect.")
        await calib_sender.stop()
        return None
    logger.info("Calibration sender agent started successfully.")

    return calib_sender


async def main(target, command_file="/app/src/commands/command.json"):
    os.makedirs("received_photos", exist_ok=True)

    with open(command_file, "r") as file:
        data = json.load(file)
    commands = data["commands"]

    # First, send the scan command to gate_handler
    logger.info("Sending scan command to gate handler...")
    # scan_result = await run_scan_command_sender()

    # if scan_result:
    #     logger.info(f"Scan completed with result: {scan_result}")
    # else:
    #     logger.warning("Scan command completed but no result was returned")

    # Start all controllers concurrently
    target_ids = target.split(",")
    controllers = []
    running_tasks = []

    try:
        # Start all controllers concurrently
        for target_id in target_ids:
            target_bot = f"alpha-pi-4b-agent-{target_id}@prosody"
            logger.info(f"Starting alphabot controller {target_bot} to run commands...")
            alphabot_controller = await run_alphabot_controller(target_bot, commands)
            controllers.append(alphabot_controller)

        logger.info(f"Started {len(controllers)} controllers concurrently. Press Ctrl+C to stop.")

        # Wait for all controllers to complete their instructions
        while any(
            any(behavior.is_running for behavior in controller.behaviours)
            for controller in controllers
        ):
            await asyncio.sleep(1)

        logger.info("All alphabot controllers have completed their instructions.")

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        # Stop all controllers
        for controller in controllers:
            if controller is not None:
                await controller.stop()
        logger.info("All agents stopped.")

if __name__ == "__main__":
    asyncio.run(main(target=TARGET))
