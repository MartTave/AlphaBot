import logging
import os

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour, PeriodicBehaviour
from spade.message import Message

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AlphabotController(Agent):
    class SendMessageBehaviour(OneShotBehaviour):
        def __init__(self, recipient_jid, message_body):
            super().__init__()
            self.recipient_jid = recipient_jid
            self.message_body = message_body

        async def run(self):
            msg = Message(to=self.recipient_jid)
            msg.set_metadata("performative", "inform")
            msg.body = self.message_body

            logger.info(
                f"Sending message to {self.recipient_jid}: {self.message_body}"
            )
            await self.send(msg)
            logger.info("Message sent!")

    class SendInstructionsBehaviour(PeriodicBehaviour):
        def __init__(self, recipient_jid, instructions, period=5.0):
            super().__init__(period=period)
            self.recipient_jid = recipient_jid
            self.instructions = instructions
            self.current_index = 0

        async def run(self):
            if self.current_index < len(self.instructions):
                instruction = self.instructions[self.current_index]
                msg = Message(to=self.recipient_jid)
                msg.set_metadata("performative", "inform")
                msg.body = instruction

                logger.info(
                    f"Sending instruction to {self.recipient_jid}: {instruction}"
                )
                await self.send(msg)
                logger.info(
                    f"Instruction {self.current_index + 1}/{len(self.instructions)} sent!"
                )

                self.current_index += 1
            else:
                logger.info("All instructions sent. Stopping behavior.")
                self.kill()

    async def setup(self):
        logger.info("XMPP Client agent started")


async def main():
    xmpp_jid = os.getenv("XMPP_JID")
    xmpp_password = os.getenv("XMPP_PASSWORD")
    robot_recipient = os.getenv(
        "ROBOT_RECIPIENT", "alpha-pi-zero-agent@prosody"
    )

    # Get instructions from environment variable
    instructions_str = os.getenv("ROBOT_INSTRUCTIONS", "forward")
    instructions = [instr.strip() for instr in instructions_str.split(",")]

    logger.info(f"Starting XMPP client with JID: {xmpp_jid}")
    logger.info(f"Recipient: {robot_recipient}")
    logger.info(f"Instructions: {instructions}")

    alphabot_controller = AlphabotController(
        jid=xmpp_jid, password=xmpp_password
    )

    await alphabot_controller.start(auto_register=True)

    # Add behavior to send instructions periodically
    send_instructions_behaviour = alphabot_controller.SendInstructionsBehaviour(
        robot_recipient, instructions
    )
    alphabot_controller.add_behaviour(send_instructions_behaviour)

    try:
        # Keep the agent running until all instructions are sent
        while any(
            behavior.is_running
            for behavior in alphabot_controller.behaviours.values()
        ):
            await asyncio.sleep(1)

        # Wait a bit more before stopping
        await asyncio.sleep(2)
    finally:
        await alphabot_controller.stop()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
