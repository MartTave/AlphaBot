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
                instruction = "command:" + self.instructions[self.current_index]
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
