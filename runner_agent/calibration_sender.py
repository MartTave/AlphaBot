import datetime
import logging
import os

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message



class CalibrationSender(Agent):
    class SendMessageBehavior(OneShotBehaviour):
        def __init__(self, recipient_jid):
            super().__init__()
            self.recipient_jid = recipient_jid
            self.message_body = "<"

        async def run(self):
            msg = Message(to=self.recipient_jid)
            msg.set_metadata("performative", "inform")
            msg.body = self.message_body
            print("Sending message to calibration agent...")
            await self.send(msg)
            res = await self.receive(5000)
            print(res)
    
    async def setup(self):
        b = self.SendMessageBehavior(recipient_jid="calibration_agent@prosody")
        self.add_behaviour(b)