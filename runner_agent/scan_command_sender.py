import logging
import os

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message


class ScanCommandSender(Agent):
    class SendScanCommandBehaviour(OneShotBehaviour):
        def __init__(self, recipient_jid):
            super().__init__()
            self.recipient_jid = recipient_jid
            self.message_body = "scan"

        async def run(self):
            msg = Message(to=self.recipient_jid)
            msg.set_metadata("performative", "inform")
            msg.body = self.message_body
            print(f"Sending 'scan' command to {self.recipient_jid}...")
            await self.send(msg)
            
            # Wait for a response with a 5-second timeout
            response = await self.receive(5000)
            if response:
                print(f"Received response: {response}")
            else:
                print("No response received within timeout period")
    
    async def setup(self):
        b = self.SendScanCommandBehaviour(recipient_jid="gate_handler@prosody")
        self.add_behaviour(b)