import datetime
import logging
import os
import base64
import io
from PIL import Image

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour
from spade.message import Message


class ImageTester(Agent):
    class SendMessageBehavior(OneShotBehaviour):
        def __init__(self):
            super().__init__()
            self.response = None
            self.image_chunks = {}
            self.total_chunks = 0
            self.total_size = 0
            self.received_chunks = 0
            self.complete_image = None

        async def receive_image_chunks(self, timeout=30):
            """Receive and reassemble chunked image data"""
            start_time = datetime.datetime.now()
            timeout_delta = datetime.timedelta(seconds=timeout)
            
            # Wait for the metadata message first
            metadata_msg = await self.receive(timeout=5)
            if not metadata_msg:
                print("No metadata received within timeout period")
                return False
                
            if metadata_msg.get_metadata("content_type") != "image_metadata":
                print(f"Unexpected message type: {metadata_msg.get_metadata('content_type')}")
                return False
                
            # Parse metadata from message metadata instead of body
            try:
                # Get metadata from message metadata fields
                self.total_chunks = int(metadata_msg.get_metadata("total_chunks"))
                self.total_size = int(metadata_msg.get_metadata("total_size"))
                
                # Print the descriptive message from the body
                print(metadata_msg.body)
                
            except Exception as e:
                print(f"Error parsing metadata: {e}")
                return False
                
            # Receive all chunks
            while self.received_chunks < self.total_chunks:
                # Check timeout
                if datetime.datetime.now() - start_time > timeout_delta:
                    print(f"Timeout receiving chunks. Got {self.received_chunks} of {self.total_chunks}")
                    return False
                    
                chunk_msg = await self.receive(timeout=5)
                if not chunk_msg:
                    print("No chunk received within timeout")
                    continue
                    
                if chunk_msg.get_metadata("content_type") != "image_chunk":
                    print(f"Unexpected message type: {chunk_msg.get_metadata('content_type')}")
                    continue
                    
                # Store the chunk
                chunk_index = int(chunk_msg.get_metadata("chunk_index"))
                self.image_chunks[chunk_index] = chunk_msg.body
                self.received_chunks += 1
                
                if self.received_chunks % 5 == 0:  # Log progress every 5 chunks
                    print(f"Received {self.received_chunks}/{self.total_chunks} chunks")
                    
            # Reassemble the complete image
            if len(self.image_chunks) == self.total_chunks:
                reassembled = ""
                for i in range(self.total_chunks):
                    reassembled += self.image_chunks[i]
                    
                if len(reassembled) == self.total_size:
                    self.complete_image = reassembled
                    print("Image reassembled successfully")
                    return True
                else:
                    print(f"Reassembled size ({len(reassembled)}) doesn't match expected size ({self.total_size})")
                    return False
            else:
                print(f"Missing chunks. Got {len(self.image_chunks)}/{self.total_chunks}")
                return False

        async def run(self):
            print(f"Hello from {self}")
            msg = Message(to="camera_agent@prosody")
            msg.set_metadata("performative", "inform")
            msg.body = "Requesting top-view picture"
            print("Sending message to camera_agent..")
            await self.send(msg)
            
            # Receive and reassemble chunked image
            success = await self.receive_image_chunks()
            
            if success and self.complete_image:
                print("Successfully received complete image")
                
                # Save and process the image
                try:
                    img_data = base64.b64decode(self.complete_image)
                    img = Image.open(io.BytesIO(img_data))
                    width, height = img.size
                    print(f"Image resolution: {width}x{height}")
                    
                    # Save the received image
                    os.makedirs("received_photos", exist_ok=True)
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"received_photos/image_{timestamp}.jpg"
                    with open(filename, "wb") as f:
                        f.write(img_data)
                    print(f"Image saved to {filename}")
                    
                except Exception as e:
                    print(f"Error processing image: {e}")
            else:
                print("Failed to receive complete image")
