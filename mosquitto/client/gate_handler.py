import asyncio
import functools
import sys
import logging
import json
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("GateHandler")

# Force Python to use unbuffered stdout
import os
os.environ['PYTHONUNBUFFERED'] = '1'

logger.info("Starting gate handler script with NEW EVENT FLOW:")
logger.info("1. First starting gate (gate1/gate2) → global_start")
logger.info("2. Second starting gate (gate1/gate2) → no event")  
logger.info("3. First ending gate (gate3/gate4) → delta_start")
logger.info("4. Second ending gate (gate3/gate4) → global_end")

import paho.mqtt.client as mqtt
from bleak import BleakClient, BleakScanner
from bleak.exc import BleakError

# Import SPADE components for XMPP
from spade.agent import Agent
from spade.behaviour import CyclicBehaviour, OneShotBehaviour
from spade.message import Message


class gate:
    # These should be fixed for every gate. If it doesn't work, do "a ble-scan -d [address of the device]" to get the real ones
    BUTTON_UUID = "794f1fe3-9be8-4875-83ba-731e1037a881"
    SENSOR_UUID = "794f1fe3-9be8-4875-83ba-731e1037a883"

    def __init__(self, address, name):
        self.address = address
        self.name = name
        self.client = None
        self.device = None
        self.connected = False
        self.last_reconnect_attempt = 0
        self.reconnect_interval = 5  # seconds between reconnection attempts
        self.reconnect_attempts = 0  # Track number of reconnect attempts

    async def connect(self):
        try:
            self.device = await BleakScanner.find_device_by_address(self.address)
            if not self.device:
                logger.warning(f"Device {self.name} ({self.address}) not found!")
                self.connected = False
                return False
            else:
                logger.info(f"Found device {self.name} ({self.address})!")
                self.client = BleakClient(self.device, disconnected_callback=self._on_disconnect)
                logger.info(f"Connecting to {self.name}...")
                await self.client.connect()
                logger.info(f"Connection successful to {self.name}.")
                self.connected = True
                return True
        except BleakError as e:
            logger.error(f"Connection error to {self.name}: {str(e)}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to {self.name}: {str(e)}")
            self.connected = False
            return False
    
    def _on_disconnect(self, client):
        logger.warning(f"⚠️ Gate {self.name} disconnected!")
        self.connected = False

    async def disconnect(self):
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            self.connected = False
            logger.info(f"Disconnected from {self.name}")

    async def try_reconnect(self):
        """Attempt to reconnect the gate if it's disconnected"""
        now = time.time()
        if not self.connected and (now - self.last_reconnect_attempt) > self.reconnect_interval:
            self.last_reconnect_attempt = now
            self.reconnect_attempts += 1
            logger.info(f"Attempting to reconnect to {self.name} (attempt #{self.reconnect_attempts})...")
            return await self.connect()
        return self.connected

    async def start_listening_sensor(self, func):
        if self.connected and self.client:
            try:
                await self.client.start_notify(self.SENSOR_UUID, func)
                logger.info(f"Started sensor notifications for {self.name}")
                return True
            except Exception as e:
                logger.error(f"Error starting notifications on {self.name}: {str(e)}")
                self.connected = False
                return False
        return False

    async def stop_listening_sensor(self):
        if self.connected and self.client:
            try:
                await self.client.stop_notify(self.SENSOR_UUID)
                return True
            except Exception:
                return False
        return False

    def is_connected(self):
        return self.connected


class MQTTGateHandler:
    def __init__(self, broker_address="mosquitto", broker_port=1883):
        # MQTT setup
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.broker_address = broker_address
        self.broker_port = broker_port

        # Create event loop for callbacks
        self.loop = asyncio.get_event_loop()

        # Gates setup
        # Gates are pre-assigned to a pair
        # as they are named {A,B}{0,1}
        self.gates = {
            "gate1": "F3:E4:9B:6F:CD:EC",
            "gate2": "F9:CB:B8:48:B5:92",
            "gate3": "D0:31:D0:79:F2:F3",
            "gate4": "EF:3E:B7:B5:D3:F9",
        }
        # No longer need gate pairs since starting/ending gates aren't paired
        # Just keeping this for compatibility with existing code
        self.gate_pairs = {}
        self.gate_instances = {}
        
        # Gate connection status
        self.all_gates_connected = False
        self.connected_gate_count = 0

        # Define starting gates and ending gates
        # In our setup, gate1 and gate2 are starting gates, gate3 and gate4 are ending gates
        self.starting_gates = ["gate1", "gate2"]
        self.ending_gates = ["gate3", "gate4"]

        # Gate passage tracking
        self.passed_gates = {}  # Dictionary tracking passage order
        self.gate_count = 0     # Counter to track passage order

        # Event tracking
        self.first_starting_gate = None  # The first starting gate passed
        self.second_starting_gate = None # The second starting gate passed
        self.first_ending_gate = None    # The first ending gate passed

        # Define event types - these are simple functions that return event names
        self.events = {
            "global_start": lambda: "global_start",
            "delta_start": lambda: "delta_start",
            "global_end": lambda: "global_end"
        }
        
        # MQTT topics
        self.reset_topic = "timer/reset"
        self.status_topic = "timer/status"
        
        # System is active only if all gates are connected
        self.system_active = False
        
    def on_connect(self, client, userdata, flags, rc, properties=None):
        logger.info(f"Connected to MQTT broker with result code {rc}")
        # Subscribe to reset topic
        client.subscribe(self.reset_topic)
        logger.info(f"Subscribed to reset topic: {self.reset_topic}")
        
    def on_message(self, client, userdata, msg):
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            logger.info(f"Received message on topic {topic}: {payload}")
            
            if topic == self.reset_topic and payload.lower() == "reset":
                logger.info("Manual reset command received via MQTT")
                self.reset_gate_status()
                # Publish confirmation
                self.mqtt_client.publish(self.status_topic, "reset_complete")
        except Exception as e:
            logger.error(f"Error processing message: {str(e)}")

    def sensor_callback(self, gate_name, sender, data):
        # Only process gate events if the system is active
        if self.system_active:
            asyncio.create_task(self.sensor_handler(gate_name, sender, data))
        else:
            logger.debug(f"Ignoring gate event from {gate_name} - system not active")

    async def sensor_handler(self, gate_name, sender, data):
        """
        Handle sensor events from gates.

        The correct event flow is:
        1. First starting gate passage (gate1 or gate2) -> global_start
        2. Second starting gate passage (the other of gate1/gate2) -> no event
        3. First ending gate passage (gate3 or gate4) -> delta_start 
        4. Second ending gate passage (the other of gate3/gate4) -> global_end
        """
        # Convert the hex data to a meaningful value
        # Assuming the sensor data indicates a passage when it's non-zero
        if any(data):  # Check if any byte in data is non-zero
            timestamp = int(datetime.now().timestamp() * 1000)
            # Initialize variables
            event_type = ""
            should_reset = False

            # Skip if we've already seen this gate
            if gate_name in self.passed_gates:
                logger.info(f"Gate {gate_name} already passed, ignoring repeated passage")
                return

            # Record this gate passage with its order
            self.gate_count += 1
            self.passed_gates[gate_name] = self.gate_count
            logger.info(f"Gate passage #{self.gate_count} detected at {gate_name}")

            # CASE 1: First starting gate passage (gate1 or gate2)
            if gate_name in self.starting_gates and self.first_starting_gate is None:
                self.first_starting_gate = gate_name
                event_type = self.events["global_start"]()
                logger.info(f"First starting gate passage at {gate_name}, triggering {event_type}")

            # CASE 2: Second starting gate passage (the other of gate1/gate2)
            elif gate_name in self.starting_gates and self.second_starting_gate is None:
                self.second_starting_gate = gate_name
                logger.info(f"Second starting gate passage at {gate_name}, no event triggered")

            # CASE 3: First ending gate passage (gate3 or gate4)
            elif gate_name in self.ending_gates and self.first_ending_gate is None:
                self.first_ending_gate = gate_name
                event_type = self.events["delta_start"]()
                logger.info(f"First ending gate passage at {gate_name}, triggering {event_type}")

            # CASE 4: Second ending gate passage (the other of gate3/gate4)
            elif gate_name in self.ending_gates and gate_name != self.first_ending_gate:
                event_type = self.events["global_end"]()
                logger.info(f"Second ending gate passage at {gate_name}, triggering {event_type}")
                
                # Complete event flow detected - auto reset the system for a new round
                logger.info("🔄 Complete event flow detected - Automatically resetting for a new round 🔄")
                # We'll reset after publishing the event
                should_reset = True
            
            # Any other gate passage (shouldn't happen with this logic but just in case)
            else:
                logger.info(f"Gate {gate_name} passage doesn't match any event trigger conditions")
                should_reset = False
                
            if event_type:
                logger.info(f"Event triggered: {event_type}")
                message = {
                    "timestamp": timestamp,
                    "event": event_type,
                }
                # Publish to MQTT topic specific to this gate
                self.mqtt_client.publish("timer", str(message))
                logger.info(f"Published passage event for {gate_name}: {event_type}")

            # Log current state after processing
            self.log_status()
            
            # Reset system if we completed the full event flow
            if should_reset:
                # Add a small delay to make sure events are processed
                await asyncio.sleep(0.5)
                self.reset_gate_status()
                logger.info("🟢 System reset complete - Ready for new race 🟢")

    async def setup_gates(self):
        """Initial setup of all gates - creates gate instances but doesn't connect yet"""
        for gate_name, address in self.gates.items():
            gate_instance = gate(address, gate_name)
            self.gate_instances[gate_name] = gate_instance
            logger.info(f"Created gate instance for {gate_name}")
        
        # Just initialize the gate instances without connecting
        logger.info("Gate instances created but not connecting yet - waiting for scan command")
        
    async def connect_all_gates(self):
        """Attempt to connect to all gates"""
        connected_count = 0
        total_gates = len(self.gate_instances)
        
        for gate_name, gate_instance in self.gate_instances.items():
            if await gate_instance.connect():
                # Create a callback that properly handles the async nature of sensor_handler
                callback = functools.partial(self.sensor_callback, gate_name)
                if await gate_instance.start_listening_sensor(callback):
                    connected_count += 1
                    logger.info(f"✅ {gate_name} connected and listening")
                else:
                    logger.warning(f"⚠️ {gate_name} connected but failed to start notifications")
            else:
                logger.warning(f"❌ Failed to connect to {gate_name}")
                
        self.connected_gate_count = connected_count
        self.update_system_status()
        
        return connected_count == total_gates

    async def check_reconnect_gates(self):
        """Check all gates and attempt to reconnect any disconnected ones"""
        prev_connected_count = self.connected_gate_count
        self.connected_gate_count = 0
        
        for gate_name, gate_instance in self.gate_instances.items():
            # If gate is disconnected, try to reconnect
            if not gate_instance.is_connected():
                if await gate_instance.try_reconnect():
                    # Successfully reconnected, start listening again
                    callback = functools.partial(self.sensor_callback, gate_name)
                    await gate_instance.start_listening_sensor(callback)
                    logger.info(f"✅ Reconnected to {gate_name} successfully")
                else:
                    logger.debug(f"Still waiting for {gate_name} to become available...")
            
            # Count connected gates
            if gate_instance.is_connected():
                self.connected_gate_count += 1
        
        # Update system status if connection count changed
        if prev_connected_count != self.connected_gate_count:
            self.update_system_status()
            
        return self.all_gates_connected

    def update_system_status(self):
        """Update system status based on connected gates"""
        total_gates = len(self.gate_instances)
        prev_status = self.system_active
        
        self.all_gates_connected = (self.connected_gate_count == total_gates)
        self.system_active = self.all_gates_connected
        
        # Report status changes
        if self.system_active != prev_status:
            if self.system_active:
                status_msg = f"🟢 SYSTEM ACTIVE: All {total_gates} gates connected"
                self.mqtt_client.publish(self.status_topic, "all_gates_connected")
            else:
                status_msg = f"🔴 SYSTEM INACTIVE: Only {self.connected_gate_count}/{total_gates} gates connected"
                self.mqtt_client.publish(self.status_topic, f"gates_disconnected:{total_gates-self.connected_gate_count}")
            
            logger.info(status_msg)
        
        # Regular status report
        logger.info(f"Gate Status: {self.connected_gate_count}/{total_gates} connected")
        for name, gate_inst in self.gate_instances.items():
            logger.info(f"  - {name}: {'✅ Connected' if gate_inst.is_connected() else '❌ Disconnected'}")

    def reset_gate_status(self):
        """Reset all gate passage tracking and prepare for a new race"""
        if self.gate_count > 0:
            logger.info(f"===== RESETTING GATE SYSTEM =====")
            logger.info(f"Gates passed in this session: {self.passed_gates}")
            logger.info(f"First starting gate was: {self.first_starting_gate}")
            logger.info(f"Second starting gate was: {self.second_starting_gate}")
            logger.info(f"First ending gate was: {self.first_ending_gate}")
            logger.info(f"===== RESET COMPLETE =====")
            
        # Reset all tracking variables
        self.first_starting_gate = None
        self.second_starting_gate = None
        self.first_ending_gate = None
        self.passed_gates.clear()
        self.gate_count = 0

    def log_status(self):
        """Log the current status of gate passages"""
        logger.info(f"Gate Status - First start gate: {self.first_starting_gate}, Second start gate: {self.second_starting_gate}, " +
                   f"First end gate: {self.first_ending_gate}, Passages: {self.passed_gates}")

    async def run(self):
        try:
            # Connect to MQTT broker
            logger.info("Connecting to MQTT broker...")
            self.mqtt_client.connect(self.broker_address, self.broker_port, 60)
            # Start MQTT loop in a non-blocking way
            self.mqtt_client.loop_start()

            # Setup BLE gates
            logger.info("Setting up BLE gates...")
            await self.setup_gates()

            # Keep the program running and check gate connections
            logger.info("Gate handler running and monitoring gate connections...")
            logger.info(f"AUTO-RESET: Enabled - System will reset after full event flow")
            logger.info(f"MANUAL-RESET: Available via MQTT topic '{self.reset_topic}' with payload 'reset'")
            
            counter = 0
            while True:
                await asyncio.sleep(1)
                
                # Check/reconnect gates every 5 seconds
                counter += 1
                if counter % 5 == 0:
                    await self.check_reconnect_gates()
                
                # Log full status every minute
                if counter % 60 == 0:
                    self.log_status()
                    counter = 0

        except KeyboardInterrupt:
            logger.info("Shutting down...")
        finally:
            # Cleanup
            for gate_name, gate_instance in self.gate_instances.items():
                await gate_instance.stop_listening_sensor()
                await gate_instance.disconnect()
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()


class GateHandlerAgent(Agent):
    """
    XMPP Agent that handles gate operations and responds to XMPP messages.
    Integrates with MQTTGateHandler for BLE gate functionality.
    """
    def __init__(self, jid, password, mqtt_broker_address="mosquitto", mqtt_broker_port=1883):
        super().__init__(jid, password)
        # Create the MQTT gate handler as a component
        self.gate_handler = MQTTGateHandler(broker_address=mqtt_broker_address, broker_port=mqtt_broker_port)
        self.mqtt_initialized = False
        
        # Log XMPP connection details for debugging
        logger.info(f"Initializing GateHandlerAgent with JID: {jid}")

    async def setup(self):
        """Set up the agent behaviors when it starts"""
        logger.info(f"XMPP Gate Handler Agent {str(self.jid)} starting...")
        # Add the behavior to listen for XMPP messages
        self.add_behaviour(self.ListenForCommandsBehavior())
        
        # Start the gate handler initialization in the background
        self.add_behaviour(self.InitializeGateHandlerBehavior())

    class InitializeGateHandlerBehavior(OneShotBehaviour):
        """Behavior to initialize the MQTT gate handler"""
        async def run(self):
            logger.info("Initializing MQTT Gate Handler...")
            # Connect to MQTT broker
            self.agent.gate_handler.mqtt_client.connect(
                self.agent.gate_handler.broker_address, 
                self.agent.gate_handler.broker_port, 
                60
            )
            # Start MQTT loop in a non-blocking way
            self.agent.gate_handler.mqtt_client.loop_start()
            
            # Setup BLE gates
            logger.info("Setting up BLE gates...")
            await self.agent.gate_handler.setup_gates()
            
            # Mark as initialized
            self.agent.mqtt_initialized = True
            
            # Add the cyclic behavior to check gate connections
            gate_monitor = self.agent.MonitorGatesBehavior()
            self.agent.add_behaviour(gate_monitor)
            
            logger.info("Gate handler initialization complete")

    class ListenForCommandsBehavior(CyclicBehaviour):
        """Behavior to listen for incoming XMPP commands"""
        async def run(self):
            msg = await self.receive(timeout=10)  # Wait for messages
            
            if msg:
                sender = str(msg.sender)
                command = msg.body.strip().lower()
                logger.info(f"Received command '{command}' from {sender}")
                
                if command == "scan":
                    # Add behavior to initiate gate scanning
                    logger.info(f"Received 'scan' command from {sender}, initiating gate scanning")
                    self.agent.add_behaviour(
                        self.agent.ScanGatesBehavior(requester_jid=sender)
                    )
                elif command == "gate_status":
                    # Add behavior to send gate status
                    self.agent.add_behaviour(
                        self.agent.SendGateStatusBehavior(requester_jid=sender)
                    )
                elif command == "reset_gates":
                    # Reset the gates and send confirmation
                    self.agent.gate_handler.reset_gate_status()
                    response = Message(to=sender)
                    response.set_metadata("performative", "inform")
                    response.body = json.dumps({"status": "Gates reset successfully"})
                    await self.send(response)

    class ScanGatesBehavior(OneShotBehaviour):
        """
        Behavior to scan for BLE gates with limited reconnection attempts.
        Will try to connect to gates a maximum of 3 times before giving up.
        """
        def __init__(self, requester_jid):
            super().__init__()
            self.requester_jid = requester_jid
            self.max_reconnect_attempts = 3
            
        async def run(self):
            logger.info(f"Initiating gate scanning for {self.requester_jid}...")
            
            # Reset reconnect attempts counter for each gate
            for name, gate_inst in self.agent.gate_handler.gate_instances.items():
                gate_inst.reconnect_attempts = 0
                
            # First attempt to connect to all gates
            connected = await self.scan_gates_with_retry()
            
            # Prepare response message
            response = Message(to=self.requester_jid)
            response.set_metadata("performative", "inform")
            
            # Get gate connection status
            gate_status = {}
            for name, gate_inst in self.agent.gate_handler.gate_instances.items():
                gate_status[name] = {
                    "connected": gate_inst.is_connected(),
                    "address": gate_inst.address,
                    "reconnect_attempts": getattr(gate_inst, 'reconnect_attempts', 0)
                }
            
            # Create response
            result = {
                "success": connected,
                "gates_connected": self.agent.gate_handler.connected_gate_count,
                "total_gates": len(self.agent.gate_handler.gate_instances),
                "scan_complete": True,
                "gate_status": gate_status
            }
            
            response.body = json.dumps(result)
            await self.send(response)
            logger.info(f"Gate scanning results sent to {self.requester_jid}")
            
        async def scan_gates_with_retry(self):
            """
            Attempt to scan and connect to all gates with limited retries.
            Returns True if all gates connected, False otherwise.
            """
            # First attempt
            logger.info("Starting initial gate scan (attempt 1 of 3)...")
            connected_gates = []
            
            # Try to connect to each gate
            for name, gate_inst in self.agent.gate_handler.gate_instances.items():
                if not gate_inst.is_connected():
                    gate_inst.reconnect_attempts = 1  # Count first attempt
                    if await gate_inst.connect():
                        # If connected, start notifications
                        callback = functools.partial(self.agent.gate_handler.sensor_callback, name)
                        await gate_inst.start_listening_sensor(callback)
                        connected_gates.append(name)
                        logger.info(f"✅ Gate {name} connected on attempt 1")
                    else:
                        logger.warning(f"❌ Gate {name} failed on connection attempt 1")
                else:
                    connected_gates.append(name)
                    logger.info(f"✅ Gate {name} already connected")
            
            # If not all gates connected, try a second attempt
            if len(connected_gates) < len(self.agent.gate_handler.gate_instances):
                logger.info("Some gates failed to connect. Starting second attempt (attempt 2 of 3)...")
                
                # Wait briefly before retry
                await asyncio.sleep(2)
                
                # Try to connect to each gate that failed
                for name, gate_inst in self.agent.gate_handler.gate_instances.items():
                    if name not in connected_gates and gate_inst.reconnect_attempts < self.max_reconnect_attempts:
                        gate_inst.reconnect_attempts = 2  # Count second attempt
                        if await gate_inst.connect():
                            # If connected, start notifications
                            callback = functools.partial(self.agent.gate_handler.sensor_callback, name)
                            await gate_inst.start_listening_sensor(callback)
                            connected_gates.append(name)
                            logger.info(f"✅ Gate {name} connected on attempt 2")
                        else:
                            logger.warning(f"❌ Gate {name} failed on connection attempt 2")
            
            # If still not all gates connected, try a third attempt
            if len(connected_gates) < len(self.agent.gate_handler.gate_instances):
                logger.info("Some gates still failed to connect. Starting final attempt (attempt 3 of 3)...")
                
                # Wait a bit longer before final retry
                await asyncio.sleep(3)
                
                # Try to connect to each gate that failed
                for name, gate_inst in self.agent.gate_handler.gate_instances.items():
                    if name not in connected_gates and gate_inst.reconnect_attempts < self.max_reconnect_attempts:
                        gate_inst.reconnect_attempts = 3  # Count third attempt
                        if await gate_inst.connect():
                            # If connected, start notifications
                            callback = functools.partial(self.agent.gate_handler.sensor_callback, name)
                            await gate_inst.start_listening_sensor(callback)
                            connected_gates.append(name)
                            logger.info(f"✅ Gate {name} connected on final attempt")
                        else:
                            logger.warning(f"❌ Gate {name} failed on final connection attempt")
            
            # Update connection count
            self.agent.gate_handler.connected_gate_count = len(connected_gates)
            self.agent.gate_handler.update_system_status()
            
            # If all gates connected, consider it a success
            all_connected = len(connected_gates) == len(self.agent.gate_handler.gate_instances)
            
            if all_connected:
                logger.info("✅ All gates successfully connected!")
            else:
                logger.warning(f"⚠️ Not all gates could be connected. Connected {len(connected_gates)} of {len(self.agent.gate_handler.gate_instances)}")
            
            return all_connected

    class SendGateStatusBehavior(OneShotBehaviour):
        """Behavior to send current gate status"""
        def __init__(self, requester_jid):
            super().__init__()
            self.requester_jid = requester_jid
            
        async def run(self):
            logger.info(f"Sending gate status to {self.requester_jid}...")
            
            # Prepare response message
            response = Message(to=self.requester_jid)
            response.set_metadata("performative", "inform")
            
            # Gather status information
            status = {
                "system_active": self.agent.gate_handler.system_active,
                "connected_gates": self.agent.gate_handler.connected_gate_count,
                "total_gates": len(self.agent.gate_handler.gate_instances),
                "gates": {},
                "race_status": {
                    "first_starting_gate": self.agent.gate_handler.first_starting_gate,
                    "second_starting_gate": self.agent.gate_handler.second_starting_gate,
                    "first_ending_gate": self.agent.gate_handler.first_ending_gate,
                    "passed_gates": self.agent.gate_handler.passed_gates
                }
            }
            
            # Add individual gate status
            for name, gate_inst in self.agent.gate_handler.gate_instances.items():
                status["gates"][name] = {
                    "connected": gate_inst.is_connected(),
                    "address": gate_inst.address
                }
            
            response.body = json.dumps(status)
            await self.send(response)
            logger.info(f"Gate status sent to {self.requester_jid}")

    class MonitorGatesBehavior(CyclicBehaviour):
        """Behavior to periodically check and reconnect gates"""
        async def on_start(self):
            """Initialize variables on behavior start"""
            self.check_interval = 5  # Check every 5 seconds
            logger.info("Gate monitoring behavior started")
            
        async def run(self):
            """Run one iteration of gate monitoring"""
            if self.agent.mqtt_initialized:
                # Check each gate and respect the reconnection limit
                for name, gate_inst in self.agent.gate_handler.gate_instances.items():
                    if not gate_inst.is_connected() and gate_inst.reconnect_attempts < 3:
                        # Only try to reconnect if we haven't reached the limit
                        await gate_inst.try_reconnect()
                        if gate_inst.is_connected():
                            # If reconnected, restart the sensor notifications
                            callback = functools.partial(self.agent.gate_handler.sensor_callback, name)
                            await gate_inst.start_listening_sensor(callback)
                
                # Update connection status after reconnection attempts
                connected_count = sum(1 for g in self.agent.gate_handler.gate_instances.values() if g.is_connected())
                prev_count = self.agent.gate_handler.connected_gate_count
                
                if connected_count != prev_count:
                    self.agent.gate_handler.connected_gate_count = connected_count
                    self.agent.gate_handler.update_system_status()
                    
            # Wait before next check
            await asyncio.sleep(self.check_interval)


async def run_handler():
    """Run the gate handler with both MQTT and XMPP support"""
    # Get XMPP credentials from environment variables or use defaults
    xmpp_server = os.environ.get("XMPP_SERVER", "prosody")
    xmpp_username = os.environ.get("XMPP_USERNAME", "gate_handler")
    xmpp_password = os.environ.get("XMPP_PASSWORD", "top_secret")
    
    jid = f"{xmpp_username}@{xmpp_server}"
    password = xmpp_password
    
    logger.info(f"Starting Gate Handler Agent with JID: {jid}")
    
    # Create and start the agent
    gate_agent = GateHandlerAgent(jid, password)
    
    try:
        # Start agent and wait for it to be ready
        await gate_agent.start(auto_register=True)
        
        if not gate_agent.is_alive():
            logger.error("Gate handler agent couldn't connect to XMPP server. Check Prosody configuration.")
            await gate_agent.stop()
            return
        
        logger.info("Gate handler agent connected successfully. Running...")
        
        # Keep the program running
        while gate_agent.is_alive():
            await asyncio.sleep(1)
    except Exception as e:
        logger.error(f"Error in gate handler: {str(e)}")
        logger.exception(e)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        # Cleanup
        if hasattr(gate_agent, 'gate_handler'):
            for gate_name, gate_instance in gate_agent.gate_handler.gate_instances.items():
                await gate_instance.stop_listening_sensor()
                await gate_instance.disconnect()
            
            if hasattr(gate_agent.gate_handler, 'mqtt_client'):
                gate_agent.gate_handler.mqtt_client.loop_stop()
                gate_agent.gate_handler.mqtt_client.disconnect()
        
        await gate_agent.stop()
        logger.info("Gate handler agent stopped")


if __name__ == "__main__":
    asyncio.run(run_handler())