import asyncio
import functools
import sys
import logging
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
            logger.info(f"Attempting to reconnect to {self.name}...")
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
        
        # First connection attempt for all gates
        await self.connect_all_gates()

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


async def run_handler():
    handler = MQTTGateHandler()
    await handler.run()


if __name__ == "__main__":
    asyncio.run(run_handler())