import asyncio
import functools
from datetime import datetime

import paho.mqtt.client as mqtt
from bleak import BleakClient, BleakScanner


class gate:
    # These should be fixed for every gate. If it doesn't work, do "a ble-scan -d [address of the device]" to get the real ones
    BUTTON_UUID = "794f1fe3-9be8-4875-83ba-731e1037a881"
    LED_CHAR_UUID = "794f1fe3-9be8-4875-83ba-731e1037a882"
    SENSOR_UUID = "794f1fe3-9be8-4875-83ba-731e1037a883"

    def __init__(self, address):
        self.address = address

    async def connect(self):
        self.device = await BleakScanner.find_device_by_address(self.address)
        if not self.device:
            print("Device not found!")
            return
        else:
            print(f"Found device {self.address}!")
            self.client = BleakClient(self.device)
            print(f"Connecting to {self.address}...")
            await self.client.connect()
            print("Connexion successful.")

    async def disconnect(self):
        await self.client.disconnect()

    async def light_led(self, rgb):
        """
        Lights the led to a specific color given with its rgb value.

        Parameters
        ----------
                        rgb (list[int]): The RGB value of the color we want on the led, format: [r,g,b] - ex: [255,255,255] for white

        """
        message = bytes(rgb)
        await self.client.write_gatt_char(self.LED_CHAR_UUID, message)

    async def light_led_green(self):
        await self.light_led([0, 255, 0])

    async def shut_led(self):
        await self.light_led([0, 0, 0])

    async def start_listening_sensor(self, func):
        await self.client.start_notify(self.SENSOR_UUID, func)

    async def stop_listening_sensor(self):
        await self.client.stop_notify(self.SENSOR_UUID)


class MQTTGateHandler:
    def __init__(self, broker_address="localhost", broker_port=1883):
        # MQTT setup
        self.mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.broker_address = broker_address
        self.broker_port = broker_port

        # Create event loop for callbacks
        self.loop = asyncio.get_event_loop()

        # Gates setup
        self.gates = {
            "gate1": "D0:31:D0:79:F2:F3",
            "gate2": "F9:CB:B8:48:B5:92",
            # Add gate3 address when you have it
        }
        self.gate_instances = {}

    def on_connect(self, client, userdata, flags, reason_code, properties):
        print(f"Connected to MQTT broker with result code {reason_code}")
        # You can subscribe to topics here if needed
        # self.mqtt_client.subscribe("some/control/topic")

    def on_message(self, client, userdata, msg):
        # Handle incoming MQTT messages if needed
        print(f"Received message on topic {msg.topic}: {msg.payload.decode()}")

    def sensor_callback(self, gate_name, sender, data):
        """Non-async callback that schedules the async handler"""
        asyncio.create_task(self.sensor_handler(gate_name, sender, data))

    async def sensor_handler(self, gate_name, sender, data):
        """Handler for gate sensor data that publishes to MQTT"""
        # Convert the hex data to a meaningful value
        # Assuming the sensor data indicates a passage when it's non-zero
        if any(data):  # Check if any byte in data is non-zero
            timestamp = datetime.now().isoformat()
            message = {
                "gate": gate_name,
                "timestamp": timestamp,
                "event": "passage_detected",
            }
            # Publish to MQTT topic specific to this gate
            self.mqtt_client.publish(f"gates/{gate_name}", str(message))
            print(f"Published passage event for {gate_name}")

            # Light up the LED briefly to indicate detection
            gate = self.gate_instances[gate_name]
            await gate.light_led_green()
            await asyncio.sleep(0.5)
            await gate.shut_led()

    async def setup_gates(self):
        """Initialize and connect to all gates"""
        for gate_name, address in self.gates.items():
            gate_instance = gate(address)
            await gate_instance.connect()
            self.gate_instances[gate_name] = gate_instance

            # Create a callback that properly handles the async nature of sensor_handler
            callback = functools.partial(self.sensor_callback, gate_name)
            await gate_instance.start_listening_sensor(callback)
            print(f"Setup complete for {gate_name}")

    async def run(self):
        """Main run loop"""
        try:
            # Connect to MQTT broker
            print("Connecting to MQTT broker...")
            self.mqtt_client.connect(self.broker_address, self.broker_port, 60)
            # Start MQTT loop in a non-blocking way
            self.mqtt_client.loop_start()

            # Setup BLE gates
            print("Setting up BLE gates...")
            await self.setup_gates()

            # Keep the program running
            while True:
                await asyncio.sleep(1)

        except KeyboardInterrupt:
            print("Shutting down...")
        finally:
            # Cleanup
            for gate_name, gate_instance in self.gate_instances.items():
                await gate_instance.stop_listening_sensor()
                await gate_instance.disconnect()
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()


async def main():
    handler = MQTTGateHandler()
    await handler.run()


if __name__ == "__main__":
    asyncio.run(main())
