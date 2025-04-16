# AlphaBot Repository

This repository contains the code and configuration for the AlphaBot project, which includes various agents and services to control and monitor an AlphaBot robot. The project uses Docker to containerize the different components and Prosody for XMPP communication.

## Repository Structure

- `alphabot_agent/`: Contains the code for the AlphaBot agent that controls the robot.
- `calibration_agent/`: Contains the code for the calibration agent.
- `prosody/`: Contains the configuration and modules for the Prosody XMPP server.
- `runner_agent/`: Contains the code for the runner agent that sends commands to the AlphaBot.
- `env/`: Contains environment configuration files.
- `run_remote.sh`: Script to deploy and run the agents on remote Raspberry Pi devices.

## Running the Docker Containers

### Prerequisites

- Docker and Docker Compose installed on your machine.
- Ensure the IP addresses in the `.env` file are correctly set for your network.

### Steps to Run

#### Run the containers

1. **Build and Run the Docker Containers**

   Navigate to the root of the repository and run the following command to build and start the Docker containers:

   ```sh
   docker-compose up --build
   ```

   This will start the following services:
   - `alphabot_agent`: The agent controlling the AlphaBot.
   - `calibration_agent`: The agent responsible for calibrating the AlphaBot.
   - `prosody`: The Prosody XMPP server.
   - `api`: The API server for message monitoring.
   - `dashboard`: The dashboard for monitoring the XMPP messages.

2. **Running on Remote Raspberry Pi**

   Use the `run_remote.sh` script to deploy and run the agents on remote Raspberry Pi devices. The script requires a host parameter (1 or 2) to specify which Raspberry Pi to deploy to.

   ```sh
   ./run_remote.sh <host_number>
   ```

   Replace `<host_number>` with `1` or `2` depending on the target Raspberry Pi.

#### Command runner

Use the command :
```sh
uv run --project ./runner_agent --env-file ./env/prosody.env ./runner_agent/runner.py
```

## Configuration

### Environment Variables

The `.env` file contains the following environment variables:

- `PROSODY_IP`: IP address of the Prosody server.
- `AGENT_1_IP`: IP address of the first Raspberry Pi.
- `AGENT_2_IP`: IP address of the second Raspberry Pi.

### Prosody Configuration

The `prosody/prosody.cfg.lua` file contains the configuration for the Prosody XMPP server, including the enabled modules and SSL/TLS settings.

### Docker Compose

The `docker-compose.yml` file defines the services and their configurations. It includes the following services:

- `alphabot_agent`
- `calibration_agent`
- `prosody`
- `api`
- `dashboard`

## Agents

### AlphaBot Agent

The AlphaBot agent is responsible for controlling the AlphaBot robot. It uses the `RPi.GPIO` library to interact with the hardware and the `spade` library for XMPP communication.

### Calibration Agent

The calibration agent is responsible for calibrating the AlphaBot's sensors and camera. It uses the `camera-sync` library for camera calibration and the `spade` library for XMPP communication.

### Runner Agent

The runner agent sends commands to the AlphaBot agent. It includes various command files in the `commands` directory, which define the instructions to be sent to the AlphaBot.

## API

The API server is built using Bun and provides endpoints for monitoring and controlling the XMPP messages. It includes WebSocket support for real-time message updates.

### Endpoints

- `/api/status`: Returns the status of the API server.
- `/api/ban`: Endpoint to ban an agent.
- `/api/messages`: Endpoint to receive and broadcast XMPP messages.

## Dashboard

The dashboard service provides a web interface for monitoring the XMPP messages. It uses the `dij0s/expose` image.

## Generating Certificates

The `prosody/generate-certs.sh` script generates the necessary SSL/TLS certificates for the Prosody server.

```sh
cd prosody
./generate-certs.sh
```

This will generate the certificates in the `prosody/certs` directory.
