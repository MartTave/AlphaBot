#!/bin/bash

# Sourcing default env file
. ./.env

USER="raspi"

# IP address of the Prosody server

# Local agent folder path
LOCAL_AGENT_PATH="./agent"
# Remote path where to copy the agent folder
REMOTE_PATH="/home/raspi/app"

# Check if host parameter is provided
if [ $# -ne 1 ]; then
    echo "Error: Host parameter required (1 or 2)"
    echo "Usage: $0 <host_number>"
    exit 1
fi

# Validate host parameter
if [ "$1" != "1" ] && [ "$1" != "2" ]; then
    echo "Error: Host parameter must be 1 or 2"
    echo "Usage: $0 <host_number>"
    exit 1
fi

if [ "$1" = "1" ]; then
    #HOST="192.168.237.51"
    AGENT_NAME="alpha-pi-4b-agent-1"
    HOST=$AGENT_1_IP
else
    #HOST="192.168.237.52"
    AGENT_NAME="alpha-pi-4b-agent-2"
    HOST=$AGENT_2_IP
fi

# Copy changes into the remote server
rsync -avz --progress --exclude '.venv' --exclude '__pycache__' --exclude 'received_photos' --exclude 'node_modules' --exclude '.git' ./ ${USER}@${HOST}:${REMOTE_PATH}

# Then connect and restart docker compose
echo "Starting the stack in attached mode..."
ssh -t ${USER}@${HOST} "sudo systemctl stop alphabot.service;export AGENT_NAME=${AGENT_NAME};cd /home/raspi/app && docker compose down && docker compose up alphabot_agent"

echo "Docker-compose stopped.."
