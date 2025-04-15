#!/bin/bash
set -e  # Exit on error

# Ensure we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create certs directory if it doesn't exist
mkdir -p certs
cd certs

echo "Generating CA key and certificate..."
openssl genrsa -out ca.key 4096
openssl req -x509 -new -nodes -key ca.key -sha256 -days 3650 -out ca.crt \
    -subj "/C=CH/ST=Valais/L=Sierre/O=HES-SO/CN=prosody"

echo "Generating server key and CSR..."
openssl genrsa -out prosody.key 2048
openssl req -new -key prosody.key -out prosody.csr \
    -subj "/C=CH/ST=Valais/L=Sierre/O=HES-SO/CN=prosody"

echo "Signing the CSR with our CA..."
openssl x509 -req -in prosody.csr -CA ca.crt -CAkey ca.key -CAcreateserial \
    -out prosody.crt -days 365 -sha256

echo "Setting permissions..."
chmod 644 *.crt *.key

echo "Certificate generation complete!"
ls -l
