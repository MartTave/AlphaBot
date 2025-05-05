import asyncio
import logging
import os
import ssl

import aiosasl
import aioxmpp.security_layer
from spade.container import Container

from .scan_command_sender import ScanCommandSender

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Get XMPP credentials from environment variables
XMPP_JID = os.getenv("XMPP_JID", "runner") + "@prosody"
XMPP_PASSWORD = os.getenv("XMPP_PASSWORD")

def create_ssl_context():
    """Create SSL context with our certificate"""
    ctx = ssl.create_default_context()
    ctx.load_verify_locations("/app/certs/prosody.crt")
    return ctx


# Configure global security settings for SPADE
Container.security_layer = aioxmpp.security_layer.SecurityLayer(
    ssl_context_factory=create_ssl_context,
    certificate_verifier_factory=aioxmpp.security_layer.PKIXCertificateVerifier,
    tls_required=True,
    sasl_providers=[
        aiosasl.PLAIN(
            credential_provider=lambda _: (
                XMPP_JID,
                os.getenv("XMPP_PASSWORD"),
            )
        )
    ],
)


async def run_scan_command_sender():
    """Initialize and start the scan command sender agent"""
    logger.info(f"Starting ScanCommandSender with JID: {XMPP_JID}")
    
    # Create and start the agent
    scan_sender = ScanCommandSender(XMPP_JID, XMPP_PASSWORD)
    await scan_sender.start(auto_register=True)
    
    # Check if agent started successfully
    if not scan_sender.is_alive():
        logger.error("Scan command sender agent couldn't connect.")
        await scan_sender.stop()
        return None
    
    logger.info("Scan command sender agent started successfully.")
    return scan_sender


async def main():
    try:
        # Start the scan command sender agent
        scan_sender = await run_scan_command_sender()
        if not scan_sender:
            logger.error("Failed to start scan command sender agent.")
            return
        
        # Wait for the behavior to complete
        while any(behavior.is_running for behavior in scan_sender.behaviours):
            await asyncio.sleep(1)
            
        logger.info("Scan command has been sent.")
        
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt. Shutting down...")
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        if 'scan_sender' in locals() and scan_sender:
            await scan_sender.stop()
        logger.info("Agent stopped.")


if __name__ == "__main__":
    asyncio.run(main())