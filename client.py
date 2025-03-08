import asyncio
import logging
from pathlib import Path
import sys

# Configure logging
logs_dir = Path('logs')
logs_dir.mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(logs_dir / 'client.log')
    ]
)
logger = logging.getLogger(__name__)

from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

from models.mcp_client import MCPClient

async def main():
    if len(sys.argv) < 2:
        logger.error("Usage: python client.py <path_to_server_script>")
        sys.exit(1)
        
    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())