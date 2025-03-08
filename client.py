import asyncio
import logging
from pathlib import Path

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
    client = MCPClient()
    try:
        await client.connect_to_server()
        await client.chat_loop()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        await client.cleanup()

if __name__ == "__main__":
    asyncio.run(main())