import os
from pathlib import Path 
from pydantic import validate_call
from dotenv import load_dotenv

from jwstnoobfriend.utils.log import getLogger


logger = getLogger(__name__)

def find_project_root() -> Path:
    """Find the project root directory"""
    markers = ['.git', 'pyproject.toml', '.env']
    current_dir = Path.cwd().absolute()
        
    while current_dir != current_dir.parent:
        if any((current_dir / marker).exists() for marker in markers):
            return current_dir
        current_dir = current_dir.parent

    return current_dir

def find_env_file() -> Path:
    """Find the .env file, prioritizing the current directory over the project root."""
    project_root = find_project_root()
    current_dir = Path.cwd().absolute()
    
    env_root = project_root / '.env'
    env_current = current_dir / '.env'
    if env_current.exists():
        return env_current
    elif env_root.exists():
        return env_root
    else:
        logger.warning("No .env file found in the current directory or project root.")
        return None
        
def load_environment():
    """Load environment variables from the .env file."""
    env_file = find_env_file()
    if env_file:
        load_dotenv(env_file)
        logger.info(f"Loaded environment variables from {env_file}")
    else:
        logger.warning("No .env file found to load environment variables.")
