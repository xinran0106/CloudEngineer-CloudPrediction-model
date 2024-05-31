import logging
import sys
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

def get_data(url: str, attempts: int = 4, wait: int = 3, wait_multiple: int = 2) -> bytes:
    """Acquires data from URL

    Args:
        url: The URL from which data is to be acquired
        attempts: The number of attempts to make to acquire the data
        wait: The initial wait time between attempts
        wait_multiple: The multiple by which the wait time increases with each attempt

    Returns:
        The bytes representation of the data acquired
    """
    for attempt in range(attempts):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()  # Raise an exception if the response contains an HTTP error status code.
            return response.content
        except requests.exceptions.RequestException as e:
            if attempt < attempts - 1:
                wait_time = wait * (wait_multiple ** attempt)
                print(f"Download failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(f"Download failed after {attempts} attempts. Error: {e}")
    return None

def write_data(data: bytes, save_path: Path) -> None:
    """Writes data to specified file path

    Args:
        data: The bytes representation of the data to be written
        save_path: The local file path to which the data is to be written
    """
    try:
        with open(save_path, "wb") as file:
            file.write(data)
        print(f"Data written successfully to {save_path}")
    except Exception as e:  # pylint: disable=broad-except
        print(f"An error occurred while writing data to {save_path}: {e}")

def acquire_data(url: str, save_path: Path) -> None:
    """Acquires data from specified URL and saves it to a local file path

    Args:
        url: The URL from which data is to be acquired
        save_path: The local file path to which the acquired data is to be saved
    """
    url_contents = get_data(url)
    if url_contents is not None:
        try:
            write_data(url_contents, save_path)
            logger.info("Data written to %s", save_path)
        except FileNotFoundError:
            logger.error("Please provide a valid file location to save dataset to.")
            sys.exit(1)
        except IOError as e:
            logger.error("Error occurred while trying to write dataset to file: %s", e)
            sys.exit(1)
    else:
        logger.error("Failed to download data from the URL.")
        sys.exit(1)
