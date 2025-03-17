import logging
import os
from datetime import datetime

# Create 'logs' directory if it doesn't exist
logs_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_dir, exist_ok=True)

# Define the log file name with timestamp
log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
log_file_path = os.path.join(logs_dir, log_file)

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] [%(filename)s:%(lineno)d] [%(levelname)s] - %(message)s",
    level=logging.INFO,
)

# Logging startup message
if __name__ == "__main__":  # <-- Fixed incorrect `main` string
    logging.info("Logging has started successfully!")
