import sys

from loguru import logger

# –––––– LOG ONLY EVENTS WITH LEVEL=WARNING OR HIGHER ––––––
logger.remove()
logger.add(sys.stdout, level="WARNING")
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

# –––––– LOG CERTAIN EVENTS TO A FILE INSTEAD OF TERMINAL ––––––
logger.add("my_log.log", level="DEBUG", rotation="100 MB")  # NOTE creates new file every 100MB
# ––––––––––––––––––––––––

logger.debug("This is a debug message (1/5 severity)")
logger.info("This is an info message (2/5 severity)")
logger.warning("This is a warning message (3/5 severity)")
logger.error("This is an error message (4/5 severity)")
logger.critical("This is a critical message (5/5 severity)")
