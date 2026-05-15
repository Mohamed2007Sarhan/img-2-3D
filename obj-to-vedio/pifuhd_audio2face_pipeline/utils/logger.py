import sys
from loguru import logger
from config import LOG_LEVEL
logger.remove()
logger.add(sys.stderr, level=LOG_LEVEL, format='<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{line}</cyan> - {message}', colorize=True)
logger.add('outputs/pipeline.log', level='DEBUG', rotation='10 MB', retention='7 days')

def get_logger(name: str):
    return logger.bind(name=name)
