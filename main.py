from app.util import get_logger

logger = get_logger(__name__)

def main():
    """
    Main function to run the HaruPyQuant application.
    """
    logger.info("Initializing HaruPyQuant...")
    logger.info("HaruPyQuant application started.")
    # In the future, this will initialize and run the trading bot.
    logger.info("HaruPyQuant application finished.")

if __name__ == "__main__":
    main() 