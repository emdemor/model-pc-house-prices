import logging


def set():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("logs.txt"), logging.StreamHandler()],
    )

    return logging.getLogger()
