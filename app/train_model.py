from src.base import logger
from app.preprocess import train_preprocessor
from app.optimization import optimize_regressor
from app.regression import train_regressor

LOGGER = logger.set()

EXTRACT_DATA = False


def train_model():

    LOGGER.info("FUNCTION: train_model")

    # Train the preprocess pipeline
    train_preprocessor(extract_data=EXTRACT_DATA)

    # Otimizar parametros
    optimize_regressor()

    # train regressor model
    train_regressor()


if __name__ == "__main__":
    train_model()
