from preprocess import train_preprocessor
from optimization import optimize_regressor
from regression import train_regressor


def train_model():
    
    # Train the preprocess pipeline
    train_preprocessor(extract_data=True)

    # Otimizar parametros
    optimize_regressor()

    # train regressor model
    train_regressor()

if __name__ == "__main__":
    train_model()