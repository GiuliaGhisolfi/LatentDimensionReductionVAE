from src.autoencoders.ae import AutoEncoder
from src.latent_vector_approximator import LatentVectorApproximator
from src.utils.utils import (get_training_and_validation_sets_gray_scale,
                             loss_function)


def main():
    X_train, Y_train, cX_train, X_val, Y_val, cX_val = get_training_and_validation_sets_gray_scale()

    input_dim = X_train[0].shape
    latent_dim = 128
    conditions_dim = cX_train[0].shape[0]

    #model = AutoEncoder(input_dim=input_dim, latent_dim=latent_dim)
    #model.train(X_train, X_val, epochs=2, batch_size=32)

    model = LatentVectorApproximator(input_dim=input_dim, latent_dim=latent_dim, conditions_dim=conditions_dim,
        loss_function=loss_function, metrics=['mse', 'mae'])
    model.train(X_train[:100], cX_train[:100], X_val[:20], cX_val[:20], pre_train_epochs=1, epochs=1, batch_size=50)

if __name__ == '__main__':
    main()