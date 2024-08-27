import pandas as pd
from sklearn.model_selection import GridSearchCV, ParameterGrid

from src.autoencoders.vae import VariationalAutoEncoder
from src.utils.utils import (get_training_and_validation_sets_gray_scale,
                             loss_function)

params_grid = ParameterGrid({
    'input_dim': [[1, 64, 64]],
    'latent_dim': [1280],
    'n_filters': [[3, 5, 5, 3]],
    'kernel_size': [[32, 16, 3, 1]],
    'stride': [[2, 2, 1, 1]],
    'padding': [['same', 'same', 'same', 'same']],
    'hidden_dims': [[2560], []], # 
    'activation': ['relu'],
    'learning_rate': [0.002], #0.001, 0.005, 0.003, 0.0005, 0.01
    'dropout': [0.3],
    'alpha': [0.3, 0.2],  # LeakyReLU alpha
    'loss_function': [loss_function],
    'patience': [10],
    'metrics': [['mse', 'mae']]
})

params_grid2 = ParameterGrid({
    'input_dim': [[1, 64, 64]],
    'latent_dim': [1280],
    'n_filters': [[4, 4]],
    'kernel_size': [[64, 16]],
    'stride': [[2, 2]],
    'padding': [['same', 'same']],
    'hidden_dims': [[2560], []],
    'activation': ['relu'],
    'learning_rate': [0.002], #0.001, 0.005, 0.003, 0.0005, 0.01
    'dropout': [0.3],
    'alpha': [0.3, 0.2],  # LeakyReLU alpha
    'loss_function': [loss_function],
    'patience': [10],
    'metrics': [['mse', 'mae']]
})


def grid_search_vae(params, csv_file='grid_search/grid_search_vae_2.csv'):
    X_train, Y_train, cX_train, X_val, Y_val, cX_val = get_training_and_validation_sets_gray_scale()

    results = pd.DataFrame(columns=['hidden_dims', 'learning_rate', 'dropout', 'alpha', 'train_loss',
        'val_loss', 'train_mse', 'val_mse', 'train_mae', 'val_mae'])

    for i, param in enumerate(params):
        print(f'Iteration {i+1}/{len(params)}')
        model = VariationalAutoEncoder(**param)
        model.train(X_train, X_val, epochs=100, batch_size=32)

        new_row = {
            'hidden_dims': param['hidden_dims'],
            'learning_rate': param['learning_rate'],
            'dropout': param['dropout'],
            'alpha': param['alpha'],
            'train_loss': [model.ae.history.history['loss']],
            'val_loss': [model.ae.history.history['val_loss']],
            'train_mse': [model.ae.history.history['mse']],
            'val_mse': [model.ae.history.history['val_mse']],
            'train_mae': [model.ae.history.history['mae']],
            'val_mae': [model.ae.history.history['val_mae']]
        }

        results = pd.concat([results, pd.DataFrame(new_row, index=[0])], ignore_index=True)
        results.to_csv(csv_file, index=False)


if __name__ == '__main__':
    grid_search_vae(params_grid2)