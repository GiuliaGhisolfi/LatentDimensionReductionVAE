from sklearn.model_selection import train_test_split

from autoencoders.ae import AutoEncoder
from autoencoders.vae import VariationalAutoEncoder
from utils.dataloader import Dataset

path = 'data/'
scenario = 'no_obj'
RANDOM_SEED = 42

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

def main():
    # load_data
    feature_set = Dataset(path, condition=scenario)
    X_train, Y_train, cX_train = feature_set.get_training_set()

    # split data
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=RANDOM_SEED)

    # convert tensor to numpy
    X_train, X_val, Y_train, Y_val = tensor_to_numpy(X_train
        ), tensor_to_numpy(X_val), tensor_to_numpy(Y_train), tensor_to_numpy(Y_val)

    # build model
    ae = VariationalAutoEncoder(input_dim=X_train[0].shape, latent_dim=2)
    
    # train model
    ae.train(X_train, X_val, epochs=2, batch_size=32)

if __name__ == '__main__':
    main()
