import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import scipy.io
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from Networks.nn import training
from torch.nn import functional as F

# Constants
BATCH_SIZE = 30
X_DIM = 401
HIDDEN_DIM = 200
MID_DIM = 100
LATENT_DIM = 50
LR = 1e-3
EPOCHS = 10
NAME_OF_MODEL_FILE = 'CVAE_Species1.pt'

# Number of generated samples and count for class one
NUM_GENERATED_SAMPLES = 150
CLASS_ONE_COUNT = 75
CLASS_ZERO_COUNT = NUM_GENERATED_SAMPLES - CLASS_ONE_COUNT

def clean_data(x):
    data = x[1:, :]
    scaler = MinMaxScaler(feature_range=(0, 0.99))
    data_scale = scaler.fit_transform(data).transpose()
    labels = x[0, :]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels).reshape(-1, 1)
    X = torch.tensor(data_scale, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    return X, y



if __name__ == '__main__':
    # Load data
    name_of_file = 'Data/randomize2Species.mat'
    name_of_variable = 'randSpecies1'
    toy_data = scipy.io.loadmat(name_of_file)
    data_array = np.asarray(toy_data[name_of_variable]).astype('float32')
    X, y = clean_data(data_array)
    kwargs = {'num_workers': 1, 'pin_memory': True}

    loader = DataLoader(list(zip(X, y)), shuffle=True, batch_size=BATCH_SIZE)
     
    model = training(loader)
    
    # Generate new samples
    my_list = [0] * CLASS_ZERO_COUNT + [1] * CLASS_ONE_COUNT
    np.random.shuffle(my_list)
    
    new_data = torch.tensor(my_list, dtype=torch.int64)
    new_data = F.one_hot(new_data, num_classes=2).float()
    generated_samples = model.decoder(torch.randn(NUM_GENERATED_SAMPLES, LATENT_DIM), new_data)

    array_species1_case2 = generated_samples[:CLASS_ONE_COUNT].detach().numpy()
    array_species2_case2 = generated_samples[CLASS_ONE_COUNT:].detach().numpy()

    case2 = {"Species1": array_species1_case2, "Species2": array_species2_case2}
    scipy.io.savemat("Generated_Data/Case2_Real_Model_" + "NAME_OF_MODEL_FILE.mat", case2)

    # Saving real data
    species1 = np.resize(data_array[1:, data_array[0, :] == 1.0][:, :75], (75, 401))
    species2 = np.resize(data_array[1:, data_array[0, :] == 2.0][:, :75], (75, 401))

    case1 = {"Species1": species1, "Species2": species2}
    scipy.io.savemat("Generated_Data/Case1_Real_Model_" + "NAME_OF_MODEL_FILE.mat", case1)
