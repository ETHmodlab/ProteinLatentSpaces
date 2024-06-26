import torch
import numpy as np

from dataset import ProteinDataset
from utils import average_structures
import autoencoder as autoencoder

def main():
    
    train_range = range(170, 175)
    val_range = range(175, 176)
    data_path = './data/'
    latent_dim = 40
    epochs = 250
    lr = 0.0001
    p_enc = .8
    p_dec = .5

    train_dataset = make_dataset(train_range, data_path, False)
    val_dataset = make_dataset(val_range, data_path, False)

    model = autoencoder.Autoencoder(p_drop_enc=p_enc, p_drop_dec=p_dec, latent_dim=latent_dim)
    autoencoder.fit(model, train_dataset, val_dataset, epochs)

    torch.save(model.state_dict(), f'{data_path}autoencoder/ld_{latent_dim}_hd_{epochs}_dr_enc_{p_enc}_dr_dec_{p_dec}_lr_{lr}.pt')


def make_dataset(model_ids: list, 
                data_path: str,
                average: bool
                ):
    ''' 
    Create a Tensordataset for the latent vectors of the models in model_ids

    Parameters
    ----------
    model_ids: list[int]
        The model ids of the GTNN models

    data_path: str
        Path to the directory that contains the subdirectory latent

    average: bool
        Whether to average the latent vectors of different structures of each protein

    Returns
    --------
    torch.utils.data.TensorDataset
        A dataset that contains the latent vectors of all models
    '''

    X_all = None

    for model in model_ids:

        dataset = ProteinDataset(model, data_path)
        sids = dataset.structure_ids
        X = dataset.get_structure_latent_features(sids)

        if average:
                
            tids = [dataset._get_tid_from_structure_id(i) for i in sids]
            X, tids = average_structures(X, tids)

        if X_all is None:
            X_all = X
        
        else:
            X_all = np.concatenate((X_all, X), axis=0)

        
    mean = np.array([np.average(X_all[:,i]) for i in range(len(X_all[0]))])
    std = np.array([np.std(X_all[:,i]) for i in range(len(X_all[0]))])

    for i in range(len(X_all)):

        X_all[i] -= mean
        X_all[i] /= std
        
    
    return torch.utils.data.TensorDataset(torch.Tensor(X_all))

            
if __name__ == '__main__':
    main()

