import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(1)
np.random.seed(1)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Autoencoder(nn.Module):
    '''
    A simple symmetric autoencoder with 3 hidden layers
    '''

    def __init__(self,
                p_drop_enc: float = .8,
                p_drop_dec: float = .5,
                input_dim: int = 512,
                hidden_dim: int = 250,
                latent_dim: int = 40,
                encode_only: bool = False
                ):
        ''' 
        Parameters
        ----------
        p_drop_enc: float
            The dropout rate of the encoder hidden layer, default = .8
        
        p_drop_dec: float
            The dropout rate of the decoder hidden layer, default = .5

        input_dim: int
            Feature dimension of the input, default = 512        
        
        hidden_dim: int
            Hidden layer size, default = 250

        latent_dim: int
            Number of latent features, default = 40

        encode_only: bool
            Use when instantiating the encoder from trained weights
        '''

        super().__init__()

        self.encode_only = encode_only

        self.norm1 = nn.LayerNorm(latent_dim)

        self.encoder = nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.Dropout(p_drop_enc),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, latent_dim)
            
        )

        self.decoder = nn.Sequential(
            torch.nn.Linear(latent_dim, hidden_dim),
            torch.nn.Dropout(p_drop_dec),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim)
        )

        self.apply(self._init_weights)


    def forward(self, x):

        x = self.encoder(x)
        x = self.norm1(x)

        if not self.encode_only:
            x = self.decoder(x)
            
        return x
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.bias is not None:
                module.bias.data.zero_()



def fit(model: Autoencoder, 
        dataset: torch.utils.data.TensorDataset,
        val_dataset: torch.utils.data.TensorDataset,
        epochs: int,
        lr: float = 1e-4,
        visualize: bool = False
        ):
    ''' 
    Train a model on the data in dataset

    Parameters
    -----------
    model: Autoencoder
        The model to train

    dataset: torch.utils.data.TensorDataset
        The training data

    val_dataset: torch.utils.data.TensorDataset
        The validation data

    epochs: int
        Number of training epochs

    lr: float
        Learning rate, default = 1e-4

    visualize: bool
        Whether to create a loss development plot, default = False

    Returns
    --------
    float, float
        Training and validation loss in the last epoch
    '''

    model = model.to(device)

    metric_function = torch.nn.MSELoss(reduction = 'mean')
    loss_function = torch.nn.MSELoss(reduction = 'mean')

    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = 1e-10)
    
    loader = torch.utils.data.DataLoader(dataset = dataset,
                                     batch_size = 64,
                                     shuffle = True)
    
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                     batch_size = 64,
                                     shuffle = False)
    
    losses = []
    val_metric = []
    metric = []

    for epoch in tqdm(range(epochs)):

        metric_sum = 0
        val_metric_sum = 0
        total_loss = 0

        for sample in loader:
            
            model.train()
            optimizer.zero_grad()

            sample = sample[0].to(device)

            reconstructed = model(sample)
            loss = loss_function(reconstructed, sample)

            total_loss += loss.item()
       

            loss.backward()
            optimizer.step()

            model.eval()

            with torch.no_grad():

                reconstructed = model(sample)
                metric_sum += metric_function(reconstructed, sample).item()

      
        model.eval()

        with torch.no_grad():
          for sample in val_loader:

              sample = sample[0].to(device)

              reconstructed = model(sample)

              val_metric_sum += metric_function(reconstructed, sample).item()
      
      
      
      
        metric.append(metric_sum/len(loader))
        val_metric.append(val_metric_sum/len(val_loader))
        losses.append(total_loss/len(loader))

    if visualize:
        # create a plot of the training and validation loss
        plt.xlabel('Epoch')
        plt.ylabel('Error')
        x = range(1,epochs+1)
    
        plt.plot(x, losses, color='blue', linestyle='dashed', label= 'training loss')
        plt.plot(x, metric, color='blue', label = 'training error')
        plt.plot(x, val_metric, color='orange', label='validation error')
        plt.ylim(0.5,1.5)
        plt.legend()
        plt.savefig('loss_plot.png', dpi=450)
        plt.close()

    return metric[-1], val_metric[-1]


def predict(model: Autoencoder, dataset: torch.utils.data.TensorDataset):
    '''
    Make predictions on dataset using the given model

    Parameters
    ----------
    model: Autoencoder
        The trained model to use for prediction

    dataset: torch.utils.data.TensorDataset
        The data to make predictions for

    Returns
    -------
    outputs: np.ndarray
        The predicted values for each sample in dataset
    '''
    loader = torch.utils.data.DataLoader(dataset = dataset,
                                    batch_size = 64,
                                    shuffle = False)
    
    model.to(device)
    outputs = None

    with torch.no_grad():

        for sample in loader:

            sample = sample[0].to(device)

            if outputs is None:
                outputs = model(sample).cpu().numpy()

            else:
                outputs = np.concatenate([outputs, model(sample).cpu().numpy()], axis = 0)
    
    
    return outputs
    
    

    
