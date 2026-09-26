import warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv




class Autoencoder(nn.Module):
    """
    Plain autoencoder with the same interface as GraphAutoencoder.

    This class keeps the method signatures:
        encoder(x, edge_index)
        decoder(x, edge_index)
        forward(x, edge_index)

    Parameters
    ----------
    encoder_dim : list of int
        A list specifying layer dimensions, e.g. [input_dim, hidden1, hidden2, latent_dim].

    activation : str, optional
        Activation function to use between linear layers.
        Supported: {'relu', 'sigmoid', 'tanh', 'leakyrelu'}.
        Default is 'relu'.
    """

    def __init__(self, encoder_dim, activation='relu'):
        super(Autoencoder, self).__init__()

        self._dim = len(encoder_dim) - 1

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()

        # encoder
        for i in range(self._dim):
            self.encoder_layers.append(nn.Linear(encoder_dim[i], encoder_dim[i + 1]))

        # decoder
        decoder_dim = list(reversed(encoder_dim))
        for i in range(self._dim):
            self.decoder_layers.append(nn.Linear(decoder_dim[i], decoder_dim[i + 1]))

        if activation == 'relu':
            self._activation = nn.ReLU()
        elif activation == 'sigmoid':
            self._activation = nn.Sigmoid()
        elif activation == 'tanh':
            self._activation = nn.Tanh()
        elif activation == 'leakyrelu':
            self._activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

    def encoder(self, x, edge_index=None):
        """
        Encode input features into latent embeddings.

        Parameters
        ----------
        x : torch.Tensor
            Input feature matrix of shape (n_samples, in_dim).

        edge_index : torch.Tensor, optional
            Unused. Kept only for compatibility with GraphAutoencoder.

        Returns
        -------
        x : torch.Tensor
            Latent embeddings of shape (n_samples, latent_dim).
        """
        for i in range(self._dim):
            x = self.encoder_layers[i](x)
            if i < self._dim - 1:
                x = self._activation(x)

        x = F.normalize(x, p=2, dim=1)
        return x

    def decoder(self, x, edge_index=None):
        """
        Decode latent embeddings to reconstruct original features.

        Parameters
        ----------
        x : torch.Tensor
            Latent embeddings of shape (n_samples, latent_dim).

        edge_index : torch.Tensor, optional
            Unused. Kept only for compatibility with GraphAutoencoder.

        Returns
        -------
        x : torch.Tensor
            Reconstructed feature matrix of shape (n_samples, in_dim).
        """
        for i in range(self._dim):
            x = self.decoder_layers[i](x)
            if i < self._dim - 1:
                x = self._activation(x)
        return x

    def forward(self, x, edge_index=None):
        """
        Perform full encoder-decoder forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input feature matrix of shape (n_samples, in_dim).

        edge_index : torch.Tensor, optional
            Unused. Kept only for compatibility.

        Returns
        -------
        x_hat : torch.Tensor
            Reconstructed feature matrix of shape (n_samples, in_dim).
        """
        latent = self.encoder(x, edge_index)
        x_hat = self.decoder(latent, edge_index)
        return x_hat


class Prediction_mlp(nn.Module):

    """
    A fully connected multi-layer perceptron (MLP) for cross-modality prediction between latent embeddings.

    Parameters
    ----------
    prediction_dim : list of int
        A list defining the hidden dimensions of the MLP prediction module.
    
    activation : str, optional
        Activation function to use between hidden layers. Must be one of
        `{'relu', 'sigmoid', 'tanh', 'leakyrelu'}`. Default is `'relu'`.

    Methods
    -------
    forward(x)
        Apply the MLP to an input embedding and return predicted embedding.
    """

    def __init__(self,
                 prediction_dim,
                 activation='relu'):

        super(Prediction_mlp, self).__init__()

        self._depth = len(prediction_dim) - 1
        self._activation = activation
        self._prediction_dim = prediction_dim

        encoder_layers = []
        for i in range(self._depth):
            encoder_layers.append(
                nn.Linear(self._prediction_dim[i], self._prediction_dim[i + 1]))
            if i < self._depth - 1:
                if self._activation == 'sigmoid':
                    encoder_layers.append(nn.Sigmoid())
                elif self._activation == 'leakyrelu':
                    encoder_layers.append(nn.LeakyReLU(0.2, inplace=True))
                elif self._activation == 'tanh':
                    encoder_layers.append(nn.Tanh())
                elif self._activation == 'relu':
                    encoder_layers.append(nn.ReLU())
                else:
                    raise ValueError('Unknown activation type %s' % self._activation)
        self._encoder = nn.Sequential(*encoder_layers)


    def forward(self, x):
        """
        Perform full forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input cell embedding.
        
        Returns
        -------
        output : torch.Tensor
            Predicted cell embedding.
        """
        output = self._encoder(x)
        output = F.normalize(output, p=2, dim=1)
        return output

