from torch import nn


class MLPRegressor(nn.Module):
    """
    A small multilayer perceptron for molecular property regression.

    Input:
        one row of molecular descriptors

    Output:
        one predicted solubility value

    Example architecture when hidden_dims = [32, 16]:

        descriptors -> Linear(3, 32) -> ReLU
                    -> Linear(32, 16) -> ReLU
                    -> Linear(16, 1)
                    -> predicted solubility
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        dropout: bool,
        dropout_p: float,
    ):
        super().__init__()

        layers = []
        current_dim = input_dim

        for hidden_dim in hidden_dims:
            # Linear layer:
            # learns weighted combinations of the previous layer's values.
            layers.append(nn.Linear(current_dim, hidden_dim))

            # Optional normalisation experiment:
            # BatchNorm can sometimes stabilise training by normalising
            # hidden activations inside the network.
            #
            # Try uncommenting this line and comparing the loss curve.
            # layers.append(nn.BatchNorm1d(hidden_dim))

            # Activation function:
            # ReLU introduces nonlinearity. Without activations, stacked
            # linear layers would still behave like one linear model.
            layers.append(nn.ReLU())

            # Alternative activation experiments:
            # Replace ReLU above with one of these and compare training.
            #
            # layers.append(nn.Tanh())
            # layers.append(nn.LeakyReLU())
            # layers.append(nn.ELU())

            # Dropout:
            # randomly switches off hidden units during training.
            # This can reduce overfitting by making the network less reliant
            # on any single pathway through the model.
            if dropout:
                layers.append(nn.Dropout(dropout_p))

            current_dim = hidden_dim

        # Final layer:
        # maps the last hidden representation to one regression output.
        layers.append(nn.Linear(current_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # squeeze(-1) changes shape from (batch_size, 1) to (batch_size,)
        # so it matches the target values used by the loss function.
        return self.net(x).squeeze(-1)