import torch
from torch import nn


class XRDCNN(nn.Module):
    """
    A small 1D convolutional neural network for XRD classification.

    The model receives an XRD pattern as a sequence of intensity values.

    Input shape:
        (batch_size, 1, n_points)

    Why 1 channel?
        Each pattern has one intensity trace, unlike an RGB image which has
        three colour channels.

    Output:
        one score per crystal-system class

    The highest scoring class becomes the model prediction.
    """

    def __init__(
        self,
        n_points: int,
        n_classes: int,
        channels: list[int],
        kernel_size: int,
        dropout: bool,
        dropout_p: float,
    ):
        super().__init__()

        # ------------------------------------------------------------
        # Feature extractor
        # ------------------------------------------------------------
        # This part scans across the XRD pattern and learns local peak
        # motifs. A convolutional filter is a small sliding window over
        # neighbouring 2θ positions.

        feature_layers = []
        in_channels = 1

        for layer_number, out_channels in enumerate(channels, start=1):
            # Conv1d:
            # learns local filters over the diffraction pattern.
            feature_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )

            # Optional normalisation experiment:
            # BatchNorm can stabilise training by normalising activations
            # inside each convolutional layer.
            #
            # Try uncommenting this line and comparing the loss curve.
            # feature_layers.append(nn.BatchNorm1d(out_channels))

            # Activation function:
            # ReLU introduces nonlinearity.
            feature_layers.append(nn.ReLU())

            # Alternative activation experiments:
            # Replace ReLU above with one of these and compare training.
            #
            # feature_layers.append(nn.LeakyReLU())
            # feature_layers.append(nn.ELU())
            # feature_layers.append(nn.Tanh())

            # Pooling:
            # MaxPool keeps strong local responses and shortens the signal.
            # This reduces computation and encourages the model to focus on
            # prominent diffraction features.
            feature_layers.append(nn.MaxPool1d(2))

            # Alternative pooling experiments:
            #
            # Average pooling keeps smoother local information instead of
            # only the strongest response.
            # feature_layers.append(nn.AvgPool1d(2))
            #
            # Removing pooling keeps more spatial detail but increases the
            # flattened classifier size and may overfit more easily.
            # Comment out the pooling layer above to test this.

            in_channels = out_channels

        self.features = nn.Sequential(*feature_layers)

        # ------------------------------------------------------------
        # Work out classifier input size automatically
        # ------------------------------------------------------------
        # After convolutions and pooling, the pattern is shorter and has
        # more channels. This dummy pass avoids manually calculating the
        # flattened size.

        with torch.no_grad():
            dummy_pattern = torch.zeros(1, 1, n_points)
            flattened_size = self.features(dummy_pattern).reshape(1, -1).shape[1]

        # ------------------------------------------------------------
        # Classifier
        # ------------------------------------------------------------
        # The feature extractor produces learned XRD features.
        # The classifier maps those features to crystal-system scores.

        classifier_layers = [
            nn.Linear(flattened_size, 32),
            nn.ReLU(),
        ]

        # Optional classifier normalisation experiment:
        # classifier_layers.insert(1, nn.BatchNorm1d(32))

        if dropout:
            # Dropout randomly switches off some hidden values during
            # training. This can reduce overfitting on small datasets.
            classifier_layers.append(nn.Dropout(dropout_p))

        classifier_layers.append(nn.Linear(32, n_classes))

        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x):
        # x starts as:
        #   (batch_size, 1, n_points)

        x = self.features(x)

        # Flatten learned feature maps into one vector per sample:
        #   (batch_size, channels, reduced_points)
        #        -> (batch_size, flattened_size)
        x = x.reshape(x.shape[0], -1)

        # Return raw class scores.
        # CrossEntropyLoss will handle converting these scores internally.
        return self.classifier(x)
