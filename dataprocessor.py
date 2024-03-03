import torch
from abc import ABCMeta, abstractmethod

class DataProcessor(torch.nn.Module, metaclass=ABCMeta):
    def __init__(self):
        """DataProcessor exposes functionality for pre-
        and post-processing data during training or inference.

        To be a valid DataProcessor within the Trainer requires
        that the following methods are implemented:

        - to(device): load necessary information to device, in keeping
            with PyTorch convention
        - preprocess(data): processes data from a new batch before being
            put through a model's forward pass
        - postprocess(out): processes the outputs of a model's forward pass
            before loss and backward pass
        - wrap(self, model):
            wraps a model in preprocess and postprocess steps to create one forward pass
        - forward(self, x):
            forward pass providing that a model has been wrapped
        """
        super().__init__()

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess(self, x):
        pass

    @abstractmethod
    def postprocess(self, x):
        pass

    @abstractmethod
    def wrap(self, model):
        pass

    @abstractmethod
    def forward(self, x):
        pass

class DefaultDataProcessor(DataProcessor):
    def __init__(
        self, in_normalizer=None, out_normalizer=None, positional_encoding=None
    ):
        """A simple processor to pre/post process data before training/inferencing a model.

        Parameters
        ----------
        in_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the input samples
        out_normalizer : Transform, optional, default is None
            normalizer (e.g. StandardScaler) for the target and predicted samples
        positional_encoding : Processor, optional, default is None
            class that appends a positional encoding to the input
        """
        super().__init__()
        self.in_normalizer = in_normalizer
        self.out_normalizer = out_normalizer
        self.positional_encoding = positional_encoding
        self.device = "cpu"

    def wrap(self, model):
        self.model = model
        return self

    def to(self, device):
        if self.in_normalizer is not None:
            self.in_normalizer = self.in_normalizer.to(device)
        if self.out_normalizer is not None:
            self.out_normalizer = self.out_normalizer.to(device)
        self.device = device
        return self

    def preprocess(self, data_dict, batched=True):
        x = data_dict["x"].to(self.device)
        y = data_dict["y"].to(self.device)

        if self.in_normalizer is not None:
            x = self.in_normalizer.transform(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x, batched=batched)
        if self.out_normalizer is not None and self.train:
            y = self.out_normalizer.transform(y)

        data_dict["x"] = x
        data_dict["y"] = y

        return data_dict

    def postprocess(self, output, data_dict):
        y = data_dict["y"]
        if self.out_normalizer and not self.train:
            output = self.out_normalizer.inverse_transform(output)
            y = self.out_normalizer.inverse_transform(y)
        data_dict["y"] = y
        return output, data_dict

    def forward(self, **data_dict):
        data_dict = self.preprocess(data_dict)
        output = self.model(data_dict["x"])
        output = self.postprocess(output)
        return output, data_dict
