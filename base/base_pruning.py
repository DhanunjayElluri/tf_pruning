from abc import ABC, abstractmethod
from utils.config import Config


class BasePruning(ABC):
    def __init__(self, config, dataset):
        self.config = Config.from_json(config)
        self.dataset = dataset

    @abstractmethod
    def set_params(self):
        raise NotImplementedError

    @abstractmethod
    def compile_prune_model(self):
        raise NotImplementedError

    @abstractmethod
    def fit_prune_model(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate_and_save_model(self):
        raise NotImplementedError

    @abstractmethod
    def quantization(self):
        raise NotImplementedError