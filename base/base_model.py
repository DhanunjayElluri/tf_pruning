from abc import ABC, abstractmethod
from utils.config import Config


class BaseModel(ABC):
    def __init__(self, config, dataset):
        self.config = Config.from_json(config)
        self.dataset = dataset

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def compile_model(self):
        raise NotImplementedError

    @abstractmethod
    def fit_model(self):
        raise NotImplementedError

    @abstractmethod
    def evaluate_and_save_model(self):
        raise NotImplementedError
