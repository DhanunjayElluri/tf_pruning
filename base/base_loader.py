from abc import ABC, abstractmethod
from utils.config import Config


class DataLoader(ABC):
    def __init__(self, config):
        self.config = Config.from_json(config)

    @abstractmethod
    def load_data(self):
        raise NotImplementedError

    @abstractmethod
    def _preprocess_data(self, datapoint, num_classes, image_size):
        raise NotImplementedError
