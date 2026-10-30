from abc import ABC, abstractmethod

class AbstractModel(ABC):
    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path, device):
        pass