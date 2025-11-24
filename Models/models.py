from abc import ABC, abstractmethod

class Model(ABC):

    @abstractmethod
    def price(self, **kwargs):
        pass

    