from abc import ABC, abstractmethod

class Artist(ABC):

    @abstractmethod
    def draw(self, axes):
        pass

