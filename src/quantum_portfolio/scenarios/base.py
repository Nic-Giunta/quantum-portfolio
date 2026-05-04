from abc import ABC, abstractmethod


class Scenario(ABC):
    @abstractmethod
    def apply(self, returns): ...
