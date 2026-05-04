from abc import ABC, abstractmethod


class BasePlugin(ABC):
    name: str
    @abstractmethod
    def register(self, registry): ...
