import torch
from abc import ABC, abstractmethod
from typing import Any, Dict

from typing_extensions import Self


class BaseModel(ABC, torch.nn.Module):
    def __init__(self, num_classes, **kwargs):
        super().__init__()
        self.init_params = dict(num_classes=num_classes) | kwargs
    
    def to_config(self):
        return {"type": self.__class__} | self.init_params
    
    @classmethod
    @abstractmethod
    def load_from_config(cls, config: Dict[str, Any]) -> Self:
        return cls(**config)