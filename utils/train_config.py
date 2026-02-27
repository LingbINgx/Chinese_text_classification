from dataclasses import dataclass, fields
from pathlib import Path
from typing import Type, TypeVar, Any
import yaml
from loguru import logger

from .wraps import logger_return

T = TypeVar("T", bound="BaseConfig")

@dataclass
class BaseConfig:
    @staticmethod
    def _lowercase_keys(obj: Any):
        if isinstance(obj, dict):
            return {
                str(k).lower(): BaseConfig._lowercase_keys(v) for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [BaseConfig._lowercase_keys(i) for i in obj]
        else:
            return obj

    @classmethod
    @logger_return
    def from_yaml(cls: Type[T], path: str | Path) -> T:
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        data = cls._lowercase_keys(data)

        valid_fields = {f.name for f in fields(cls)}
        unknown = set(data.keys()) - valid_fields
        if unknown:
            raise ValueError(f"Unknown config fields: {unknown}")

        return cls(**data)

    def to_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}


