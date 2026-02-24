from dataclasses import dataclass, fields
from pathlib import Path
from typing import Type, TypeVar, Any
import yaml

T = TypeVar("T", bound="BaseConfig")

class BaseConfig:
    @staticmethod
    def _lowercase_keys(obj: Any):
        if isinstance(obj, dict):
            return {
                str(k).lower(): BaseConfig._lowercase_keys(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [BaseConfig._lowercase_keys(i) for i in obj]
        else:
            return obj

    @classmethod
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


@dataclass
class TrainConfig(BaseConfig):
    train_file_path: str
    val_file_path: str
    test_file_path: str
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    learning_rate: float
    epochs: int
    output_dir: str



@dataclass
class TrainConfig_transformer(TrainConfig):
    model_name: str
    weight_decay: float
    warmup_ratio: float
    dropout_prob: float
    freeze_encoder: bool
    
    
    
    
@dataclass
class TrainConfig_scratch_transformer(TrainConfig):
    weight_decay: float
    warmup_ratio: float
    min_freq: int
    max_vocab_size: int | None
    embed_dim: int
    num_heads: int
    num_layers: int
    ffn_dim: int
    dropout: float





TrainConfig_class_map = {
    "transformer": TrainConfig_transformer,
    "scratch_transformer": TrainConfig_scratch_transformer,
    
}
    
def get_config(config_path: str | Path, model_name: str):
     config_class = TrainConfig_class_map.get(model_name)
     if not config_class:
         raise ValueError(f"Unknown model name: {model_name}")
     return config_class.from_yaml(config_path)


if __name__ == "__main__":
    config = get_config("../params_scratch.yaml", "scratch_transformer")
    print(config)
    print(config.learning_rate)