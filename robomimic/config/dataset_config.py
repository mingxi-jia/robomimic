import itertools
from enum import Enum
from importlib import resources
from importlib.resources import Package
from types import SimpleNamespace
from typing import Dict, List, Union

import yaml
from toolz.dicttoolz import assoc_in


class Objective(Enum):
    DATASET_GENERATION = "dataset_generation"

class BiDirectionalEnum(Enum):
    @classmethod
    def key_from_value(cls, value):
        return next((e for e in cls if e.value == value))
    def __repr__(self): return self.value

# class DatasetGenerationConfigSchema:
#     dataset_dir: str
#     output_dir: str
#     dataset_file: str # can be within subdirectories. Using this to generate output file name
#     num_cameras: int
#     num_workers: int
#     num_demos: int
#     env_xml_filename: str

def load_config(objective: str, config_name: str, overrides: List[Dict]= []) -> SimpleNamespace:
    def _config_dir(objective: str) -> Package:
        import robomimic.config.dataset as dataset_config_dir
        if objective == Objective.DATASET_GENERATION:
            return dataset_config_dir
        else:
            raise ValueError(f"Invalid config objective: {objective}")
        
    def _load_config_yaml(config_dir: Package, config_name: str) -> dict:
        config_filename = f"{config_name.split('.')[0]}.yml"
        with resources.open_text(config_dir, config_filename) as f:
            return yaml.safe_load(f)
    
    config_dir = _config_dir(objective)
    config_data = _load_config_yaml(config_dir, config_name)
    config_data = updated_config(config_data, overrides)
    return SimpleNamespace(**config_data)

"""
Stolen from https://github.com/sami-bg
"""
def parse_config_overrides(overrides: List[str]) -> List[Dict]:
    def every_other_element(seq):
        return itertools.islice(seq, 0, None, 2)   
    
    def _parse_value(val: str) -> Union[int, str, None]:
        if val.lower() in {"null", "none"}: return None
        try:                                return int(val)
        except ValueError:                  return str(val)
    
    return [
        assoc_in({}, key.split("."), _parse_value(val))
        for key, val in zip(
            every_other_element(overrides),
            every_other_element(overrides[1:])
        )
    ]

"""
Stolen from https://github.com/sami-bg
"""
def updated_config(cfg: dict, overrides: List[Dict]) -> dict:
    def nested_dict_iter(d, parent_keys=None):
        parent_keys = parent_keys or []
        for k, v in d.items():
            current_path = parent_keys + [k]
            if isinstance(v, dict):
                yield from nested_dict_iter(v, current_path)
            else:
                yield current_path, v

    for override in overrides:
        for path, value in nested_dict_iter(override):
            cfg = assoc_in(cfg, path, value)
    
    return cfg