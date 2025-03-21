import yaml
from pydantic import BaseModel, Field, field_validator
from enum import Enum
from typing import List, Optional, Dict, Any, ClassVar

# Custom StrEnum implementation for Python < 3.11
class StrEnum(str, Enum):
    def __str__(self) -> str:
        return self.value
    
    def __repr__(self) -> str:
        return str(self)

class DATAMODE(StrEnum):
    XRAY = "xray"
    CRYOEM = "cryoem"

# Path and file configuration
class PathConfig(BaseModel):
    path: str = ""
    file_root: str = ""
    template_pdb: Optional[str] = None
    input_msa: Optional[str] = None
    sub_msa_path: Optional[str] = None
    sub_delmat_path: Optional[str] = None
    msa_feat_init_path: Optional[str] = None
    starting_bias: Optional[str] = None
    starting_weights: Optional[str] = None
    uuid_hex: Optional[str] = None

# Hardware and execution configuration
class ExecutionConfig(BaseModel):
    cuda_device: int = 0
    num_of_runs: int = 1
    verbose: bool = False

# Optimization parameters
class OptimizationParams(BaseModel):
    additive_learning_rate: float = 0.05
    multiplicative_learning_rate: float = 1.0
    weight_decay: Optional[float] = 0.0001
    batch_sub_ratio: float = 0.7
    number_of_batches: int = 1
    rbr_opt_algorithm: str = "lbfgs"
    rbr_lbfgs_learning_rate: float = 150.0
    smooth_stage_epochs: Optional[int] = 50
    phase2_final_lr: float = 1e-3
    l2_weight: float = 1e-7

# Feature flags
class FeatureFlags(BaseModel):
    solvent: bool = True
    sfc_scale: bool = True
    refine_sigmaA: bool = True
    additional_chain: bool = False
    bias_from_fullmsa: bool = False
    chimera_profile: bool = False

# Algorithm parameters
class AlgorithmConfig(BaseModel):
    bias_version: int = 3
    iterations: int = 100
    init_recycling: int = 1
    domain_segs: Optional[List[int]] = None
    optimization: OptimizationParams = Field(default_factory=OptimizationParams)
    features: FeatureFlags = Field(default_factory=FeatureFlags)

# Data-specific configuration
class DataConfig(BaseModel):
    datamode: DATAMODE = "xray"
    free_flag: str = "R-free-flags"
    testset_value: int = 1
    min_resolution: Optional[float] = None
    max_resolution: Optional[float] = None
    voxel_spacing: float = 4.5
    msa_subratio: Optional[float] = None
    w_plddt: float = 0.0

    @field_validator('datamode', mode='before')
    @classmethod
    def validate_datamode(cls, v):
        if isinstance(v, str):
            try:
                return DATAMODE(v)
            except ValueError:
                valid_values = [e.value for e in DATAMODE]
                raise ValueError(f"Invalid datamode: {v}. Must be one of: {valid_values}")
        return v
    
    model_config = {
        "use_enum_values": True
    }

# Main configuration class
class RocketRefinmentConfig(BaseModel):
    # Metadata
    note: str = ""
    
    # Nested configurations
    paths: PathConfig = Field(default_factory=PathConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    
    model_config = {
        "use_enum_values": True
    }

    # Mapping for flat to nested structure conversion
    _flat_to_nested_map: ClassVar[Dict[str, str]] = {
        # Paths
        "path": "paths.path",
        "file_root": "paths.file_root",
        "template_pdb": "paths.template_pdb",
        "input_msa": "paths.input_msa",
        "sub_msa_path": "paths.sub_msa_path",
        "sub_delmat_path": "paths.sub_delmat_path",
        "msa_feat_init_path": "paths.msa_feat_init_path",
        "starting_bias": "paths.starting_bias",
        "starting_weights": "paths.starting_weights",
        "uuid_hex": "paths.uuid_hex",
        
        # Execution
        "cuda_device": "execution.cuda_device",
        "num_of_runs": "execution.num_of_runs",
        "verbose": "execution.verbose",
        
        # Algorithm
        "bias_version": "algorithm.bias_version",
        "iterations": "algorithm.iterations",
        "init_recycling": "algorithm.init_recycling",
        "domain_segs": "algorithm.domain_segs",
        
        # Optimization
        "additive_learning_rate": "algorithm.optimization.additive_learning_rate",
        "multiplicative_learning_rate": "algorithm.optimization.multiplicative_learning_rate",
        "weight_decay": "algorithm.optimization.weight_decay",
        "batch_sub_ratio": "algorithm.optimization.batch_sub_ratio",
        "number_of_batches": "algorithm.optimization.number_of_batches",
        "rbr_opt_algorithm": "algorithm.optimization.rbr_opt_algorithm",
        "rbr_lbfgs_learning_rate": "algorithm.optimization.rbr_lbfgs_learning_rate",
        "smooth_stage_epochs": "algorithm.optimization.smooth_stage_epochs",
        "phase2_final_lr": "algorithm.optimization.phase2_final_lr",
        "l2_weight": "algorithm.optimization.l2_weight",
        
        # Features
        "solvent": "algorithm.features.solvent",
        "sfc_scale": "algorithm.features.sfc_scale",
        "refine_sigmaA": "algorithm.features.refine_sigmaA",
        "additional_chain": "algorithm.features.additional_chain",
        "bias_from_fullmsa": "algorithm.features.bias_from_fullmsa",
        "chimera_profile": "algorithm.features.chimera_profile",
        
        # Data
        "datamode": "data.datamode",
        "free_flag": "data.free_flag",
        "testset_value": "data.testset_value",
        "min_resolution": "data.min_resolution",
        "max_resolution": "data.max_resolution",
        "voxel_spacing": "data.voxel_spacing",
        "msa_subratio": "data.msa_subratio",
        "w_plddt": "data.w_plddt",
        
        # Metadata
        "note": "note"
    }
    
    # Helper methods for backward compatibility
    def __getattr__(self, name):
        """Allow access to nested attributes directly for backward compatibility"""
        if name in self._flat_to_nested_map:
            path = self._flat_to_nested_map[name].split('.')
            value = self
            for part in path:
                value = getattr(value, part)
            return value
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    # YAML serialization methods
    # TODO: make the saved yml have fields in order
    # def to_yaml_file(self, file_path: str) -> None:
    #     with open(file_path, "w") as file:
    #         yaml.dump(self.model_dump(), file)

    def to_yaml_file(self, file_path: str) -> None:
        """Save configuration to YAML with fields in specific order"""
        # Convert model to dict
        model_dict = self.model_dump()
        
        # Create an ordered dictionary with the desired field order
        ordered_dict = {}
        # Define the order of top-level fields
        field_order = ["note", "data", "paths", "execution", "algorithm"]
        
        # Add fields in the specified order
        for field in field_order:
            if field in model_dict:
                ordered_dict[field] = model_dict[field]
        
        # Add any remaining fields that weren't in our order list
        for key, value in model_dict.items():
            if key not in ordered_dict:
                ordered_dict[key] = value
    
        # Write to file
        with open(file_path, "w") as file:
            yaml.dump(ordered_dict, file, sort_keys=False)
    
    def to_flat_yaml_file(self, file_path: str) -> None:
        """Save configuration in the old flat format for backward compatibility"""
        flat_dict = self.to_flat_dict()
        with open(file_path, "w") as file:
            yaml.dump(flat_dict, file)
    
    def to_flat_dict(self) -> Dict[str, Any]:
        """Convert the nested structure to a flat dictionary"""
        result = {}
        for flat_key, nested_path in self._flat_to_nested_map.items():
            path_parts = nested_path.split('.')
            value = self
            for part in path_parts:
                value = getattr(value, part)
            result[flat_key] = value
        return result

    @classmethod
    def from_yaml_file(cls, file_path: str):
        with open(file_path, "r") as file:
            payload = yaml.safe_load(file)
        
        # Try to determine if this is a flat or nested format
        if any(key in payload for key in ["paths", "algorithm", "data", "execution"]):
            # This appears to be a nested format
            return cls.model_validate(payload)
        else:
            # This appears to be a flat format
            return cls.from_flat_dict(payload)
    
    @classmethod
    def from_flat_dict(cls, flat_dict: Dict[str, Any]):
        """Create an instance from a flat dictionary (old format)"""
        # Initialize nested dictionaries
        nested_dict = {
            "paths": {},
            "execution": {},
            "algorithm": {
                "optimization": {},
                "features": {}
            },
            "data": {},
            "note": flat_dict.get("note", "")
        }
        
        # Map flat keys to nested structure
        for flat_key, value in flat_dict.items():
            if flat_key in cls._flat_to_nested_map:
                nested_path = cls._flat_to_nested_map[flat_key].split('.')
                
                # Navigate to the correct nested dictionary
                target_dict = nested_dict
                for part in nested_path[:-1]:
                    target_dict = target_dict[part]
                
                # Set the value in the nested structure
                target_dict[nested_path[-1]] = value
        
        # Create the instance
        return cls.model_validate(nested_dict)