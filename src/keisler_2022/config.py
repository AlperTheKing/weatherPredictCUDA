from __future__ import annotations

import importlib.resources as ir
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataConfig:
    # Shipped artifacts (current defaults reference the repo's data/ files).
    # These names mirror the existing filenames to avoid breakage.
    weights_file: str = "good_era5_forecast_batch001_feats0256_blocks009_steps12_stride02_noise0.0000_l10.0000000_lr0.000003_lrd0_ethereal-brook-316_val1.0347996_idx030200.pkl"
    normalizer_file: str = "temporal_normalizer_rk-era5-data_zarr-era5_1979begin_2020end_03hr_6phys_181lat_360lon_13levels_blosc1comp_Corder_monolith.npz.gz"

    senders_receivers_encoder: str = "senders_receivers_encoder.npz.gz"
    senders_receivers_processor: str = "senders_receivers_processor.npz.gz"
    senders_receivers_decoder: str = "senders_receivers_decoder.npz.gz"

    node_features_e: str = "node_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz"
    edge_features_e: str = "edge_features_n71042_e112246_s-8416688801745003395_r-6736346125390000850.npz.gz"
    node_features_p: str = (
        "node_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz"
    )
    edge_features_p: str = (
        "edge_features_n5882_e41162_s-1135048384487896564_r7866883539119236492.npz.gz"
    )
    node_features_d: str = "node_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz"
    edge_features_d: str = "edge_features_n71042_e112246_s-6736346125390000850_r-8416688801745003395.npz.gz"

    orography_landsea_file: str = "orography_landsea.npz.gz"

    # Optional local cache for user-built artifacts (not required for defaults)
    cache_dir: Optional[Path] = None  # e.g., Path("~/.cache/keisler_2022").expanduser()


@dataclass(frozen=True)
class GraphConfig:
    reso_era5_deg: float = 1.0
    h3_level: int = 2


@dataclass(frozen=True)
class ModelConfig:
    n_features: int = 256
    n_processor_blocks: int = 9
    n_mlp_layers_encoder: int = 2
    n_mlp_layers_processor: int = 2
    n_mlp_layers_decoder: int = 2
    n_channels_out: int = 78
    use_lat: bool = True
    use_lon: bool = True
    use_doy: bool = True


@dataclass(frozen=True)
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    graphs: GraphConfig = field(default_factory=GraphConfig)
    data: DataConfig = field(default_factory=DataConfig)


def resolve_artifact(filename: str) -> str:
    """Locate a shipped data artifact, preferring the installed package path."""
    try:
        candidate = ir.files("keisler_2022") / "data" / filename
        with ir.as_file(candidate) as materialized_path:
            path_str = str(materialized_path)
            if os.path.exists(path_str):
                return path_str
    except Exception:
        pass
    return os.path.join("data", filename)
