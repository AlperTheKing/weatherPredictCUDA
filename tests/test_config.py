from pathlib import Path

import pytest

from keisler_2022.config import Config, DataConfig, GraphConfig, ModelConfig


def test_config_defaults() -> None:
    """Test that Config can be instantiated with default values."""
    config = Config()

    # Check that nested configs are created
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.graphs, GraphConfig)
    assert isinstance(config.data, DataConfig)


def test_model_config_defaults() -> None:
    """Test ModelConfig default values."""
    model = ModelConfig()

    assert model.n_features == 256
    assert model.n_processor_blocks == 9
    assert model.n_mlp_layers_encoder == 2
    assert model.n_mlp_layers_processor == 2
    assert model.n_mlp_layers_decoder == 2
    assert model.n_channels_out == 78
    assert model.use_lat is True
    assert model.use_lon is True
    assert model.use_doy is True


def test_model_config_custom() -> None:
    """Test ModelConfig with custom values."""
    model = ModelConfig(
        n_features=128,
        n_processor_blocks=6,
        use_lat=False,
    )

    assert model.n_features == 128
    assert model.n_processor_blocks == 6
    assert model.use_lat is False
    # Other values should still be defaults
    assert model.use_lon is True
    assert model.use_doy is True


def test_graph_config_defaults() -> None:
    """Test GraphConfig default values."""
    graph = GraphConfig()

    assert graph.reso_era5_deg == 1.0
    assert graph.h3_level == 2


def test_graph_config_custom() -> None:
    """Test GraphConfig with custom values."""
    graph = GraphConfig(reso_era5_deg=2.0, h3_level=3)

    assert graph.reso_era5_deg == 2.0
    assert graph.h3_level == 3


def test_data_config_defaults() -> None:
    """Test DataConfig default values."""
    data = DataConfig()

    # Check that filenames are strings
    assert isinstance(data.weights_file, str)
    assert isinstance(data.normalizer_file, str)
    assert isinstance(data.senders_receivers_encoder, str)
    assert isinstance(data.orography_landsea_file, str)

    # Check that cache_dir is None by default
    assert data.cache_dir is None


def test_data_config_custom_cache() -> None:
    """Test DataConfig with custom cache directory."""
    cache_path = Path("~/.cache/keisler_2022").expanduser()
    data = DataConfig(cache_dir=cache_path)

    assert data.cache_dir == cache_path


def test_config_immutability() -> None:
    """Test that Config and its nested configs are immutable (frozen dataclasses)."""
    config = Config()

    with pytest.raises(AttributeError):
        setattr(config.model, "n_features", 128)

    with pytest.raises(AttributeError):
        setattr(config.graphs, "reso_era5_deg", 2.0)


def test_config_nested_access() -> None:
    """Test accessing nested config values."""
    config = Config()

    # Access nested values
    assert config.model.n_features == 256
    assert config.graphs.reso_era5_deg == 1.0
    assert config.data.weights_file.endswith(".pkl")
