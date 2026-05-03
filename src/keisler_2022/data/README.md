These data assets are shipped with the package.

- `good_era5_forecast_batch001_feats0256_blocks009_steps12_stride02...pkl`: Pre-trained model weights
- `temporal_normalizer_rk-era5...npz.gz`: Temporal normalization statistics (mean/std) for model inputs/outputs
- `orography_landsea.npz.gz`: Orography and land-sea mask used by the model
- `senders_receivers_encoder.npz.gz`: Graph connectivity for the Encoder (lat/lon → H3)
- `senders_receivers_processor.npz.gz`: Graph connectivity for the Processor (H3 → H3)
- `senders_receivers_decoder.npz.gz`: Graph connectivity for the Decoder (H3 → lat/lon)
- `edge_features*.npz.gz`: Initial edge features for the different graphs
- `node_features*.npz.gz`: Initial node features for the different graphs

These files are automatically loaded by `GraphBuilder` and `Runner` when the package is installed.
