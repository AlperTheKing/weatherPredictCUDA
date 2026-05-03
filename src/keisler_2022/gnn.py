import functools
from typing import Any

import haiku as hk
import jax
import jax.numpy as jnp
import jraph


def make_mlp(
    n_hidden_layers: int,
    n_hidden: int,
    n_out: int,
    layer_norm: bool = True,
) -> hk.Sequential:
    """Build an MLP: (Linear + ReLU) x n_hidden_layers, then Linear, optionally LayerNorm."""
    layers: list[Any] = []
    for _ in range(n_hidden_layers):
        layers.append(hk.Linear(n_hidden))
        layers.append(jax.nn.relu)
    layers.append(hk.Linear(n_out))
    if layer_norm:
        layers.append(hk.LayerNorm(axis=-1, create_scale=True, create_offset=True))
    return hk.Sequential(layers)


def edge_update_fn_encoder(
    edges: dict[str, Any],
    sender_nodes: dict[str, Any],
    receiver_nodes: dict[str, Any],
    globals_: dict[str, Any] | None,
    n_features: int = 128,
    n_mlp_layers: int = 2,
    use_lat: bool | None = None,
    use_lon: bool | None = None,
    use_doy: bool | None = None,
) -> dict[str, Any]:
    """Edge update function for Encoder GraphNet."""
    net = make_mlp(n_mlp_layers, n_features, n_features)

    concat_features_list: list[Any] = [
        edges["local_coords_row"],
        sender_nodes["data"],
        sender_nodes["solar"],
        sender_nodes["orography"],
        sender_nodes["landsea"],
    ]
    if use_lat:
        concat_features_list.append(sender_nodes["coslat"])
        concat_features_list.append(sender_nodes["sinlat"])
    if use_lon:
        concat_features_list.append(sender_nodes["coslon"])
        concat_features_list.append(sender_nodes["sinlon"])
    if use_doy:
        concat_features_list.append(sender_nodes["doy"])
    concat_features: Any = jnp.concatenate(concat_features_list, axis=-1)
    edges["features"] = net(concat_features)
    return edges


def node_update_fn_encoder(
    nodes: dict[str, Any],
    agg_sender_edges: dict[str, Any],
    agg_receiver_edges: dict[str, Any],
    globals_: dict[str, Any] | None,
    n_features: int = 128,
    n_mlp_layers: int = 2,
) -> dict[str, Any]:
    """Node update function for Encoder GraphNet."""
    net = make_mlp(n_mlp_layers, n_features, n_features)

    # AGG_RECEIVER_EDGES is the aggregation over edges where this node was the receiver.
    # nodes['inv_n_senders'] is the inverse of the number of senders to this node.
    concat_features: Any = agg_receiver_edges["features"] * nodes["inv_n_senders"]
    nodes["features"] = net(concat_features)
    return nodes


def edge_update_fn_processor(
    edges: dict[str, Any],
    sender_nodes: dict[str, Any],
    receiver_nodes: dict[str, Any],
    globals_: dict[str, Any] | None,
    n_features: int = 128,
    n_mlp_layers: int = 2,
) -> dict[str, Any]:
    """Edge update function for Processor GraphNet."""
    net = make_mlp(n_mlp_layers, n_features, n_features)

    concat_features_list: list[Any] = [
        edges["features"],
        sender_nodes["features"],
        receiver_nodes["features"],
    ]

    concat_features: Any = jnp.concatenate(concat_features_list, axis=-1)
    edges["features"] += net(concat_features)
    return edges


def node_update_fn_processor(
    nodes: dict[str, Any],
    agg_sender_edges: dict[str, Any],
    agg_receiver_edges: dict[str, Any],
    globals_: dict[str, Any] | None,
    n_features: int = 128,
    n_mlp_layers: int = 2,
) -> dict[str, Any]:
    """Node update function for Processor GraphNet."""
    net = make_mlp(n_mlp_layers, n_features, n_features)

    # AGG_RECEIVER_EDGES is the aggregation over edges where this node was the receiver.
    # AGG_SENDER_EDGES is aggregation over edges where this node was the sender.
    # nodes['inv_n_receivers'] is the inverse of the number of receivers of this node.
    # nodes['inv_n_senders'] is the inverse of the number of senders to this node.
    concat_features_list: list[Any] = [
        nodes["features"],
        nodes["coslat"],
        nodes["sinlat"],
        nodes["coslon"],
        nodes["sinlon"],
        agg_sender_edges["features"] * nodes["inv_n_receivers"],
        agg_receiver_edges["features"] * nodes["inv_n_senders"],
    ]
    concat_features: Any = jnp.concatenate(concat_features_list, axis=-1)
    nodes["features"] += net(concat_features)
    return nodes


def edge_update_fn_decoder(
    edges: dict[str, Any],
    sender_nodes: dict[str, Any],
    receiver_nodes: dict[str, Any],
    globals_: dict[str, Any] | None,
    n_features: int = 128,
    n_mlp_layers: int = 2,
) -> dict[str, Any]:
    """Edge update function for Decoder GraphNet."""
    net = make_mlp(n_mlp_layers, n_features, n_features)

    concat_features_list: list[Any] = [
        edges["local_coords_row"],
        sender_nodes["features"],
        receiver_nodes["data"],
    ]

    concat_features: Any = jnp.concatenate(concat_features_list, axis=-1)
    edges["features"] = net(concat_features)
    return edges


def node_update_fn_decoder(
    nodes: dict[str, Any],
    agg_sender_edges: dict[str, Any],
    agg_receiver_edges: dict[str, Any],
    globals_: dict[str, Any] | None,
    n_features: int = 128,
    n_mlp_layers: int = 2,
    n_channels_out: int = 78,
) -> dict[str, Any]:
    """Node update function for Decoder GraphNet."""
    net = make_mlp(n_mlp_layers, n_features, n_channels_out, layer_norm=False)

    # AGG_RECEIVER_EDGES is the aggregation over edges where this node was the receiver.
    # nodes['inv_n_senders'] is the inverse of the number of senders to this node.
    concat_features_list: list[Any] = [
        nodes["data"],
        agg_receiver_edges["features"] * nodes["inv_n_senders"],
    ]
    concat_features: Any = jnp.concatenate(concat_features_list, axis=-1)
    nodes["change"] = net(concat_features)
    return nodes


def one_step_fn(
    graphs: dict[str, jraph.GraphsTuple],
    i_time: int,
    n_features: int = 128,
    n_processor_blocks: int = 6,
    n_mlp_layers: dict[str, int] | None = None,
    n_channels_out: int | None = None,
    use_lat: bool | None = None,
    use_lon: bool | None = None,
    use_doy: bool | None = None,
    n_node: int | None = None,
    n_node_era5: int | None = None,
    n_node_h3: int | None = None,
) -> tuple[dict[str, jraph.GraphsTuple], int]:
    assert n_mlp_layers is not None, "n_mlp_layers must be provided"
    assert n_channels_out is not None, "n_channels_out must be provided"
    graphs["e"].nodes["solar"] = graphs["e"].nodes["all_solar"][..., i_time]
    graphs["e"].nodes["doy"] = graphs["e"].nodes["all_doy"][..., i_time]
    i_time += 1

    # ENCODE
    update_edge_fn = functools.partial(
        edge_update_fn_encoder,
        n_features=n_features,
        n_mlp_layers=n_mlp_layers["e"],
        use_lat=use_lat,
        use_lon=use_lon,
        use_doy=use_doy,
    )

    update_node_fn = functools.partial(
        node_update_fn_encoder, n_features=n_features, n_mlp_layers=n_mlp_layers["e"]
    )

    encoder = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=None,
    )
    encoder = hk.remat(encoder)
    graphs["e"] = encoder(graphs["e"])

    # PROCESS
    net_edges = make_mlp(0, n_features, n_features, layer_norm=True)
    graphs["p"].edges["features"] = net_edges(graphs["p"].edges["local_coords_row"])

    graphs["p"].nodes["features"] = graphs["e"].nodes["features"][n_node_era5:]

    update_edge_fn = functools.partial(
        edge_update_fn_processor, n_features=n_features, n_mlp_layers=n_mlp_layers["p"]
    )
    update_node_fn = functools.partial(
        node_update_fn_processor, n_features=n_features, n_mlp_layers=n_mlp_layers["p"]
    )
    processor = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=None,
    )
    processor = hk.remat(processor)
    for _ in range(n_processor_blocks):
        graphs["p"] = processor(graphs["p"])

    # DECODE
    update_edge_fn = functools.partial(
        edge_update_fn_decoder, n_features=n_features, n_mlp_layers=n_mlp_layers["d"]
    )
    update_node_fn = functools.partial(
        node_update_fn_decoder,
        n_features=n_features,
        n_mlp_layers=n_mlp_layers["d"],
        n_channels_out=n_channels_out,
    )
    decoder = jraph.GraphNetwork(
        update_edge_fn=update_edge_fn,
        update_node_fn=update_node_fn,
        update_global_fn=None,
    )
    decoder = hk.remat(decoder)

    graphs["d"].nodes["data"] = graphs["e"].nodes["data"]
    graphs["d"].nodes["features"] = jnp.zeros((n_node, n_features))
    graphs["d"].nodes["features"] = (
        graphs["d"]
        .nodes["features"]
        .at[n_node_era5:]
        .set(graphs["p"].nodes["features"])
    )
    graphs["d"] = decoder(graphs["d"])

    # Add predicted change to the data
    graphs["e"].nodes["data"] += graphs["d"].nodes["change"]

    return graphs, i_time
