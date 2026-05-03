"""
The GEOMETRY defines the locations of points (nodes) in 3d, i.e. [lat, lon, r] and [x, y, z].
The CONNECTION defines the connections (edges) between the nodes, i.e. [senders, receivers].
"""

from dataclasses import dataclass
from itertools import chain
from typing import Any

import h3
import jraph
import numpy as np
from numpy.typing import NDArray

from keisler_2022.config import Config, resolve_artifact


@dataclass
class Geometry:
    """Represents geometry of points in both Cartesian (xyz) and spherical (latlonr) coordinates.

    Both arrays have shape (n_pix, 3) and represent the same set of points.
    - xyz: [x, y, z] in Cartesian coordinates
    - latlonr: [lat, lon, r] in radians (lat: -π/2 to π/2, lon: 0 to 2π)
    """

    xyz: NDArray[np.floating]
    latlonr: NDArray[np.floating]

    def __post_init__(self) -> None:
        """Validate that xyz and latlonr have compatible shapes."""
        assert len(self.xyz.shape) == 2
        assert len(self.latlonr.shape) == 2
        assert self.xyz.shape[0] == self.latlonr.shape[0]
        assert self.xyz.shape[1] == 3
        assert self.latlonr.shape[1] == 3

    @property
    def n_pix(self) -> int:
        """Number of points in this geometry."""
        return int(self.xyz.shape[0])

    def concatenate(self, other: "Geometry") -> "Geometry":
        """Concatenate this geometry with another along the first axis."""
        return Geometry(
            xyz=np.concatenate([self.xyz, other.xyz], axis=0),
            latlonr=np.concatenate([self.latlonr, other.latlonr], axis=0),
        )


class StaticGraph:
    def __init__(
        self,
        geometry: Geometry,
        senders: NDArray[np.integer] | list[int],
        receivers: NDArray[np.integer] | list[int],
        node_features_savename: str,
        edge_features_savename: str,
    ) -> None:
        assert len(senders) == len(receivers)

        self.geometry = geometry
        self.n_pix = geometry.n_pix
        self.n_node = self.n_pix
        self.n_edge = len(senders)
        self.senders = senders
        self.receivers = receivers
        self.node_features_savename = node_features_savename
        self.edge_features_savename = edge_features_savename
        self.nodes = self.define_node_features()
        self.edges = self.define_edge_features()

    def define_node_features(self) -> dict[str, Any]:
        tmp = np.load(self.node_features_savename, allow_pickle=True)
        nodes = {}
        for key in tmp.keys():
            nodes[key] = tmp[key]
        del nodes["local_coords"]
        return nodes

    def define_edge_features(self) -> dict[str, Any]:
        tmp = np.load(self.edge_features_savename, allow_pickle=True)
        edges = {}
        for key in tmp.keys():
            edges[key] = tmp[key]
        return edges

    def jraph(self) -> jraph.GraphsTuple:
        """Create a jraph GraphsTuple representation of the graph"""
        this_nodes = dict(self.nodes)
        this_edges = dict(self.edges)
        # Ensure index dtypes are int32 for JAX/XLA portability
        senders = np.asarray(self.senders, dtype=np.int32)
        receivers = np.asarray(self.receivers, dtype=np.int32)
        n_node = np.asarray([self.n_pix], dtype=np.int32)
        n_edge = np.asarray([self.n_edge], dtype=np.int32)
        graph = jraph.GraphsTuple(
            nodes=this_nodes,
            edges=this_edges,
            n_node=n_node,
            n_edge=n_edge,
            senders=senders,
            receivers=receivers,
            globals=None,
        )
        return graph


def xyz_from_latlonr(
    lat: NDArray[np.floating], lon: NDArray[np.floating], r: NDArray[np.floating]
) -> NDArray[np.floating]:
    """LAT and LON are in radians.
    LAT goes from -PI/2 (south pole) to +PI/2 (north pole)
    LON goes from 0 to 2PI
    """
    x = r * np.cos(lon) * np.cos(lat)
    y = r * np.sin(lon) * np.cos(lat)
    z = r * np.sin(lat)
    xyz = np.vstack([x, y, z]).T  # has shape (n_pix, 3)
    return xyz


def geometry_era5(reso_degrees: float) -> Geometry:
    """Given RESO_DEGREES, return Geometry for an ERA5 lat/lon grid."""
    deg2rad = np.pi / 180
    lat1d = np.arange(-90, +91, reso_degrees) * deg2rad
    lon1d = np.arange(0, 360, reso_degrees)
    lon1d[lon1d >= 180] -= 360
    lon1d *= deg2rad
    lon2d, lat2d = np.meshgrid(lon1d, lat1d)
    lat = lat2d.ravel()
    lon = lon2d.ravel()
    r = np.ones_like(lat)
    xyz = xyz_from_latlonr(lat, lon, r)  # has shape (n_pix, 3)
    latlonr = np.array([lat, lon, r]).T  # has shape (n_pix, 3)
    return Geometry(xyz=xyz, latlonr=latlonr)


def indices_h3(h3_level: int) -> list[str]:
    level_current = 0
    ind_current = sorted(list(h3.get_res0_indexes()))
    while level_current < h3_level:
        level_current += 1
        ind_current = sorted(
            list(
                chain.from_iterable(
                    [list(h3.h3_to_children(i, level_current)) for i in ind_current]
                )
            )
        )
    return ind_current


def geometry_h3(h3_level: int = 2) -> Geometry:
    ind_use = indices_h3(h3_level)
    n_pix = len(ind_use)
    latlon = np.array([h3.h3_to_geo(i) for i in ind_use])
    deg2rad = np.pi / 180.0
    latlon *= deg2rad
    lat = latlon[:, 0]
    lon = latlon[:, 1]
    r = np.ones(n_pix)
    xyz = xyz_from_latlonr(lat, lon, r)  # has shape (n_pix, 3)
    latlonr = np.array([lat, lon, r]).T  # has shape (n_pix, 3)
    return Geometry(xyz=xyz, latlonr=latlonr)


class GraphBuilder:
    """Builds encoder, processor, and decoder graphs from config.

    After calling :meth:`build`, the following attributes are available:
    ``geometry_era5``, ``geometry_h3``, ``geometry_all``,
    ``static_graphs``, ``n_node_era5``, ``n_node_h3``, ``n_node``.
    """

    def __init__(self, config: Config | None = None):
        self.config = config if config is not None else Config()

    def build(self) -> dict[str, jraph.GraphsTuple]:
        self.geometry_era5 = geometry_era5(self.config.graphs.reso_era5_deg)
        self.geometry_h3 = geometry_h3(h3_level=self.config.graphs.h3_level)
        self.geometry_all = self.geometry_era5.concatenate(self.geometry_h3)

        def _load_sr(filename: str) -> tuple[NDArray[np.integer], NDArray[np.integer]]:
            tmp = np.load(resolve_artifact(filename))
            return tmp["senders"].astype("int32"), tmp["receivers"].astype("int32")

        senders_e, receivers_e = _load_sr(self.config.data.senders_receivers_encoder)
        senders_p, receivers_p = _load_sr(self.config.data.senders_receivers_processor)
        senders_d, receivers_d = _load_sr(self.config.data.senders_receivers_decoder)

        self.static_graphs: dict[str, StaticGraph] = {
            "e": StaticGraph(
                self.geometry_all,
                senders_e,
                receivers_e,
                node_features_savename=resolve_artifact(
                    self.config.data.node_features_e
                ),
                edge_features_savename=resolve_artifact(
                    self.config.data.edge_features_e
                ),
            ),
            "p": StaticGraph(
                self.geometry_h3,
                senders_p,
                receivers_p,
                node_features_savename=resolve_artifact(
                    self.config.data.node_features_p
                ),
                edge_features_savename=resolve_artifact(
                    self.config.data.edge_features_p
                ),
            ),
            "d": StaticGraph(
                self.geometry_all,
                senders_d,
                receivers_d,
                node_features_savename=resolve_artifact(
                    self.config.data.node_features_d
                ),
                edge_features_savename=resolve_artifact(
                    self.config.data.edge_features_d
                ),
            ),
        }

        self.n_node_era5 = self.geometry_era5.n_pix
        self.n_node_h3 = self.geometry_h3.n_pix
        self.n_node = self.n_node_era5 + self.n_node_h3

        return {k: v.jraph() for k, v in self.static_graphs.items()}
