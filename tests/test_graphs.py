import numpy as np

from keisler_2022.graphs import (
    Geometry,
    geometry_era5,
    geometry_h3,
    indices_h3,
    xyz_from_latlonr,
)


def test_xyz_from_latlonr_unit_sphere() -> None:
    """Test xyz_from_latlonr on unit sphere."""
    # Test at north pole
    lat = np.array([np.pi / 2])
    lon = np.array([0.0])
    r = np.array([1.0])
    xyz = xyz_from_latlonr(lat, lon, r)

    assert xyz.shape == (1, 3)
    assert np.isclose(xyz[0, 0], 0.0, atol=1e-10)  # x
    assert np.isclose(xyz[0, 1], 0.0, atol=1e-10)  # y
    assert np.isclose(xyz[0, 2], 1.0, atol=1e-10)  # z

    # Test at equator, prime meridian
    lat = np.array([0.0])
    lon = np.array([0.0])
    r = np.array([1.0])
    xyz = xyz_from_latlonr(lat, lon, r)

    assert np.isclose(xyz[0, 0], 1.0, atol=1e-10)  # x
    assert np.isclose(xyz[0, 1], 0.0, atol=1e-10)  # y
    assert np.isclose(xyz[0, 2], 0.0, atol=1e-10)  # z


def test_xyz_from_latlonr_multiple_points() -> None:
    """Test xyz_from_latlonr with multiple points."""
    lat = np.array([0.0, np.pi / 2, -np.pi / 2])
    lon = np.array([0.0, 0.0, 0.0])
    r = np.array([1.0, 1.0, 1.0])
    xyz = xyz_from_latlonr(lat, lon, r)

    assert xyz.shape == (3, 3)
    # Check that all points are on unit sphere
    for i in range(3):
        dist = np.sqrt(np.sum(xyz[i] ** 2))
        assert np.isclose(dist, 1.0, atol=1e-10)


def test_geometry_creation() -> None:
    """Test Geometry creation and validation."""
    n_points = 10
    xyz = np.random.randn(n_points, 3).astype(np.float32)
    # Normalize to unit sphere
    xyz = xyz / np.linalg.norm(xyz, axis=1, keepdims=True)

    # Create latlonr from xyz (inverse transformation)
    r = np.ones(n_points)
    lat = np.arcsin(xyz[:, 2])  # z = r * sin(lat)
    lon = np.arctan2(xyz[:, 1], xyz[:, 0])  # y/x = tan(lon)
    lon[lon < 0] += 2 * np.pi  # Normalize to [0, 2π]
    latlonr = np.column_stack([lat, lon, r])

    geometry = Geometry(xyz=xyz, latlonr=latlonr)

    assert geometry.n_pix == n_points
    assert geometry.xyz.shape == (n_points, 3)
    assert geometry.latlonr.shape == (n_points, 3)


def test_geometry_concatenate() -> None:
    """Test Geometry concatenation."""
    n1, n2 = 5, 7
    xyz1 = np.random.randn(n1, 3).astype(np.float32)
    latlonr1 = np.random.randn(n1, 3).astype(np.float32)
    geo1 = Geometry(xyz=xyz1, latlonr=latlonr1)

    xyz2 = np.random.randn(n2, 3).astype(np.float32)
    latlonr2 = np.random.randn(n2, 3).astype(np.float32)
    geo2 = Geometry(xyz=xyz2, latlonr=latlonr2)

    geo_combined = geo1.concatenate(geo2)

    assert geo_combined.n_pix == n1 + n2
    assert geo_combined.xyz.shape == (n1 + n2, 3)
    assert geo_combined.latlonr.shape == (n1 + n2, 3)
    assert np.allclose(geo_combined.xyz[:n1], xyz1)
    assert np.allclose(geo_combined.xyz[n1:], xyz2)


def test_geometry_era5_shape() -> None:
    """Test geometry_era5 returns correct shape."""
    geo = geometry_era5(reso_degrees=1.0)

    # 1.0 degree resolution: 181 lats (-90 to +90), 360 lons (0 to 359)
    expected_n_pix = 181 * 360
    assert geo.n_pix == expected_n_pix
    assert geo.xyz.shape == (expected_n_pix, 3)
    assert geo.latlonr.shape == (expected_n_pix, 3)


def test_geometry_era5_coordinates() -> None:
    """Test geometry_era5 coordinate ranges."""
    geo = geometry_era5(reso_degrees=1.0)

    # Check latitude range: -π/2 to π/2
    lat = geo.latlonr[:, 0]
    assert np.min(lat) >= -np.pi / 2 - 1e-6
    assert np.max(lat) <= np.pi / 2 + 1e-6

    # Check longitude range: should be normalized to [0, 2π] or [-π, π]
    lon = geo.latlonr[:, 1]
    # The code normalizes lon >= 180 to subtract 360, so range should be [-π, π]
    assert np.min(lon) >= -np.pi - 1e-6
    assert np.max(lon) <= np.pi + 1e-6

    # Check r is always 1.0
    r = geo.latlonr[:, 2]
    assert np.allclose(r, 1.0)


def test_geometry_era5_unit_sphere() -> None:
    """Test that geometry_era5 points are on unit sphere."""
    geo = geometry_era5(reso_degrees=1.0)

    # Check that all points are approximately on unit sphere
    distances = np.linalg.norm(geo.xyz, axis=1)
    assert np.allclose(distances, 1.0, atol=1e-6)


def test_geometry_h3_shape() -> None:
    """Test geometry_h3 returns correct shape for level 2."""
    geo = geometry_h3(h3_level=2)

    # H3 level 2 should have 5882 cells
    assert geo.n_pix == 5882
    assert geo.xyz.shape == (5882, 3)
    assert geo.latlonr.shape == (5882, 3)


def test_geometry_h3_coordinates() -> None:
    """Test geometry_h3 coordinate ranges."""
    geo = geometry_h3(h3_level=2)

    # Check latitude range: -π/2 to π/2
    lat = geo.latlonr[:, 0]
    assert np.min(lat) >= -np.pi / 2 - 1e-6
    assert np.max(lat) <= np.pi / 2 + 1e-6

    # Check longitude range
    lon = geo.latlonr[:, 1]
    assert np.min(lon) >= -np.pi - 1e-6
    assert np.max(lon) <= np.pi + 1e-6

    # Check r is always 1.0
    r = geo.latlonr[:, 2]
    assert np.allclose(r, 1.0)


def test_geometry_h3_unit_sphere() -> None:
    """Test that geometry_h3 points are on unit sphere."""
    geo = geometry_h3(h3_level=2)

    # Check that all points are approximately on unit sphere
    distances = np.linalg.norm(geo.xyz, axis=1)
    assert np.allclose(distances, 1.0, atol=1e-6)


def test_indices_h3_levels() -> None:
    """Test indices_h3 returns correct number of indices for different levels."""
    # Level 0: should have 122 base cells
    ind0 = indices_h3(h3_level=0)
    assert len(ind0) == 122

    # Level 1: should have more cells
    ind1 = indices_h3(h3_level=1)
    assert len(ind1) > len(ind0)

    # Level 2: should have even more cells
    ind2 = indices_h3(h3_level=2)
    assert len(ind2) > len(ind1)
    assert len(ind2) == 5882


def test_indices_h3_unique() -> None:
    """Test that indices_h3 returns unique H3 indices."""
    ind = indices_h3(h3_level=2)

    # All indices should be unique
    assert len(ind) == len(set(ind))

    # All should be valid H3 indices (strings)
    assert all(isinstance(i, str) for i in ind)
    assert all(len(i) > 0 for i in ind)
