# SPDX-License-Identifier: Apache-2.0
""" Algorithms to create cell positions. """

import logging

import numpy as np

from rusty_placement.algorithms import uniform

L = logging.getLogger(__name__)


def _assert_cubic_voxels(voxel_data):
    """Helper function that verifies whether the voxels of given voxel data are
    cubic.
    """
    a, b, c = np.abs(voxel_data.voxel_dimensions)
    assert np.isclose(a, b) and np.isclose(a, c)


def _get_cell_count(density, density_factor):
    """Helper function that counts the number of cells per voxel and the total
    number of cells.
    """
    voxel_mm3 = density.voxel_volume / 1e9  # voxel volume is in um^3
    cell_count_per_voxel = density.raw * density_factor * voxel_mm3
    cell_count = int(np.round(np.sum(cell_count_per_voxel)))

    return cell_count_per_voxel, cell_count


def _get_seed(cell_count_per_voxel, voxel_data):
    """Helper function to calculate seed for Poisson disc sampling. The seed
    is set in a low-density area, to try to avoid that the algorithm gets
    stuck in the high-density areas. Other pitfalls of the Poisson disc
    sampling algorithm are illustrated on
    http://devmag.org.za/2009/05/03/poisson-disk-sampling/.
    """
    assert np.all(cell_count_per_voxel >= 0)
    positives = np.ma.masked_values(cell_count_per_voxel, 0)
    idcs = np.unravel_index(np.argmin(positives), cell_count_per_voxel.shape)
    return voxel_data.indices_to_positions(idcs) + voxel_data.voxel_dimensions / 2.0


def get_bbox_indices_nonzero_entries(data):
    """Calculate bounding box of indices of non-zero entries of a given
    three-dimensional numpy.array.
    """
    idx = np.nonzero(data)
    return np.array(
        [
            [np.min(idx[0]), np.min(idx[1]), np.min(idx[2])],
            [np.max(idx[0]), np.max(idx[1]), np.max(idx[2])],
        ]
    )


def get_bbox_nonzero_entries(data, bbox, voxel_dimensions):
    """Calculate bounding box of non-zero entries of a given three-dimensional
    numpy.array.

    Args:
        data: three-dimensional numpy.array
        bbox: original bbox
        voxel_dimensions: numpy.array with voxel size in each dimension
    """
    bbox_idx_nonzero = get_bbox_indices_nonzero_entries(data)
    bbox_nonzero = bbox[0, :] + bbox_idx_nonzero * voxel_dimensions
    bbox_nonzero[1, :] += voxel_dimensions
    return bbox_nonzero


def create_cell_positions(density, density_factor=1.0, method="basic", method_options = None, seed=None):
    """Given cell density volumetric data, create cell positions.

    Total cell count is calculated based on cell density values.

    Args:
        density(VoxelData): cell density (count / mm^3)
        density_factor(float): reduce / increase density proportionally for all
            voxels. Default is 1.0.
        method(str): algorithm used for cell position creation.
            Default is ``basic`` and the possible values are:

            - ``basic``: generated positions may collide or form clusters
            - ``poisson_disc``: positions are created with poisson disc sampling algorithm
              where minimum distance between points is modulated based on density values

        seed(int): (optional) the numpy random seed to be used.
            Defaults to None, in which case the seed is not set and the outcome
            cannot be predicted.

    Returns:
        numpy.array: array of positions of shape (cell_count, 3) where each row represents
        a cell and the columns correspond to (x, y, z).
    """
    if np.count_nonzero(density.raw < 0) != 0:
        raise ValueError("Found negative densities, aborting")

    if seed is not None:
        np.random.seed(seed)  # make the output reproducible

    method_options = method_options or {}

    position_generators = {
        "basic": uniform.create_cell_positions,
    }

    return position_generators[method](density, density_factor)
