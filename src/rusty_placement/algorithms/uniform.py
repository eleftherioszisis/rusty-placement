from cell_placement._algorithms import uniform


def create_cell_positions(density, density_factor):
    """Create cell positions given cell density volumetric data (using uniform distribution).

    Within voxels, samples are created according to a uniform distribution.

    The total cell count is calculated based on cell density values.

    Args:
        density(VoxelData): cell density (count / mm^3)
        density_factor(float): reduce / increase density proportionally for all
            voxels. Default is 1.0.

    Returns:
        numpy.array: array of positions of shape (cell_count, 3) where each row
        represents a cell and the columns correspond to (x, y, z).
    """
    return uniform(
        density.voxel_dimensions.astype(float),
        density.raw.astype(float),
        density.offset.astype(float),
    )
