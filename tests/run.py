from pathlib import Path

from rusty_placement.main import create_cell_positions
import voxcell


DATA_DIR = Path(__file__).parent.parent.parent / "DATA"


if __name__ == "__main__":


    density = voxcell.VoxelData.load_nrrd(DATA_DIR / "Generic_Excitatory_Neuron_MType_Generic_Excitatory_Neuron_EType.nrrd")

    res = create_cell_positions(density, density_factor=1.0, method="basic")

    print(res)
