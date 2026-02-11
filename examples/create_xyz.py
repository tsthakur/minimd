import argparse
import numpy as np


def create_fcc_xyz(filename, element="Ar", n_cells=3, rho=0.8):
    """
    Generate an FCC lattice and write it to an XYZ file.
    
    Parameters
    ----------
    filename : str
        Output XYZ filename.
    element : str
        Element symbol (default: "Ar").
    n_cells : int
        Supercell size (default: 3).
    rho : float
        Target density (default: 0.8).
    
    Returns
    -------
    positions : np.ndarray
        Array of atomic positions with shape (n_atoms, 3).
    L : float
        Box length.
    """
    n_atoms = 4 * n_cells**3
    vol = n_atoms / rho
    L = vol ** (1/3)
    a = L / n_cells

    # Using FCC basis
    basis = np.array([[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]])
    positions = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                for b in basis:
                    positions.append((np.array([ix, iy, iz]) + b) * a)
    positions = np.array(positions)

    with open(filename, "w") as f:
        f.write(f'{n_atoms}\n')
        f.write(f'FCC lattice, L={L:.6f}\n')
        for p in positions:
            f.write(f'{element} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n')
    
    return positions, L


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FCC lattice XYZ file")
    parser.add_argument("filename", help="Output XYZ filename")
    parser.add_argument("-e", "--element", default="Ar", help="Element symbol (default: Ar)")
    parser.add_argument("-n", "--n_cells", type=int, default=3, help="Supercell size (default: 3)")
    parser.add_argument("-r", "--rho", type=float, default=0.8, help="Target density (default: 0.8)")
    
    args = parser.parse_args()
    positions, L = create_fcc_xyz(args.filename, args.element, args.n_cells, args.rho)
    print(f"Created FCC lattice with {len(positions)} atoms, L={L:.6f}")
