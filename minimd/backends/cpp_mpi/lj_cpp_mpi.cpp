#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <cstring>
#include <mpi.h>

namespace py = pybind11;


void mpi_init() {
    int initialized;
    MPI_Initialized(&initialized);
    if (!initialized) MPI_Init(nullptr, nullptr);
}

void mpi_finalize() { MPI_Finalize(); }
int  mpi_rank()     { int r; MPI_Comm_rank(MPI_COMM_WORLD, &r); return r; }
int  mpi_size()     { int s; MPI_Comm_size(MPI_COMM_WORLD, &s); return s; }

// Only useful in case of serial code, for MPI we always rebuild
bool check_rebuild(const py::array_t<double>& positions,
    const py::array_t<double>& last_positions, double r_skin) {
    auto pos = positions.unchecked<2>();
    auto last = last_positions.unchecked<2>();
    double half_skin_sq = (r_skin / 2.0) * (r_skin / 2.0);

    for (ssize_t i = 0; i < pos.shape(0); ++i) {
        double dx = pos(i, 0) - last(i, 0);
        double dy = pos(i, 1) - last(i, 1);
        double dz = pos(i, 2) - last(i, 2);
        if (dx*dx + dy*dy + dz*dz > half_skin_sq) {
            return true;
        }
    }
    return false;
}

// Ghost atom approach instead of minimum image convention (serial version)
std::tuple<py::array_t<double>, py::array_t<int>, py::array_t<double>>
build_ghost_atoms(const py::array_t<double>& positions,
    const py::array_t<double>& box, double r_list) {

    auto pos = positions.unchecked<2>();
    auto box_u = box.unchecked<1>();
    ssize_t n_real = pos.shape(0);

    std::vector<double> ghost_pos_vec;
    std::vector<int> ghost_map_vec;
    std::vector<double> ghost_shift_vec;

    int shifts[3] = {-1, 0, 1};

    for (int sx : shifts) {
        for (int sy : shifts) {
            for (int sz : shifts) {
                if (sx == 0 && sy == 0 && sz == 0) continue;

                double shift_x = sx * box_u(0);
                double shift_y = sy * box_u(1);
                double shift_z = sz * box_u(2);

                for (ssize_t i = 0; i < n_real; ++i) {
                    // Atom qualifies for copying if it is near the boundary from which the shift originates
                    bool qual_x = (sx == 0) ||
                                  (sx == +1 && pos(i, 0) < r_list) ||
                                  (sx == -1 && pos(i, 0) > box_u(0) - r_list);
                    bool qual_y = (sy == 0) ||
                                  (sy == +1 && pos(i, 1) < r_list) ||
                                  (sy == -1 && pos(i, 1) > box_u(1) - r_list);
                    bool qual_z = (sz == 0) ||
                                  (sz == +1 && pos(i, 2) < r_list) ||
                                  (sz == -1 && pos(i, 2) > box_u(2) - r_list);

                    if (qual_x && qual_y && qual_z) {
                        ghost_pos_vec.push_back(pos(i, 0) + shift_x);
                        ghost_pos_vec.push_back(pos(i, 1) + shift_y);
                        ghost_pos_vec.push_back(pos(i, 2) + shift_z);
                        ghost_map_vec.push_back(static_cast<int>(i));
                        ghost_shift_vec.push_back(shift_x);
                        ghost_shift_vec.push_back(shift_y);
                        ghost_shift_vec.push_back(shift_z);
                    }
                }
            }
        }
    }

    ssize_t n_ghost = static_cast<ssize_t>(ghost_map_vec.size());

    py::array_t<double> ghost_positions({n_ghost, static_cast<ssize_t>(3)});
    py::array_t<int> ghost_map(n_ghost);
    py::array_t<double> ghost_shifts({n_ghost, static_cast<ssize_t>(3)});

    if (n_ghost > 0) {
        std::memcpy(ghost_positions.mutable_data(), ghost_pos_vec.data(),
                    n_ghost * 3 * sizeof(double));
        std::memcpy(ghost_map.mutable_data(), ghost_map_vec.data(),
                    n_ghost * sizeof(int));
        std::memcpy(ghost_shifts.mutable_data(), ghost_shift_vec.data(),
                    n_ghost * 3 * sizeof(double));
    }

    return std::make_tuple(ghost_positions, ghost_map, ghost_shifts);
}

// Update ghost positions when not rebuilding neighbour list
py::array_t<double> update_ghost_positions(
    const py::array_t<double>& positions,
    const py::array_t<int>& ghost_map,
    const py::array_t<double>& ghost_shifts) {

    auto pos = positions.unchecked<2>();
    auto gmap = ghost_map.unchecked<1>();
    auto gshifts = ghost_shifts.unchecked<2>();
    ssize_t n_ghost = gmap.shape(0);

    py::array_t<double> ghost_positions({n_ghost, static_cast<ssize_t>(3)});
    auto gpos = ghost_positions.mutable_unchecked<2>();

    for (ssize_t g = 0; g < n_ghost; ++g) {
        int real_idx = gmap(g);
        gpos(g, 0) = pos(real_idx, 0) + gshifts(g, 0);
        gpos(g, 1) = pos(real_idx, 1) + gshifts(g, 1);
        gpos(g, 2) = pos(real_idx, 2) + gshifts(g, 2);
    }

    return ghost_positions;
}

// Serial neighbour list 
std::tuple<py::array_t<int>, py::array_t<int>>
build_neighbour_list(const py::array_t<double>& positions,
    const py::array_t<double>& ghost_positions,
    const py::array_t<int>& ghost_map,
    double r_list) {

    auto pos = positions.unchecked<2>();
    auto gpos = ghost_positions.unchecked<2>();
    auto gmap = ghost_map.unchecked<1>();
    ssize_t n_real = pos.shape(0);
    ssize_t n_ghost = gpos.shape(0);
    double r_list_sq = r_list * r_list;

    std::vector<int> pair_i_vec;
    std::vector<int> pair_j_vec;

    for (ssize_t i = 0; i < n_real; ++i) {
        for (ssize_t j = i + 1; j < n_real; ++j) {
            double dx = pos(j, 0) - pos(i, 0);
            double dy = pos(j, 1) - pos(i, 1);
            double dz = pos(j, 2) - pos(i, 2);
            double dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < r_list_sq) {
                pair_i_vec.push_back(static_cast<int>(i));
                pair_j_vec.push_back(static_cast<int>(j));
            }
        }
    }

    for (ssize_t i = 0; i < n_real; ++i) {
        for (ssize_t g = 0; g < n_ghost; ++g) {
            // To avoid double counting
            if (gmap(g) <= static_cast<int>(i)) continue;

            double dx = gpos(g, 0) - pos(i, 0);
            double dy = gpos(g, 1) - pos(i, 1);
            double dz = gpos(g, 2) - pos(i, 2);
            double dist_sq = dx*dx + dy*dy + dz*dz;

            if (dist_sq < r_list_sq) {
                pair_i_vec.push_back(static_cast<int>(i));
                pair_j_vec.push_back(static_cast<int>(n_real + g));
            }
        }
    }

    py::array_t<int> pair_i(pair_i_vec.size(), pair_i_vec.data());
    py::array_t<int> pair_j(pair_j_vec.size(), pair_j_vec.data());
    return std::make_tuple(pair_i, pair_j);
}

// Serial force evaluation
std::tuple<py::array_t<double>, double>
compute_forces(const py::array_t<double>& positions,
    const py::array_t<double>& ghost_positions,
    const py::array_t<int>& ghost_map,
    const py::array_t<int>& pair_i,
    const py::array_t<int>& pair_j,
    double r_cut, double sigma, double epsilon) {

    auto pos = positions.unchecked<2>();
    auto gpos = ghost_positions.unchecked<2>();
    auto gmap = ghost_map.unchecked<1>();
    auto pi = pair_i.unchecked<1>();
    auto pj = pair_j.unchecked<1>();

    ssize_t n_real = pos.shape(0);
    ssize_t n_pairs = pi.shape(0);

    py::array_t<double> forces_np({n_real, static_cast<ssize_t>(3)});
    std::memset(forces_np.mutable_data(), 0, n_real * 3 * sizeof(double));
    auto forces = forces_np.mutable_unchecked<2>();

    double r_cut_sq = r_cut * r_cut;
    double sigma_sq = sigma * sigma;

    double inv_rc2 = sigma_sq / r_cut_sq;
    double inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2;
    double inv_rc12 = inv_rc6 * inv_rc6;
    double v_shift = 4.0 * epsilon * (inv_rc12 - inv_rc6);

    double energy = 0.0;

    for (ssize_t k = 0; k < n_pairs; ++k) {
        int i = pi(k);
        int j = pj(k);

        double pj_x, pj_y, pj_z;
        if (j < n_real) {
            pj_x = pos(j, 0);
            pj_y = pos(j, 1);
            pj_z = pos(j, 2);
        } else {
            ssize_t g = j - n_real;
            pj_x = gpos(g, 0);
            pj_y = gpos(g, 1);
            pj_z = gpos(g, 2);
        }

        double dx = pj_x - pos(i, 0);
        double dy = pj_y - pos(i, 1);
        double dz = pj_z - pos(i, 2);

        double dist_sq = dx*dx + dy*dy + dz*dz;

        if (dist_sq <= 0.0 || dist_sq >= r_cut_sq) {
            continue;
        }

        double inv_r2 = sigma_sq / dist_sq;
        double inv_r6 = inv_r2 * inv_r2 * inv_r2;
        double inv_r12 = inv_r6 * inv_r6;

        energy += epsilon * 4.0 * (inv_r12 - inv_r6) - v_shift;

        double f_over_r = epsilon * 24.0 * (2.0 * inv_r12 - inv_r6) / dist_sq;

        double fx = f_over_r * dx;
        double fy = f_over_r * dy;
        double fz = f_over_r * dz;

        forces(i, 0) -= fx;
        forces(i, 1) -= fy;
        forces(i, 2) -= fz;

        int j_real;
        if (j < n_real) {
            j_real = j;
        } else {
            j_real = static_cast<int>(gmap(j - n_real));
        };
        
        forces(j_real, 0) += fx;
        forces(j_real, 1) += fy;
        forces(j_real, 2) += fz;
    }

    return std::make_tuple(forces_np, energy);
}


// MPI functions
// We divide the cell into slabs along x direction and the MPI exchange
// takes place only along this 1 direction.
// The cell can be decomposed into 2D cuboids or 3D cubes, using MPI_Cart_create,
// but the code complexity and MPI overhead increases quickly. 
// This will be useful in case of 1000s of atom supercell, with 100s of MPI ranks,
// so not practically useful here, and would be difficult to benchmark as well.
// But is an excellent exercise for future.
// Incidentally, 3D decomposition is also what LAMMPS uses

// Similar function as build_ghost_atoms() (serial) in y,z direction, 
// and only as x-direction is covered by MPI exchange.
static void add_yz_pbc_ghosts(
    const std::vector<double>& src,   // flat [n x 3]
    int n,
    double Ly, double Lz, double r_list,
    std::vector<double>& ghost_vec)
{
    for (int sy = -1; sy <= 1; ++sy) {
        for (int sz = -1; sz <= 1; ++sz) {
            if (sy == 0 && sz == 0) continue;

            double shift_y = sy * Ly;
            double shift_z = sz * Lz;

            for (int k = 0; k < n; ++k) {
                double y = src[k*3 + 1];
                double z = src[k*3 + 2];

                bool qual_y = (sy == 0) ||
                              (sy == +1 && y < r_list) ||
                              (sy == -1 && y > Ly - r_list);
                bool qual_z = (sz == 0) ||
                              (sz == +1 && z < r_list) ||
                              (sz == -1 && z > Lz - r_list);

                if (qual_y && qual_z) {
                    ghost_vec.push_back(src[k*3]);
                    ghost_vec.push_back(y + shift_y);
                    ghost_vec.push_back(z + shift_z);
                }
            }
        }
    }
}

// All ranks receive full position array from Python. Each rank determines
// its own subdomain [x_lo, x_hi), then exchanges boundary atoms with its left
// and right neighbours via MPI_Sendrecv (halo exchange). 
// y, z PBC ghosts are generated locally from both local atoms and the received 
// x-direction ghosts, which also handles corner atoms near x-MPI and y-z boundary).
// No ghost_map or ghost_shifts like serial code, ghost positions are refreshed
// by calling this function again each time the neighbour list is rebuilt.
std::tuple<py::array_t<double>, py::array_t<double>>
build_ghost_atoms_mpi(const py::array_t<double>& positions,
    const py::array_t<double>& box, double r_list) {

    auto pos = positions.unchecked<2>();
    auto bx  = box.unchecked<1>();
    ssize_t n_total = pos.shape(0);

    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    double Lx = bx(0), Ly = bx(1), Lz = bx(2);
    double x_lo = rank       * Lx / nranks;
    double x_hi = (rank + 1) * Lx / nranks;

    // Identify this rank's local atoms
    std::vector<double> local_vec;
    for (ssize_t i = 0; i < n_total; ++i) {
        double x = pos(i, 0);
        if (x >= x_lo && x < x_hi) {
            local_vec.push_back(pos(i, 0));
            local_vec.push_back(pos(i, 1));
            local_vec.push_back(pos(i, 2));
        }
    }
    int n_local = (int)(local_vec.size() / 3);

    // MPI exchange in x direction
    int left_rank  = (rank - 1 + nranks) % nranks;
    int right_rank = (rank + 1) % nranks;

    // Atoms near the right boundary of this rank are sent to the right neighbour,
    // which uses them as its left-side ghosts. Vice versa for the left boundary.
    std::vector<double> send_right, send_left;
    for (int k = 0; k < n_local; ++k) {
        double x = local_vec[k*3];
        if (x >= x_hi - r_list) {
            send_right.push_back(local_vec[k*3]);
            send_right.push_back(local_vec[k*3 + 1]);
            send_right.push_back(local_vec[k*3 + 2]);
        }
        if (x < x_lo + r_list) {
            send_left.push_back(local_vec[k*3]);
            send_left.push_back(local_vec[k*3 + 1]);
            send_left.push_back(local_vec[k*3 + 2]);
        }
    }

    int n_send_right = (int)(send_right.size() / 3);
    int n_send_left  = (int)(send_left.size()  / 3);
    int n_recv_left  = 0, n_recv_right = 0;

    // Exchange counts (send right -> receive from left; send left -> receive from right)
    MPI_Sendrecv(&n_send_right, 1, MPI_INT, right_rank, 0,
                 &n_recv_left,  1, MPI_INT, left_rank,  0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(&n_send_left,  1, MPI_INT, left_rank,  1,
                 &n_recv_right, 1, MPI_INT, right_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Exchange positions
    std::vector<double> recv_left(n_recv_left * 3);
    std::vector<double> recv_right(n_recv_right * 3);

    MPI_Sendrecv(send_right.data(), n_send_right * 3, MPI_DOUBLE, right_rank, 2,
                 recv_left.data(),  n_recv_left  * 3, MPI_DOUBLE, left_rank,  2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_left.data(),  n_send_left  * 3, MPI_DOUBLE, left_rank,  3,
                 recv_right.data(), n_recv_right * 3, MPI_DOUBLE, right_rank, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Apply PBC x-shift for the periodic boundary pair (rank 0 <-> rank nranks-1).
    // Atoms received from the right-most rank by rank 0 are at x ~ Lx, they need
    // shifting to x ~ something so they appear to the left of rank 0's domain.
    double shift_from_left  = (rank == 0)          ? -Lx : 0.0;
    double shift_from_right = (rank == nranks - 1) ? +Lx : 0.0;

    std::vector<double> x_ghost_vec;
    for (int k = 0; k < n_recv_left; ++k) {
        x_ghost_vec.push_back(recv_left[k*3]     + shift_from_left);
        x_ghost_vec.push_back(recv_left[k*3 + 1]);
        x_ghost_vec.push_back(recv_left[k*3 + 2]);
    }
    for (int k = 0; k < n_recv_right; ++k) {
        x_ghost_vec.push_back(recv_right[k*3]     + shift_from_right);
        x_ghost_vec.push_back(recv_right[k*3 + 1]);
        x_ghost_vec.push_back(recv_right[k*3 + 2]);
    }
    int n_x_ghosts = (int)(x_ghost_vec.size() / 3);

    // y/z PBC ghosts
    // Generated from local atoms and from the x-MPI ghosts (to cover corner atoms
    // that are near both an x-rank boundary and a y/z PBC boundary).
    std::vector<double> ghost_vec = x_ghost_vec; 
    add_yz_pbc_ghosts(local_vec,   n_local,   Ly, Lz, r_list, ghost_vec);
    add_yz_pbc_ghosts(x_ghost_vec, n_x_ghosts, Ly, Lz, r_list, ghost_vec);

    ssize_t n_ghost = (ssize_t)(ghost_vec.size() / 3);

    py::array_t<double> local_out({static_cast<ssize_t>(n_local), static_cast<ssize_t>(3)});
    py::array_t<double> ghost_out({n_ghost, static_cast<ssize_t>(3)});

    if (n_local > 0)
        std::memcpy(local_out.mutable_data(), local_vec.data(),  n_local * 3 * sizeof(double));
    if (n_ghost > 0)
        std::memcpy(ghost_out.mutable_data(), ghost_vec.data(),  n_ghost * 3 * sizeof(double));

    return std::make_tuple(local_out, ghost_out);
}

// Unlike the serial version, ghost_map[g] > i filter is not needed as with MPI, 
// ghost atoms come from other ranks, so each (local_i, ghost_g) pair exists 
// only on this rank, so no double counting is possible.
std::tuple<py::array_t<int>, py::array_t<int>>
build_neighbour_list_mpi(const py::array_t<double>& positions,
    const py::array_t<double>& ghost_positions,
    double r_list) {

    auto pos  = positions.unchecked<2>();
    auto gpos = ghost_positions.unchecked<2>();
    ssize_t n_real  = pos.shape(0);
    ssize_t n_ghost = gpos.shape(0);
    double r_list_sq = r_list * r_list;

    std::vector<int> pair_i_vec;
    std::vector<int> pair_j_vec;

    for (ssize_t i = 0; i < n_real; ++i) {
        for (ssize_t j = i + 1; j < n_real; ++j) {
            double dx = pos(j, 0) - pos(i, 0);
            double dy = pos(j, 1) - pos(i, 1);
            double dz = pos(j, 2) - pos(i, 2);
            if (dx*dx + dy*dy + dz*dz < r_list_sq) {
                pair_i_vec.push_back(static_cast<int>(i));
                pair_j_vec.push_back(static_cast<int>(j));
            }
        }
    }

    for (ssize_t i = 0; i < n_real; ++i) {
        for (ssize_t g = 0; g < n_ghost; ++g) {
            double dx = gpos(g, 0) - pos(i, 0);
            double dy = gpos(g, 1) - pos(i, 1);
            double dz = gpos(g, 2) - pos(i, 2);
            if (dx*dx + dy*dy + dz*dz < r_list_sq) {
                pair_i_vec.push_back(static_cast<int>(i));
                pair_j_vec.push_back(static_cast<int>(n_real + g));
            }
        }
    }

    py::array_t<int> pair_i(pair_i_vec.size(), pair_i_vec.data());
    py::array_t<int> pair_j(pair_j_vec.size(), pair_j_vec.data());
    return std::make_tuple(pair_i, pair_j);
}

// For real-ghost pairs, force is applied only to the local real atom i.
// The ghost atom's rank will compute and apply the opposite force to
// its own local atom — so no MPI communication is needed.
// For energy, real-real pairs contribute a full pair_energy counted once, on
// the rank that owns both. Real-ghost pairs contribute half of pair_energy
// as both this rank and the ghost atom's rank independently compute
// the same interaction. MPI_Allreduce (in the python part) is used to get 
// global potential energy.
std::tuple<py::array_t<double>, double>
compute_forces_mpi(const py::array_t<double>& positions,
    const py::array_t<double>& ghost_positions,
    const py::array_t<int>& pair_i,
    const py::array_t<int>& pair_j,
    double r_cut, double sigma, double epsilon) {

    auto pos  = positions.unchecked<2>();
    auto gpos = ghost_positions.unchecked<2>();
    auto pi   = pair_i.unchecked<1>();
    auto pj   = pair_j.unchecked<1>();

    ssize_t n_real  = pos.shape(0);
    ssize_t n_pairs = pi.shape(0);

    py::array_t<double> forces_np({n_real, static_cast<ssize_t>(3)});
    std::memset(forces_np.mutable_data(), 0, n_real * 3 * sizeof(double));
    auto forces = forces_np.mutable_unchecked<2>();

    double r_cut_sq = r_cut * r_cut;
    double sigma_sq = sigma * sigma;

    double inv_rc2  = sigma_sq / r_cut_sq;
    double inv_rc6  = inv_rc2 * inv_rc2 * inv_rc2;
    double inv_rc12 = inv_rc6 * inv_rc6;
    double v_shift  = 4.0 * epsilon * (inv_rc12 - inv_rc6);

    double energy = 0.0;

    for (ssize_t k = 0; k < n_pairs; ++k) {
        int i = pi(k);
        int j = pj(k);

        bool is_ghost = (j >= n_real);

        double pj_x, pj_y, pj_z;
        if (!is_ghost) {
            pj_x = pos(j, 0);  pj_y = pos(j, 1);  pj_z = pos(j, 2);
        } else {
            ssize_t g = j - n_real;
            pj_x = gpos(g, 0);  pj_y = gpos(g, 1);  pj_z = gpos(g, 2);
        }

        double dx = pj_x - pos(i, 0);
        double dy = pj_y - pos(i, 1);
        double dz = pj_z - pos(i, 2);

        double dist_sq = dx*dx + dy*dy + dz*dz;
        if (dist_sq <= 0.0 || dist_sq >= r_cut_sq) continue;

        double inv_r2  = sigma_sq / dist_sq;
        double inv_r6  = inv_r2 * inv_r2 * inv_r2;
        double inv_r12 = inv_r6 * inv_r6;

        double pair_energy = epsilon * 4.0 * (inv_r12 - inv_r6) - v_shift;
        double f_over_r    = epsilon * 24.0 * (2.0 * inv_r12 - inv_r6) / dist_sq;

        double fx = f_over_r * dx;
        double fy = f_over_r * dy;
        double fz = f_over_r * dz;

        forces(i, 0) -= fx;
        forces(i, 1) -= fy;
        forces(i, 2) -= fz;

        if (!is_ghost) { // so real-real pair
            
            forces(j, 0) += fx;
            forces(j, 1) += fy;
            forces(j, 2) += fz;
            energy += pair_energy;
        } else { // real-ghost pair
            energy += 0.5 * pair_energy;
        }
    }

    return std::make_tuple(forces_np, energy);
}


PYBIND11_MODULE(_lj_cpp_mpi, m) {
    // MPI lifecycle
    m.def("mpi_init",     &mpi_init,     "Initialise MPI (idempotent)");
    m.def("mpi_finalize", &mpi_finalize, "Finalise MPI");
    m.def("mpi_rank",     &mpi_rank,     "Return this rank's index");
    m.def("mpi_size",     &mpi_size,     "Return total number of ranks");

    // Serial path
    m.def("check_rebuild",         &check_rebuild,         "Check if neighbour list needs rebuild");
    m.def("build_ghost_atoms",     &build_ghost_atoms,     "Build ghost atoms for PBC (serial)");
    m.def("update_ghost_positions",&update_ghost_positions,"Update ghost positions from current real positions");
    m.def("build_neighbour_list",  &build_neighbour_list,  "Build neighbour list with ghost atoms (serial)");
    m.def("compute_forces",        &compute_forces,        "Compute LJ forces using ghost atoms (serial)");

    // MPI path
    m.def("build_ghost_atoms_mpi",    &build_ghost_atoms_mpi,    "Build local atoms and ghost atoms via MPI halo exchange");
    m.def("build_neighbour_list_mpi", &build_neighbour_list_mpi, "Build neighbour list for MPI domain decomposition");
    m.def("compute_forces_mpi",       &compute_forces_mpi,       "Compute local LJ forces for MPI");
}
