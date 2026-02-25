#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <cstring>

namespace py = pybind11;

// Only build if an atom goes outside of the r_skin shell
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

// Ghost atom approach insted of minimum image convention
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

PYBIND11_MODULE(_lj_cpp_mpi, m) {
    m.def("check_rebuild", &check_rebuild, "Check if neighbour list needs rebuild");
    m.def("build_ghost_atoms", &build_ghost_atoms, "Build ghost atoms for PBC");
    m.def("update_ghost_positions", &update_ghost_positions,
          "Update ghost positions from current real positions");
    m.def("build_neighbour_list", &build_neighbour_list,
          "Build neighbour list with ghost atoms");
    m.def("compute_forces", &compute_forces,
          "Compute LJ forces using ghost atoms");
}
