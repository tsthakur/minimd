#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <cstring>

namespace py = pybind11;

bool check_rebuild(const py::array_t<float>& positions, 
    const py::array_t<float>& _last_positions, float r_skin) {
    auto pos = positions.unchecked<2>();
    auto last_pos = _last_positions.unchecked<2>();
    
    for (ssize_t i = 0; i < pos.shape(0); ++i) {
        float dx = pos(i, 0) - last_pos(i, 0);
        float dy = pos(i, 1) - last_pos(i, 1);
        float dz = pos(i, 2) - last_pos(i, 2);
        float dist_sq = dx*dx + dy*dy + dz*dz;

        float half_skin = r_skin / 2;
        if (dist_sq > half_skin * half_skin) {
            return true;
        }
    }
    return false;
}

std::tuple<py::array_t<float>, py::array_t<float>> 
    build_neighbour_list(const py::array_t<float>& positions, 
        const py::array_t<float>& box, float r_cut, float r_skin) {
            auto pos = positions.unchecked<2>();
            auto box_u = box.unchecked<1>();
            ssize_t n_particles = pos.shape(0);
            std::vector<int> pair_i_vec;
            std::vector<int> pair_j_vec;

            float r_list = r_cut + r_skin;
            float r_list_sq = r_list * r_list;

            for (ssize_t i = 0; i < n_particles; ++i) {
                for (ssize_t j = i + 1; j < n_particles; ++j) {

                    float dx = pos(i, 0) - pos(j, 0);
                    float dy = pos(i, 1) - pos(j, 1);
                    float dz = pos(i, 2) - pos(j, 2);

                    dx -= box_u(0) * std::round(dx / box_u(0));
                    dy -= box_u(1) * std::round(dy / box_u(1));
                    dz -= box_u(2) * std::round(dz / box_u(2));

                    float dist_sq = dx*dx + dy*dy + dz*dz;

                    if (dist_sq < r_list_sq) {
                        pair_i_vec.push_back(i);
                        pair_j_vec.push_back(j);
                    }
                }
            }
            py::array_t<int> pair_i(pair_i_vec.size(), pair_i_vec.data());
            py::array_t<int> pair_j(pair_j_vec.size(), pair_j_vec.data());
            return std::make_tuple(pair_i, pair_j);
}

std::tuple<py::array_t<float>, float> 
    compute_forces(const py::array_t<float>& positions, 
        const py::array_t<float>& box, const py::array_t<int>& pair_i, 
        const py::array_t<int>& pair_j, float r_cut, float sigma, float epsilon) {
            auto pos = positions.unchecked<2>();
            auto pi = pair_i.unchecked<1>();
            auto pj = pair_j.unchecked<1>();
            auto box_u = box.unchecked<1>();
            ssize_t n_particles = pos.shape(0);
            ssize_t n_pairs = pi.shape(0);
            std::vector<ssize_t> shape = {n_particles, 3};

            py::array_t<float> forces_np(shape);
            std::memset(forces_np.mutable_data(), 0, n_particles * 3 * sizeof(float));
            auto forces = forces_np.mutable_unchecked<2>();

            float r_cut_sq = r_cut * r_cut;
            float sigma_sq = sigma * sigma;

            float inv_rc2 = sigma_sq / r_cut_sq;
            float inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2;
            float inv_rc12 = inv_rc6 * inv_rc6;
            float v_shift = 4.0 * (inv_rc12 - inv_rc6);

            float energy = 0.0;

            for (ssize_t k = 0; k < n_pairs; k++) {
                int i = pi(k);
                int j = pj(k);

                float dx = pos(j, 0) - pos(i, 0);
                float dy = pos(j, 1) - pos(i, 1);
                float dz = pos(j, 2) - pos(i, 2);

                dx -= box_u(0) * std::round(dx / box_u(0));
                dy -= box_u(1) * std::round(dy / box_u(1));
                dz -= box_u(2) * std::round(dz / box_u(2));


                float dist_sq = dx * dx + dy * dy + dz * dz;

            if (dist_sq <= 0.0 || dist_sq >= r_cut_sq) {
                continue;
            }

            float inv_r2 = sigma_sq / dist_sq;
            float inv_r6 = inv_r2 * inv_r2 * inv_r2;
            float inv_r12 = inv_r6 * inv_r6;

            energy += epsilon * (4.0 * (inv_r12 - inv_r6) - v_shift);

            float f_over_r = epsilon * 24.0 * (2.0 * inv_r12 - inv_r6) / dist_sq;

            float fx = f_over_r * dx;
            float fy = f_over_r * dy;
            float fz = f_over_r * dz;

            forces(j, 0) += fx;
            forces(j, 1) += fy;
            forces(j, 2) += fz;
            forces(i, 0) -= fx;
            forces(i, 1) -= fy;
            forces(i, 2) -= fz;
            }
            return std::make_tuple(forces_np, energy);
        }

PYBIND11_MODULE(_lj_cpp_openmp, m) {
    m.def("check_rebuild", &check_rebuild, "Check if neighbour list needs to be built");
    m.def("build_neighbour_list", &build_neighbour_list, "Build neighbour list");
    m.def("compute_forces", &compute_forces, "Compute LJ forces");
}