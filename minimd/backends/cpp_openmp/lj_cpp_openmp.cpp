#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <vector>
#include <cstring>
#include <omp.h>

namespace py = pybind11;

bool check_rebuild(const py::array_t<double>& positions, 
    const py::array_t<double>& _last_positions, double r_skin) {
    auto pos = positions.unchecked<2>();
    auto last_pos = _last_positions.unchecked<2>();
    bool needs_rebuild = false;

    #pragma omp parallel for schedule(static)
    for (ssize_t i = 0; i < pos.shape(0); ++i) {
        double dx = pos(i, 0) - last_pos(i, 0);
        double dy = pos(i, 1) - last_pos(i, 1);
        double dz = pos(i, 2) - last_pos(i, 2);
        double dist_sq = dx*dx + dy*dy + dz*dz;

        double half_skin = r_skin / 2;
        if (dist_sq > half_skin * half_skin) {
            needs_rebuild = true;
        }
    }
    return needs_rebuild;
}

std::tuple<py::array_t<double>, py::array_t<double>> 
    build_neighbour_list(const py::array_t<double>& positions, 
        const py::array_t<double>& box, double r_cut, double r_skin) {
            auto pos = positions.unchecked<2>();
            auto box_u = box.unchecked<1>();
            ssize_t n_particles = pos.shape(0);
            
            double r_list = r_cut + r_skin;
            double r_list_sq = r_list * r_list;
            std::vector<int> pair_i_vec;
            std::vector<int> pair_j_vec;
            
            #pragma omp parallel
            {   
                std::vector<int> local_pair_i, local_pair_j;
                #pragma omp for schedule(dynamic)
                for (ssize_t i = 0; i < n_particles; ++i) {
                    for (ssize_t j = i + 1; j < n_particles; ++j) {

                        double dx = pos(j, 0) - pos(i, 0);
                        double dy = pos(j, 1) - pos(i, 1);
                        double dz = pos(j, 2) - pos(i, 2);

                        dx -= box_u(0) * std::round(dx / box_u(0));
                        dy -= box_u(1) * std::round(dy / box_u(1));
                        dz -= box_u(2) * std::round(dz / box_u(2));

                        double dist_sq = dx*dx + dy*dy + dz*dz;
                        
                        if (dist_sq < r_list_sq) {
                            local_pair_i.push_back(i);
                            local_pair_j.push_back(j);
                        }
                    }
                }
                #pragma omp critical
                {
                    pair_i_vec.insert(pair_i_vec.end(), local_pair_i.begin(), local_pair_i.end());
                    pair_j_vec.insert(pair_j_vec.end(), local_pair_j.begin(), local_pair_j.end());
                }
            }
            py::array_t<int> pair_i(pair_i_vec.size(), pair_i_vec.data());
            py::array_t<int> pair_j(pair_j_vec.size(), pair_j_vec.data());
            return std::make_tuple(pair_i, pair_j);
}

std::tuple<py::array_t<double>, double> 
    compute_forces(const py::array_t<double>& positions, 
        const py::array_t<double>& box, const py::array_t<int>& pair_i, 
        const py::array_t<int>& pair_j, double r_cut, double sigma, double epsilon) {
            auto pos = positions.unchecked<2>();
            auto pi = pair_i.unchecked<1>();
            auto pj = pair_j.unchecked<1>();
            auto box_u = box.unchecked<1>();
            ssize_t n_particles = pos.shape(0);
            ssize_t n_pairs = pi.shape(0);
            std::vector<ssize_t> shape = {n_particles, 3};

            py::array_t<double> forces_np(shape);
            std::memset(forces_np.mutable_data(), 0, n_particles * 3 * sizeof(double));
            auto forces = forces_np.mutable_unchecked<2>();

            double r_cut_sq = r_cut * r_cut;
            double sigma_sq = sigma * sigma;

            double inv_rc2 = sigma_sq / r_cut_sq;
            double inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2;
            double inv_rc12 = inv_rc6 * inv_rc6;
            double v_shift = 4.0 * (inv_rc12 - inv_rc6);

            double energy = 0.0;

            #pragma omp parallel reduction(+:energy)
            {
                std::vector<double> local_f(n_particles * 3, 0.0);

                #pragma omp for
                for (ssize_t k = 0; k < n_pairs; k++) {
                    int i = pi(k);
                    int j = pj(k);

                    double dx = pos(j, 0) - pos(i, 0);
                    double dy = pos(j, 1) - pos(i, 1);
                    double dz = pos(j, 2) - pos(i, 2);

                    dx -= box_u(0) * std::round(dx / box_u(0));
                    dy -= box_u(1) * std::round(dy / box_u(1));
                    dz -= box_u(2) * std::round(dz / box_u(2));

                    double dist_sq = dx * dx + dy * dy + dz * dz;

                    if (dist_sq <= 0.0 || dist_sq >= r_cut_sq) {
                        continue;
                    }

                    double inv_r2 = sigma_sq / dist_sq;
                    double inv_r6 = inv_r2 * inv_r2 * inv_r2;
                    double inv_r12 = inv_r6 * inv_r6;

                    energy += epsilon * (4.0 * (inv_r12 - inv_r6) - v_shift);

                    double f_over_r = epsilon * 24.0 * (2.0 * inv_r12 - inv_r6) / dist_sq;

                    double fx = f_over_r * dx;
                    double fy = f_over_r * dy;
                    double fz = f_over_r * dz;

                    local_f[j*3 + 0] += fx;
                    local_f[j*3 + 1] += fy;
                    local_f[j*3 + 2] += fz;
                    local_f[i*3 + 0] -= fx;
                    local_f[i*3 + 1] -= fy;
                    local_f[i*3 + 2] -= fz;
                }

                #pragma omp critical
                for (ssize_t a = 0; a < n_particles; a++) {
                    forces(a, 0) += local_f[a*3 + 0];
                    forces(a, 1) += local_f[a*3 + 1];
                    forces(a, 2) += local_f[a*3 + 2];
                }
            }
            return std::make_tuple(forces_np, energy);
        }

void set_num_threads(int n) {
    omp_set_num_threads(n);
}

PYBIND11_MODULE(_lj_cpp_openmp, m) {
    m.def("check_rebuild", &check_rebuild, "Check if neighbour list needs to be built");
    m.def("build_neighbour_list", &build_neighbour_list, "Build neighbour list");
    m.def("compute_forces", &compute_forces, "Compute LJ forces");
    m.def("set_num_threads", &set_num_threads, "Set number of OpenMP threads");
}