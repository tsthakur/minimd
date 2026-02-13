! Fortran implementations of neighbour list and LJ force computation.
! Compile: cd minimd/backends/fortran && python -m numpy.f2py -c -m _lj_fortran lj_fortran.f90

subroutine build_neighbour_list(positions, box, r_cut, r_skin, &
                                n_atoms, max_pairs, &
                                pairs_i, pairs_j, n_pairs)
    implicit none
    integer, intent(in) :: n_atoms, max_pairs
    double precision, intent(in) :: positions(n_atoms, 3)
    double precision, intent(in) :: box(3)
    double precision, intent(in) :: r_cut, r_skin
    integer, intent(out) :: pairs_i(max_pairs), pairs_j(max_pairs)
    integer, intent(out) :: n_pairs
!f2py integer, intent(hide), depend(positions) :: n_atoms = shape(positions, 0)

    double precision :: r_list, r_list_sq, dx, dy, dz, dist_sq
    integer :: i, j

    r_list = r_cut + r_skin
    r_list_sq = r_list * r_list
    n_pairs = 0

    do i = 1, n_atoms
        do j = i + 1, n_atoms
            dx = positions(j, 1) - positions(i, 1)
            dy = positions(j, 2) - positions(i, 2)
            dz = positions(j, 3) - positions(i, 3)

            dx = dx - box(1) * nint(dx / box(1))
            dy = dy - box(2) * nint(dy / box(2))
            dz = dz - box(3) * nint(dz / box(3))

            dist_sq = dx*dx + dy*dy + dz*dz

            if (dist_sq < r_list_sq) then
                n_pairs = n_pairs + 1
                if (n_pairs <= max_pairs) then
                    pairs_i(n_pairs) = i - 1
                    pairs_j(n_pairs) = j - 1
                end if
            end if
        end do
    end do
end subroutine


subroutine check_needs_rebuild(positions, last_positions, n_atoms, &
                                r_skin, needs_rebuild)
    implicit none
    integer, intent(in) :: n_atoms
    double precision, intent(in) :: positions(n_atoms, 3)
    double precision, intent(in) :: last_positions(n_atoms, 3)
    double precision, intent(in) :: r_skin
    integer, intent(out) :: needs_rebuild
!f2py integer, intent(hide), depend(positions) :: n_atoms = shape(positions, 0)

    double precision :: half_skin_sq, dx, dy, dz, disp_sq
    integer :: i

    half_skin_sq = 0.25d0 * r_skin * r_skin
    needs_rebuild = 0

    do i = 1, n_atoms
        dx = positions(i, 1) - last_positions(i, 1)
        dy = positions(i, 2) - last_positions(i, 2)
        dz = positions(i, 3) - last_positions(i, 3)
        disp_sq = dx*dx + dy*dy + dz*dz
        if (disp_sq > half_skin_sq) then
            needs_rebuild = 1
            return
        end if
    end do
end subroutine


subroutine compute_lj_forces(positions, box, pairs_i, pairs_j, n_pairs, &
                              n_atoms, r_cut, sigma, epsilon, forces, energy)
    implicit none
    integer, intent(in) :: n_atoms, n_pairs
    double precision, intent(in) :: positions(n_atoms, 3)
    double precision, intent(in) :: box(3)
    integer, intent(in) :: pairs_i(n_pairs), pairs_j(n_pairs)
    double precision, intent(in) :: r_cut, sigma, epsilon
    double precision, intent(out) :: forces(n_atoms, 3)
    double precision, intent(out) :: energy
!f2py integer, intent(hide), depend(positions) :: n_atoms = shape(positions, 0)
!f2py integer, intent(hide), depend(pairs_i) :: n_pairs = shape(pairs_i, 0)

    double precision :: r_cut_sq, sigma_sq, inv_rc2, inv_rc6, inv_rc12, v_shift
    double precision :: dx, dy, dz, dist_sq, inv_r2, inv_r6, inv_r12
    double precision :: f_over_r, fx, fy, fz
    integer :: p, i, j

    forces = 0.0d0
    energy = 0.0d0

    if (n_pairs == 0) return

    r_cut_sq = r_cut * r_cut
    sigma_sq = sigma * sigma

    inv_rc2 = sigma_sq / r_cut_sq
    inv_rc6 = inv_rc2 * inv_rc2 * inv_rc2
    inv_rc12 = inv_rc6 * inv_rc6
    v_shift = 4.0d0 * (inv_rc12 - inv_rc6)

    do p = 1, n_pairs
        i = pairs_i(p) + 1
        j = pairs_j(p) + 1

        dx = positions(j, 1) - positions(i, 1)
        dy = positions(j, 2) - positions(i, 2)
        dz = positions(j, 3) - positions(i, 3)

        dx = dx - box(1) * nint(dx / box(1))
        dy = dy - box(2) * nint(dy / box(2))
        dz = dz - box(3) * nint(dz / box(3))

        dist_sq = dx*dx + dy*dy + dz*dz

        if (dist_sq <= 0.0d0 .or. dist_sq >= r_cut_sq) cycle

        inv_r2 = sigma_sq / dist_sq
        inv_r6 = inv_r2 * inv_r2 * inv_r2
        inv_r12 = inv_r6 * inv_r6

        energy = energy + epsilon * (4.0d0 * (inv_r12 - inv_r6) - v_shift)

        f_over_r = epsilon * 24.0d0 * (2.0d0 * inv_r12 - inv_r6) / dist_sq

        fx = f_over_r * dx
        fy = f_over_r * dy
        fz = f_over_r * dz

        forces(j, 1) = forces(j, 1) + fx
        forces(j, 2) = forces(j, 2) + fy
        forces(j, 3) = forces(j, 3) + fz
        forces(i, 1) = forces(i, 1) - fx
        forces(i, 2) = forces(i, 2) - fy
        forces(i, 3) = forces(i, 3) - fz
    end do
end subroutine
