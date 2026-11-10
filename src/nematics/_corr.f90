program nematics_corr
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64, error_unit
    implicit none

    integer, parameter :: path_len = 4096
    integer, parameter :: line_len = 1024
    integer, parameter :: max_radii = 32
    integer, parameter :: max_position_atoms = 32
    integer(int32), parameter :: endian_marker = 16909060_int32
    real(real64), parameter :: pi = acos(-1.0_real64)

    character(len=path_len) :: trajectory_path = ''
    character(len=path_len) :: director_path = ''
    character(len=path_len) :: vector_path = ''
    character(len=path_len) :: parallel_path = ''
    character(len=path_len) :: perpendicular_path = ''
    character(len=path_len) :: radial_path = ''
    character(len=path_len) :: metadata_path = ''
    character(len=path_len) :: performance_path = ''
    character(len=256) :: molecule_name = ''
    character(len=256) :: dataset_name = ''
    character(len=256) :: run_id = ''
    real(real64) :: temperature_k = 0.0_real64
    integer :: molecules = 0
    integer :: atoms_per_molecule = 0
    integer :: frame_start = 1
    integer :: frame_stop = 0
    integer :: frame_stride = 1
    integer :: correlation_rank = 0
    integer :: bins = 200
    integer :: radius_count = 0
    real(real64) :: cylinder_radii(max_radii) = 0.0_real64
    integer :: position_atom_count = 0
    integer :: position_atoms(max_position_atoms) = 0
    real(real64) :: axial_max_nm = 5.0_real64
    real(real64) :: perpendicular_max_nm = 5.0_real64
    real(real64) :: radial_max_nm = 5.0_real64
    real(real64) :: slab_half_thickness_nm = 0.2_real64

    namelist /corr_job/ trajectory_path, director_path, vector_path, &
        parallel_path, perpendicular_path, radial_path, metadata_path, &
        performance_path, molecule_name, dataset_name, run_id, temperature_k, &
        molecules, atoms_per_molecule, frame_start, frame_stop, frame_stride, &
        correlation_rank, bins, radius_count, cylinder_radii, axial_max_nm, &
        perpendicular_max_nm, radial_max_nm, slab_half_thickness_nm, &
        position_atom_count, position_atoms

    character(len=path_len) :: namelist_path
    character(len=line_len) :: line
    character(len=32) :: atom_name
    integer :: argc, cfg_unit, trajectory_unit, director_unit, vector_unit
    integer :: parallel_unit, metadata_unit
    integer :: performance_unit, ios, natoms, expected_atoms, atom_line, atom_id
    integer :: molecule_id, local_atom, frame_seen, frames_processed
    integer :: i, j, k, ir, ib
    integer(int64) :: director_frames, director_molecules, vector_frames
    integer(int64) :: vector_molecules, clock_start, clock_end, clock_rate
    integer(int64) :: pairs_visited, images_visited
    integer(int32) :: image_shifts(3, 27)
    integer(int64), allocatable :: raw_parallel(:, :), raw_perpendicular(:), raw_radial(:)
    real(real64), allocatable :: sum_parallel(:, :), sum_perpendicular(:), sum_radial(:)
    real(real64), allocatable :: g0_parallel(:, :), grank_parallel(:, :)
    real(real64), allocatable :: g0_perpendicular(:), grank_perpendicular(:)
    real(real64), allocatable :: g0_radial(:), grank_radial(:)
    real(real64), allocatable :: atom_positions(:, :, :), molecular_positions(:, :)
    real(real64), allocatable :: axis64(:, :)
    real(real32), allocatable :: axes(:, :)
    real(real64) :: box_parameters(6), box_nm(3), xyz(3), director_record(12)
    real(real64) :: reference(3), displacement(3), rij0(3), rij(3)
    real(real64) :: rho, dz, dr_perpendicular, dr_radial
    real(real64) :: rijsq, distance, rz, abs_rz, rperp, norm, pair_dot, weight
    real(real64) :: cutoff_sq, elapsed_seconds, frames_per_second
    logical :: selected

    argc = command_argument_count()
    if (argc /= 1) call fatal('usage: corr <job.nml>')
    call get_command_argument(1, namelist_path)

    open(newunit=cfg_unit, file=trim(namelist_path), status='old', action='read', iostat=ios)
    if (ios /= 0) call fatal('cannot open the corr namelist')
    read(cfg_unit, nml=corr_job, iostat=ios)
    close(cfg_unit)
    if (ios /= 0) call fatal('cannot parse the corr namelist')
    call validate_configuration()

    call system_clock(count=clock_start, count_rate=clock_rate)
    expected_atoms = molecules * atoms_per_molecule
    allocate(atom_positions(3, atoms_per_molecule, molecules))
    allocate(molecular_positions(3, molecules))
    allocate(raw_parallel(bins, radius_count), sum_parallel(bins, radius_count))
    allocate(g0_parallel(bins, radius_count), grank_parallel(bins, radius_count))
    allocate(raw_perpendicular(bins), sum_perpendicular(bins))
    allocate(g0_perpendicular(bins), grank_perpendicular(bins))
    allocate(raw_radial(bins), sum_radial(bins), g0_radial(bins), grank_radial(bins))
    raw_parallel = 0_int64
    sum_parallel = 0.0_real64
    g0_parallel = 0.0_real64
    grank_parallel = 0.0_real64
    raw_perpendicular = 0_int64
    sum_perpendicular = 0.0_real64
    g0_perpendicular = 0.0_real64
    grank_perpendicular = 0.0_real64
    raw_radial = 0_int64
    sum_radial = 0.0_real64
    g0_radial = 0.0_real64
    grank_radial = 0.0_real64
    pairs_visited = 0_int64
    images_visited = 0_int64

    if (correlation_rank > 0) then
        allocate(axes(3, molecules), axis64(3, molecules))
        axis64 = 0.0_real64
        call open_vector_file(vector_unit, vector_frames, vector_molecules)
    else
        vector_frames = 0_int64
        vector_molecules = int(molecules, int64)
    end if
    call open_director_file(director_unit, director_frames, director_molecules)
    if (director_molecules /= int(molecules, int64)) then
        call fatal('director molecule count differs from trajectory configuration')
    end if
    if (correlation_rank > 0) then
        if (vector_frames /= director_frames) call fatal('vector and director frame counts differ')
        if (vector_molecules /= int(molecules, int64)) then
            call fatal('vector molecule count differs from trajectory configuration')
        end if
    end if

    open(newunit=trajectory_unit, file=trim(trajectory_path), status='old', &
        action='read', form='formatted', iostat=ios)
    if (ios /= 0) call fatal('cannot open the Tinker ARC trajectory')

    call make_image_shifts(image_shifts)
    dz = axial_max_nm / real(bins, real64)
    dr_perpendicular = perpendicular_max_nm / real(bins, real64)
    dr_radial = radial_max_nm / real(bins, real64)
    cutoff_sq = 2.0_real64 * max(axial_max_nm, perpendicular_max_nm, radial_max_nm)**2
    frame_seen = 0
    frames_processed = 0

    frame_loop: do
        call read_next_nonempty_line(trajectory_unit, line, ios)
        if (ios < 0) exit frame_loop
        if (ios > 0) call fatal('error while reading an ARC frame header')
        read(line, *, iostat=ios) natoms
        if (ios /= 0 .or. natoms /= expected_atoms) call fatal('unexpected ARC atom count')

        read(trajectory_unit, '(A)', iostat=ios) line
        if (ios /= 0) call fatal('missing ARC periodic-box line')
        read(line, *, iostat=ios) box_parameters
        if (ios /= 0) call fatal('invalid ARC periodic-box line')
        if (maxval(abs(box_parameters(4:6) - 90.0_real64)) > 1.0e-6_real64) then
            call fatal('corr currently requires an orthorhombic periodic box')
        end if
        box_nm = box_parameters(1:3) / 10.0_real64

        frame_seen = frame_seen + 1
        selected = frame_seen >= frame_start
        if (frame_stop > 0) selected = selected .and. frame_seen <= frame_stop
        selected = selected .and. mod(frame_seen - frame_start, frame_stride) == 0

        do atom_line = 1, natoms
            read(trajectory_unit, '(A)', iostat=ios) line
            if (ios /= 0) call fatal('unexpected end of ARC atom records')
            if (.not. selected) cycle
            read(line, *, iostat=ios) atom_id, atom_name, xyz
            if (ios /= 0) call fatal('invalid ARC atom record')
            if (atom_id < 1 .or. atom_id > expected_atoms) call fatal('invalid ARC atom index')
            molecule_id = (atom_id - 1) / atoms_per_molecule + 1
            local_atom = mod(atom_id - 1, atoms_per_molecule) + 1
            atom_positions(:, local_atom, molecule_id) = xyz / 10.0_real64
        end do

        if (selected) then
            frames_processed = frames_processed + 1
            if (frames_processed > director_frames) call fatal('trajectory has more selected frames than inputs')
            read(director_unit, iostat=ios) director_record
            if (ios /= 0) call fatal('cannot read a director frame')
            if (correlation_rank > 0) then
                read(vector_unit, iostat=ios) axes
                if (ios /= 0) call fatal('cannot read an orientation-vector frame')
                axis64 = real(axes, real64)
            end if

            call calculate_centers(atom_positions, box_nm, molecular_positions)
            rho = real(molecules, real64) * real(molecules - 1, real64) / product(box_nm)

            do i = 1, molecules - 1
                do j = i + 1, molecules
                    pairs_visited = pairs_visited + 1_int64
                    if (correlation_rank == 0) then
                        weight = 1.0_real64
                    else
                        pair_dot = dot_product(axis64(:, i), axis64(:, j))
                        pair_dot = max(-1.0_real64, min(1.0_real64, pair_dot))
                        if (correlation_rank == 1) then
                            weight = pair_dot
                        else
                            weight = 0.5_real64 * (3.0_real64 * pair_dot * pair_dot - 1.0_real64)
                        end if
                    end if
                    rij0 = molecular_positions(:, j) - molecular_positions(:, i)

                    do k = 1, 27
                        images_visited = images_visited + 1_int64
                        rij = rij0 + real(image_shifts(:, k), real64) * box_nm
                        rijsq = dot_product(rij, rij)
                        if (rijsq > cutoff_sq) cycle
                        rz = dot_product(rij, director_record(4:6))
                        abs_rz = abs(rz)
                        rperp = sqrt(max(0.0_real64, rijsq - rz * rz))

                        if (abs_rz < axial_max_nm) then
                            ib = 1 + int(abs_rz / dz)
                            if (ib >= 1 .and. ib <= bins) then
                                do ir = 1, radius_count
                                    if (rperp < cylinder_radii(ir)) then
                                        norm = rho * pi * cylinder_radii(ir)**2 * dz
                                        raw_parallel(ib, ir) = raw_parallel(ib, ir) + 1_int64
                                        sum_parallel(ib, ir) = sum_parallel(ib, ir) + weight
                                        g0_parallel(ib, ir) = g0_parallel(ib, ir) + 1.0_real64 / norm
                                        grank_parallel(ib, ir) = grank_parallel(ib, ir) + weight / norm
                                    end if
                                end do
                            end if
                        end if

                        if (abs_rz < slab_half_thickness_nm .and. &
                            rperp < perpendicular_max_nm) then
                            ib = 1 + int(rperp / dr_perpendicular)
                            if (ib >= 1 .and. ib <= bins) then
                                raw_perpendicular(ib) = raw_perpendicular(ib) + 1_int64
                                sum_perpendicular(ib) = sum_perpendicular(ib) + weight
                                norm = 2.0_real64 * pi * rho * slab_half_thickness_nm * &
                                    dr_perpendicular * rperp
                                if (norm > 0.0_real64) then
                                    g0_perpendicular(ib) = g0_perpendicular(ib) + 1.0_real64 / norm
                                    grank_perpendicular(ib) = grank_perpendicular(ib) + weight / norm
                                end if
                            end if
                        end if

                        if (rijsq < radial_max_nm**2) then
                            distance = sqrt(rijsq)
                            ib = 1 + int(distance / dr_radial)
                            if (ib >= 1 .and. ib <= bins) then
                                raw_radial(ib) = raw_radial(ib) + 1_int64
                                sum_radial(ib) = sum_radial(ib) + weight
                                norm = 2.0_real64 * pi * rho * dr_radial * rijsq
                                if (norm > 0.0_real64) then
                                    g0_radial(ib) = g0_radial(ib) + 1.0_real64 / norm
                                    grank_radial(ib) = grank_radial(ib) + weight / norm
                                end if
                            end if
                        end if
                    end do
                end do
            end do
            if (mod(frames_processed, 10) == 0) then
                write(*, '(A,I0,A,I0)') 'Processed ', frames_processed, &
                    ' selected frames; current ARC frame ', frame_seen
            end if
        end if
        if (frame_stop > 0 .and. frame_seen >= frame_stop) exit frame_loop
    end do frame_loop

    close(trajectory_unit)
    close(director_unit)
    if (correlation_rank > 0) close(vector_unit)
    if (frames_processed /= director_frames) call fatal('selected trajectory frame count differs from inputs')
    if (frames_processed == 0) call fatal('no frames were processed')

    call write_parallel_csv()
    call write_one_dimensional_csv(perpendicular_path, perpendicular_max_nm, &
        raw_perpendicular, sum_perpendicular, g0_perpendicular, grank_perpendicular)
    call write_one_dimensional_csv(radial_path, radial_max_nm, raw_radial, &
        sum_radial, g0_radial, grank_radial)
    call write_metadata()

    call system_clock(count=clock_end)
    if (clock_rate > 0_int64) then
        elapsed_seconds = real(clock_end - clock_start, real64) / real(clock_rate, real64)
    else
        elapsed_seconds = 0.0_real64
    end if
    if (elapsed_seconds > 0.0_real64) then
        frames_per_second = real(frames_processed, real64) / elapsed_seconds
    else
        frames_per_second = 0.0_real64
    end if
    open(newunit=performance_unit, file=trim(performance_path), status='replace', &
        action='write', iostat=ios)
    if (ios /= 0) call fatal('cannot create performance CSV')
    write(performance_unit, '(A)') &
        'run_id,rank,frames,molecules,pairs_visited,images_visited,wall_seconds,frames_per_second'
    write(performance_unit, '(A,",",I0,",",I0,",",I0,",",I0,",",I0,",",F0.6,",",F0.6)') &
        trim(run_id), correlation_rank, frames_processed, molecules, pairs_visited, &
        images_visited, elapsed_seconds, frames_per_second
    close(performance_unit)
    write(*, '(A,I0,A,I0)') 'Completed corr rank ', correlation_rank, &
        ' for frames: ', frames_processed

contains

    subroutine validate_configuration()
        integer :: idx
        if (len_trim(trajectory_path) == 0) call fatal('trajectory_path is empty')
        if (len_trim(director_path) == 0) call fatal('director_path is empty')
        if (correlation_rank < 0 .or. correlation_rank > 2) call fatal('rank must be 0, 1, or 2')
        if (correlation_rank > 0 .and. len_trim(vector_path) == 0) then
            call fatal('vector_path is required for rank 1 or 2')
        end if
        if (molecules <= 1 .or. atoms_per_molecule <= 0) call fatal('invalid molecular dimensions')
        if (frame_start < 1 .or. frame_stride < 1 .or. frame_stop < 0) call fatal('invalid frame selection')
        if (frame_stop > 0 .and. frame_stop < frame_start) call fatal('frame stop precedes start')
        if (bins <= 0) call fatal('bins must be positive')
        if (radius_count < 1 .or. radius_count > max_radii) call fatal('invalid radius_count')
        do idx = 1, radius_count
            if (cylinder_radii(idx) <= 0.0_real64) call fatal('cylinder radii must be positive')
        end do
        if (position_atom_count < 1 .or. position_atom_count > max_position_atoms) then
            call fatal('invalid position_atom_count')
        end if
        do idx = 1, position_atom_count
            if (position_atoms(idx) < 1 .or. position_atoms(idx) > atoms_per_molecule) then
                call fatal('a position atom lies outside the molecule')
            end if
        end do
        if (axial_max_nm <= 0.0_real64 .or. perpendicular_max_nm <= 0.0_real64 .or. &
            radial_max_nm <= 0.0_real64 .or. slab_half_thickness_nm <= 0.0_real64) then
            call fatal('histogram lengths must be positive')
        end if
    end subroutine validate_configuration

    subroutine open_vector_file(unit_number, frames, nmolecules)
        integer, intent(out) :: unit_number
        integer(int64), intent(out) :: frames, nmolecules
        character(len=8) :: magic
        integer(int32) :: version, marker, bytes, components, header_size
        open(newunit=unit_number, file=trim(vector_path), status='old', action='read', &
            access='stream', form='unformatted', iostat=ios)
        if (ios /= 0) call fatal('cannot open vector input')
        read(unit_number) magic, version, marker, bytes, frames, nmolecules, components, header_size
        if (magic /= 'NEMAXIS1' .or. version /= 1 .or. marker /= endian_marker .or. &
            bytes /= 4 .or. components /= 3 .or. header_size /= 44) then
            call fatal('unsupported vector input header')
        end if
    end subroutine open_vector_file

    subroutine open_director_file(unit_number, frames, nmolecules)
        integer, intent(out) :: unit_number
        integer(int64), intent(out) :: frames
        integer(int64), intent(out) :: nmolecules
        character(len=8) :: magic
        integer(int32) :: version, marker, bytes, eigenvalues, vector_values
        integer(int32) :: record_values, header_size
        nmolecules = int(molecules, int64)
        open(newunit=unit_number, file=trim(director_path), status='old', action='read', &
            access='stream', form='unformatted', iostat=ios)
        if (ios /= 0) call fatal('cannot open director input')
        read(unit_number) magic, version, marker, bytes, frames, eigenvalues, &
            vector_values, record_values, header_size
        if (magic /= 'NEMORDR1' .or. version /= 1 .or. marker /= endian_marker .or. &
            bytes /= 8 .or. eigenvalues /= 3 .or. vector_values /= 9 .or. &
            record_values /= 12 .or. header_size /= 44) then
            call fatal('unsupported director input header')
        end if
    end subroutine open_director_file

    subroutine calculate_centers(atom_xyz, box, centers)
        real(real64), intent(in) :: atom_xyz(:, :, :), box(3)
        real(real64), intent(out) :: centers(:, :)
        integer :: mol, atom_index, atom
        do mol = 1, molecules
            reference = atom_xyz(:, position_atoms(1), mol)
            centers(:, mol) = 0.0_real64
            do atom_index = 1, position_atom_count
                atom = position_atoms(atom_index)
                displacement = atom_xyz(:, atom, mol) - reference
                displacement = displacement - anint(displacement / box) * box
                centers(:, mol) = centers(:, mol) + reference + displacement
            end do
            centers(:, mol) = centers(:, mol) / real(position_atom_count, real64)
            centers(:, mol) = centers(:, mol) - anint(centers(:, mol) / box) * box
        end do
    end subroutine calculate_centers

    subroutine make_image_shifts(shifts)
        integer(int32), intent(out) :: shifts(3, 27)
        integer :: sx, sy, sz, index
        shifts(:, 1) = 0
        index = 1
        do sx = -1, 1
            do sy = -1, 1
                do sz = -1, 1
                    if (sx == 0 .and. sy == 0 .and. sz == 0) cycle
                    index = index + 1
                    shifts(:, index) = [sx, sy, sz]
                end do
            end do
        end do
    end subroutine make_image_shifts

    subroutine write_parallel_csv()
        real(real64) :: center, g0_value, grank_value, conditional
        open(newunit=parallel_unit, file=trim(parallel_path), status='replace', &
            action='write', iostat=ios)
        if (ios /= 0) call fatal('cannot create parallel CSV')
        write(parallel_unit, '(A)') &
            'radius_nm,z_nm,raw_pair_count,raw_p_rank_sum,g0,g_rank,c_rank'
        do ir = 1, radius_count
            do ib = 1, bins
                center = (real(ib, real64) - 0.5_real64) * dz
                g0_value = g0_parallel(ib, ir) / real(frames_processed, real64)
                grank_value = grank_parallel(ib, ir) / real(frames_processed, real64)
                conditional = 0.0_real64
                if (abs(g0_value) > tiny(1.0_real64)) conditional = grank_value / g0_value
                write(parallel_unit, '(F0.8,",",F0.8,",",I0,4(",",ES24.16E3))') &
                    cylinder_radii(ir), center, raw_parallel(ib, ir), &
                    sum_parallel(ib, ir), g0_value, grank_value, conditional
            end do
        end do
        close(parallel_unit)
    end subroutine write_parallel_csv

    subroutine write_one_dimensional_csv(path, maximum, raw, raw_sum, g0, grank)
        character(len=*), intent(in) :: path
        real(real64), intent(in) :: maximum
        integer(int64), intent(in) :: raw(:)
        real(real64), intent(in) :: raw_sum(:), g0(:), grank(:)
        integer :: unit_number, index
        real(real64) :: center, g0_value, grank_value, conditional, width
        width = maximum / real(size(raw), real64)
        open(newunit=unit_number, file=trim(path), status='replace', action='write', iostat=ios)
        if (ios /= 0) call fatal('cannot create one-dimensional CSV')
        write(unit_number, '(A)') 'distance_nm,raw_pair_count,raw_p_rank_sum,g0,g_rank,c_rank'
        do index = 1, size(raw)
            center = (real(index, real64) - 0.5_real64) * width
            g0_value = g0(index) / real(frames_processed, real64)
            grank_value = grank(index) / real(frames_processed, real64)
            conditional = 0.0_real64
            if (abs(g0_value) > tiny(1.0_real64)) conditional = grank_value / g0_value
            write(unit_number, '(F0.8,",",I0,4(",",ES24.16E3))') center, raw(index), &
                raw_sum(index), g0_value, grank_value, conditional
        end do
        close(unit_number)
    end subroutine write_one_dimensional_csv

    subroutine write_metadata()
        open(newunit=metadata_unit, file=trim(metadata_path), status='replace', &
            action='write', iostat=ios)
        if (ios /= 0) call fatal('cannot create metadata YAML')
        write(metadata_unit, '(A)') 'format: nematics-corr-v1'
        write(metadata_unit, '(A)') 'system:'
        write(metadata_unit, '(A,A,A)') "  molecule: '", trim(molecule_name), "'"
        write(metadata_unit, '(A,A,A)') "  dataset: '", trim(dataset_name), "'"
        write(metadata_unit, '(A,A,A)') "  run_id: '", trim(run_id), "'"
        write(metadata_unit, '(A,F12.5)') '  temperature_k: ', temperature_k
        write(metadata_unit, '(A)') 'inputs:'
        write(metadata_unit, '(A,A,A)') "  trajectory: '", trim(trajectory_path), "'"
        write(metadata_unit, '(A,A,A)') "  director: '", trim(director_path), "'"
        if (correlation_rank > 0) write(metadata_unit, '(A,A,A)') &
            "  vectors: '", trim(vector_path), "'"
        write(metadata_unit, '(A)') 'position:'
        write(metadata_unit, '(A)') '  definition: equal_weight_atom_average'
        write(metadata_unit, '(A)', advance='no') '  local_atoms_one_based: ['
        do i = 1, position_atom_count
            if (i > 1) write(metadata_unit, '(A)', advance='no') ', '
            write(metadata_unit, '(I0)', advance='no') position_atoms(i)
        end do
        write(metadata_unit, '(A)') ']'
        write(metadata_unit, '(A)') '  pbc: selected_atoms_reconstructed_relative_to_first'
        write(metadata_unit, '(A)') '  units: nm'
        write(metadata_unit, '(A)') 'calculation:'
        write(metadata_unit, '(A,I0)') '  rank: ', correlation_rank
        write(metadata_unit, '(A,I0)') '  frames: ', frames_processed
        write(metadata_unit, '(A,I0)') '  molecules_per_frame: ', molecules
        write(metadata_unit, '(A)') '  periodic_images: 27'
        write(metadata_unit, '(A)') '  rho: N*(N-1)/V'
        write(metadata_unit, '(A,I0)') '  bins: ', bins
        write(metadata_unit, '(A,F0.8)') '  axial_max_nm: ', axial_max_nm
        write(metadata_unit, '(A,F0.8)') '  perpendicular_max_nm: ', perpendicular_max_nm
        write(metadata_unit, '(A,F0.8)') '  radial_max_nm: ', radial_max_nm
        write(metadata_unit, '(A,F0.8)') '  slab_half_thickness_nm: ', slab_half_thickness_nm
        write(metadata_unit, '(A)') 'normalization:'
        write(metadata_unit, '(A)') '  parallel: rho*pi*R^2*dz'
        write(metadata_unit, '(A)') '  perpendicular: 2*pi*rho*dzz*r*dr'
        write(metadata_unit, '(A)') '  radial: 2*pi*rho*r^2*dr'
        write(metadata_unit, '(A)') '  ensemble: normalized_accumulators_divided_by_frames'
        close(metadata_unit)
    end subroutine write_metadata

    subroutine read_next_nonempty_line(unit_number, text, status)
        integer, intent(in) :: unit_number
        character(len=*), intent(out) :: text
        integer, intent(out) :: status
        do
            read(unit_number, '(A)', iostat=status) text
            if (status /= 0) return
            if (len_trim(text) > 0) return
        end do
    end subroutine read_next_nonempty_line

    subroutine fatal(message)
        character(len=*), intent(in) :: message
        write(error_unit, '(A)') 'corr: ' // trim(message)
        error stop 1
    end subroutine fatal

end program nematics_corr
