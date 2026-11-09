program nematics_molecular_axis
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64, error_unit
    implicit none

    integer, parameter :: path_len = 4096
    integer, parameter :: line_len = 1024
    integer(int32), parameter :: format_version = 1_int32
    integer(int32), parameter :: endian_marker = 16909060_int32
    integer(int32), parameter :: dtype_bytes = 4_int32
    integer(int32), parameter :: component_count = 3_int32
    integer(int32), parameter :: header_bytes = 44_int32
    integer(int64), parameter :: frame_count_position = 21_int64
    real(real64), parameter :: pi = acos(-1.0_real64)

    character(len=path_len) :: trajectory_path = ''
    character(len=path_len) :: output_path = ''
    character(len=path_len) :: final_output_path = ''
    character(len=path_len) :: metadata_path = ''
    character(len=256) :: molecule_name = ''
    character(len=256) :: dataset_name = ''
    character(len=256) :: run_id = ''
    character(len=256) :: axis_name = 'molecular-axis'
    real(real64) :: temperature_k = 0.0_real64
    integer :: molecules = 0
    integer :: atoms_per_molecule = 0
    integer :: frame_start = 1
    integer :: frame_stop = 0
    integer :: frame_stride = 1
    integer :: start_atom = 0
    integer :: end_atom = 0
    logical :: use_minimum_image = .true.
    logical :: normalize_vectors = .true.

    namelist /molecular_axis_job/ trajectory_path, output_path, final_output_path, &
        metadata_path, molecule_name, dataset_name, run_id, axis_name, &
        temperature_k, molecules, atoms_per_molecule, frame_start, frame_stop, &
        frame_stride, start_atom, end_atom, use_minimum_image, normalize_vectors

    character(len=path_len) :: namelist_path
    character(len=line_len) :: line
    character(len=32) :: atom_name
    character(len=8) :: magic
    integer :: argc, cfg_unit, trajectory_unit, output_unit, metadata_unit
    integer :: ios, natoms, expected_atoms, atom_line, atom_id
    integer :: molecule_id, local_atom, frame_seen, frames_written
    integer(int64) :: stored_frame_count
    logical :: selected
    real(real64) :: xyz(3), box_parameters(6), cell(3, 3), inverse_cell(3, 3)
    real(real64) :: displacement(3), vector_length
    real(real64), allocatable :: start_position(:, :), end_position(:, :)
    real(real32), allocatable :: frame_axes(:, :)
    logical, allocatable :: start_seen(:), end_seen(:)

    argc = command_argument_count()
    if (argc /= 1) call fatal('usage: molecular-axis <job.nml>')
    call get_command_argument(1, namelist_path)

    open(newunit=cfg_unit, file=trim(namelist_path), status='old', &
        action='read', iostat=ios)
    if (ios /= 0) call fatal('cannot open the molecular-axis namelist')
    read(cfg_unit, nml=molecular_axis_job, iostat=ios)
    close(cfg_unit)
    if (ios /= 0) call fatal('cannot parse the molecular-axis namelist')

    call validate_configuration()
    expected_atoms = molecules * atoms_per_molecule

    allocate(start_position(3, molecules), end_position(3, molecules))
    allocate(frame_axes(3, molecules))
    allocate(start_seen(molecules), end_seen(molecules))

    open(newunit=trajectory_unit, file=trim(trajectory_path), status='old', &
        action='read', form='formatted', iostat=ios)
    if (ios /= 0) call fatal('cannot open the Tinker ARC trajectory')

    open(newunit=output_unit, file=trim(output_path), status='replace', &
        action='readwrite', access='stream', form='unformatted', iostat=ios)
    if (ios /= 0) call fatal('cannot create the binary molecular-axis output')

    magic = 'NEMAXIS1'
    stored_frame_count = 0_int64
    write(output_unit) magic
    write(output_unit) format_version
    write(output_unit) endian_marker
    write(output_unit) dtype_bytes
    write(output_unit) stored_frame_count
    write(output_unit) int(molecules, int64)
    write(output_unit) component_count
    write(output_unit) header_bytes

    frame_seen = 0
    frames_written = 0

    frame_loop: do
        call read_next_nonempty_line(trajectory_unit, line, ios)
        if (ios < 0) exit frame_loop
        if (ios > 0) call fatal('error while reading an ARC frame header')

        read(line, *, iostat=ios) natoms
        if (ios /= 0) call fatal('invalid ARC frame header')
        if (natoms /= expected_atoms) then
            write(error_unit, '(A,I0,A,I0)') &
                'ARC atom count ', natoms, ' does not match configured count ', expected_atoms
            error stop 1
        end if

        read(trajectory_unit, '(A)', iostat=ios) line
        if (ios /= 0) call fatal('missing ARC periodic-box line')
        read(line, *, iostat=ios) box_parameters
        if (ios /= 0) call fatal('invalid ARC periodic-box line')
        call cell_from_lengths_angles(box_parameters, cell, inverse_cell)

        frame_seen = frame_seen + 1
        selected = frame_seen >= frame_start
        if (frame_stop > 0) selected = selected .and. frame_seen <= frame_stop
        selected = selected .and. mod(frame_seen - frame_start, frame_stride) == 0

        if (selected) then
            start_seen = .false.
            end_seen = .false.
        end if

        do atom_line = 1, natoms
            read(trajectory_unit, '(A)', iostat=ios) line
            if (ios /= 0) call fatal('unexpected end of ARC atom records')
            if (.not. selected) cycle

            read(line, *, iostat=ios) atom_id, atom_name, xyz
            if (ios /= 0) call fatal('invalid ARC atom record')
            if (atom_id < 1 .or. atom_id > expected_atoms) then
                call fatal('ARC atom index lies outside the configured system')
            end if

            molecule_id = (atom_id - 1) / atoms_per_molecule + 1
            local_atom = mod(atom_id - 1, atoms_per_molecule) + 1
            if (local_atom == start_atom) then
                start_position(:, molecule_id) = xyz
                start_seen(molecule_id) = .true.
            end if
            if (local_atom == end_atom) then
                end_position(:, molecule_id) = xyz
                end_seen(molecule_id) = .true.
            end if
        end do

        if (selected) then
            if (.not. all(start_seen) .or. .not. all(end_seen)) then
                call fatal('one or more molecules are missing an axis endpoint')
            end if

            do molecule_id = 1, molecules
                displacement = end_position(:, molecule_id) - &
                    start_position(:, molecule_id)
                if (use_minimum_image) then
                    call minimum_image(displacement, cell, inverse_cell)
                end if
                vector_length = sqrt(sum(displacement * displacement))
                if (vector_length <= tiny(1.0_real64)) then
                    call fatal('encountered a zero-length molecular axis')
                end if
                if (normalize_vectors) displacement = displacement / vector_length
                frame_axes(:, molecule_id) = real(displacement, real32)
            end do

            write(output_unit) frame_axes
            frames_written = frames_written + 1
            if (mod(frames_written, 100) == 0) then
                write(*, '(A,I0,A,I0)') 'Wrote ', frames_written, &
                    ' selected frames; current ARC frame ', frame_seen
            end if
        end if

        if (frame_stop > 0 .and. frame_seen >= frame_stop) exit frame_loop
    end do frame_loop

    if (frames_written == 0) call fatal('the frame selection produced no output')

    stored_frame_count = int(frames_written, int64)
    write(output_unit, pos=frame_count_position) stored_frame_count
    close(output_unit)
    close(trajectory_unit)

    open(newunit=metadata_unit, file=trim(metadata_path), status='replace', &
        action='write', form='formatted', iostat=ios)
    if (ios /= 0) call fatal('cannot create the molecular-axis metadata file')
    call write_metadata(metadata_unit, frame_seen, frames_written)
    close(metadata_unit)

    write(*, '(A,I0,A,I0,A)') 'Completed: read ', frame_seen, &
        ' ARC frames and wrote ', frames_written, ' molecular-axis frames.'

contains

    subroutine validate_configuration()
        if (len_trim(trajectory_path) == 0) call fatal('trajectory_path is empty')
        if (len_trim(output_path) == 0) call fatal('output_path is empty')
        if (len_trim(final_output_path) == 0) call fatal('final_output_path is empty')
        if (len_trim(metadata_path) == 0) call fatal('metadata_path is empty')
        if (molecules <= 0) call fatal('molecules must be positive')
        if (atoms_per_molecule <= 0) call fatal('atoms_per_molecule must be positive')
        if (start_atom < 1 .or. start_atom > atoms_per_molecule) then
            call fatal('start_atom is outside the molecule')
        end if
        if (end_atom < 1 .or. end_atom > atoms_per_molecule) then
            call fatal('end_atom is outside the molecule')
        end if
        if (start_atom == end_atom) call fatal('axis endpoints must differ')
        if (frame_start < 1) call fatal('frame_start must be positive')
        if (frame_stop < 0) call fatal('frame_stop must be nonnegative')
        if (frame_stop > 0 .and. frame_stop < frame_start) then
            call fatal('frame_stop must be zero or not less than frame_start')
        end if
        if (frame_stride < 1) call fatal('frame_stride must be positive')
    end subroutine validate_configuration


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


    subroutine cell_from_lengths_angles(parameters, matrix, inverse)
        real(real64), intent(in) :: parameters(6)
        real(real64), intent(out) :: matrix(3, 3), inverse(3, 3)
        real(real64) :: a, b, c, alpha, beta, gamma, sin_gamma, cz_squared

        a = parameters(1)
        b = parameters(2)
        c = parameters(3)
        alpha = parameters(4) * pi / 180.0_real64
        beta = parameters(5) * pi / 180.0_real64
        gamma = parameters(6) * pi / 180.0_real64
        if (a <= 0.0_real64 .or. b <= 0.0_real64 .or. c <= 0.0_real64) then
            call fatal('periodic-box lengths must be positive')
        end if
        sin_gamma = sin(gamma)
        if (abs(sin_gamma) <= 1.0e-12_real64) then
            call fatal('invalid periodic-box gamma angle')
        end if

        matrix = 0.0_real64
        matrix(:, 1) = [a, 0.0_real64, 0.0_real64]
        matrix(:, 2) = [b * cos(gamma), b * sin_gamma, 0.0_real64]
        matrix(1, 3) = c * cos(beta)
        matrix(2, 3) = c * (cos(alpha) - cos(beta) * cos(gamma)) / sin_gamma
        cz_squared = c * c - matrix(1, 3)**2 - matrix(2, 3)**2
        if (cz_squared <= 0.0_real64) call fatal('invalid periodic-box geometry')
        matrix(3, 3) = sqrt(cz_squared)
        call inverse_3x3(matrix, inverse)
    end subroutine cell_from_lengths_angles


    subroutine inverse_3x3(matrix, inverse)
        real(real64), intent(in) :: matrix(3, 3)
        real(real64), intent(out) :: inverse(3, 3)
        real(real64) :: determinant

        determinant = matrix(1, 1) * &
            (matrix(2, 2) * matrix(3, 3) - matrix(2, 3) * matrix(3, 2)) &
            - matrix(1, 2) * &
            (matrix(2, 1) * matrix(3, 3) - matrix(2, 3) * matrix(3, 1)) &
            + matrix(1, 3) * &
            (matrix(2, 1) * matrix(3, 2) - matrix(2, 2) * matrix(3, 1))
        if (abs(determinant) <= tiny(1.0_real64)) then
            call fatal('periodic-box matrix is singular')
        end if

        inverse(1, 1) = (matrix(2, 2) * matrix(3, 3) - &
            matrix(2, 3) * matrix(3, 2)) / determinant
        inverse(1, 2) = (matrix(1, 3) * matrix(3, 2) - &
            matrix(1, 2) * matrix(3, 3)) / determinant
        inverse(1, 3) = (matrix(1, 2) * matrix(2, 3) - &
            matrix(1, 3) * matrix(2, 2)) / determinant
        inverse(2, 1) = (matrix(2, 3) * matrix(3, 1) - &
            matrix(2, 1) * matrix(3, 3)) / determinant
        inverse(2, 2) = (matrix(1, 1) * matrix(3, 3) - &
            matrix(1, 3) * matrix(3, 1)) / determinant
        inverse(2, 3) = (matrix(1, 3) * matrix(2, 1) - &
            matrix(1, 1) * matrix(2, 3)) / determinant
        inverse(3, 1) = (matrix(2, 1) * matrix(3, 2) - &
            matrix(2, 2) * matrix(3, 1)) / determinant
        inverse(3, 2) = (matrix(1, 2) * matrix(3, 1) - &
            matrix(1, 1) * matrix(3, 2)) / determinant
        inverse(3, 3) = (matrix(1, 1) * matrix(2, 2) - &
            matrix(1, 2) * matrix(2, 1)) / determinant
    end subroutine inverse_3x3


    subroutine minimum_image(vector, matrix, inverse)
        real(real64), intent(inout) :: vector(3)
        real(real64), intent(in) :: matrix(3, 3), inverse(3, 3)
        real(real64) :: fractional(3)

        fractional = matmul(inverse, vector)
        fractional = fractional - anint(fractional)
        vector = matmul(matrix, fractional)
    end subroutine minimum_image


    subroutine write_metadata(unit_number, frames_read, frames_stored)
        integer, intent(in) :: unit_number, frames_read, frames_stored

        write(unit_number, '(A)') 'format: nematics-molecular-axis-v1'
        write(unit_number, '(A)') 'binary:'
        write(unit_number, '(A)') "  magic: 'NEMAXIS1'"
        write(unit_number, '(A,I0)') '  header_bytes: ', header_bytes
        write(unit_number, '(A)') '  dtype: float32'
        write(unit_number, '(A)') '  byte_order: native'
        write(unit_number, '(A)') '  order: [frame, molecule, component]'
        write(unit_number, '(A)') '  components: [x, y, z]'
        write(unit_number, '(A)') 'shape:'
        write(unit_number, '(A,I0)') '  frames: ', frames_stored
        write(unit_number, '(A,I0)') '  molecules: ', molecules
        write(unit_number, '(A)') '  components: 3'
        write(unit_number, '(A)') 'source:'
        write(unit_number, '(A,A,A)') "  trajectory: '", trim(trajectory_path), "'"
        write(unit_number, '(A,A,A)') "  molecule: '", trim(molecule_name), "'"
        write(unit_number, '(A,A,A)') "  dataset: '", trim(dataset_name), "'"
        write(unit_number, '(A,A,A)') "  run_id: '", trim(run_id), "'"
        write(unit_number, '(A,F12.5)') '  temperature_k: ', temperature_k
        write(unit_number, '(A,I0)') '  frames_read: ', frames_read
        write(unit_number, '(A)') 'axis:'
        write(unit_number, '(A,A,A)') "  name: '", trim(axis_name), "'"
        write(unit_number, '(A,I0)') '  start_atom_local_one_based: ', start_atom
        write(unit_number, '(A,I0)') '  end_atom_local_one_based: ', end_atom
        if (use_minimum_image) then
            write(unit_number, '(A)') '  minimum_image: true'
        else
            write(unit_number, '(A)') '  minimum_image: false'
        end if
        if (normalize_vectors) then
            write(unit_number, '(A)') '  normalized: true'
        else
            write(unit_number, '(A)') '  normalized: false'
        end if
        if (normalize_vectors) then
            write(unit_number, '(A)') '  units: dimensionless'
        else
            write(unit_number, '(A)') '  units: angstrom'
        end if
        write(unit_number, '(A)') 'selection:'
        write(unit_number, '(A,I0)') '  start: ', frame_start
        write(unit_number, '(A,I0)') '  stop: ', frame_stop
        write(unit_number, '(A,I0)') '  stride: ', frame_stride
        write(unit_number, '(A,A,A)') "output: '", trim(final_output_path), "'"
        write(unit_number, '(A)') 'status: complete'
    end subroutine write_metadata


    subroutine fatal(message)
        character(len=*), intent(in) :: message
        write(error_unit, '(A)') 'molecular-axis: ' // trim(message)
        error stop 1
    end subroutine fatal

end program nematics_molecular_axis
