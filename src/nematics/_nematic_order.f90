program nematics_nematic_order
    use, intrinsic :: iso_fortran_env, only: int32, int64, real32, real64, error_unit
    implicit none

    integer, parameter :: path_len = 4096
    integer(int32), parameter :: endian_marker = 16909060_int32
    integer(int32), parameter :: output_version = 1_int32
    integer(int32), parameter :: output_dtype_bytes = 8_int32
    integer(int32), parameter :: output_header_bytes = 44_int32
    integer(int32), parameter :: output_record_values = 12_int32

    character(len=path_len) :: input_path = ''
    character(len=path_len) :: output_path = ''
    character(len=path_len) :: final_output_path = ''
    character(len=path_len) :: csv_path = ''
    character(len=path_len) :: metadata_path = ''
    character(len=path_len) :: performance_path = ''
    character(len=256) :: molecule_name = ''
    character(len=256) :: dataset_name = ''
    character(len=256) :: run_id = ''
    real(real64) :: temperature_k = 0.0_real64

    namelist /nematic_order_job/ input_path, output_path, final_output_path, &
        csv_path, metadata_path, performance_path, molecule_name, dataset_name, &
        run_id, temperature_k

    character(len=path_len) :: namelist_path
    character(len=8) :: input_magic, output_magic
    integer :: argc, cfg_unit, input_unit, output_unit, csv_unit, metadata_unit
    integer :: performance_unit, ios, frame, molecule, molecules
    integer(int32) :: input_version, input_marker, input_dtype_bytes
    integer(int32) :: input_components, input_header_bytes
    integer(int64) :: input_frames, input_molecules
    integer(int64) :: clock_start, clock_end, clock_rate
    real(real32), allocatable :: axes(:, :)
    real(real64) :: q(3, 3), eigenvalues(3), eigenvectors(3, 3)
    real(real64) :: axis(3), norm_squared, record(output_record_values)
    real(real64) :: elapsed_seconds, frames_per_second

    argc = command_argument_count()
    if (argc /= 1) call fatal('usage: nematic-order <job.nml>')
    call get_command_argument(1, namelist_path)

    open(newunit=cfg_unit, file=trim(namelist_path), status='old', &
        action='read', iostat=ios)
    if (ios /= 0) call fatal('cannot open the nematic-order namelist')
    read(cfg_unit, nml=nematic_order_job, iostat=ios)
    close(cfg_unit)
    if (ios /= 0) call fatal('cannot parse the nematic-order namelist')
    call validate_configuration()
    call system_clock(count=clock_start, count_rate=clock_rate)

    open(newunit=input_unit, file=trim(input_path), status='old', action='read', &
        access='stream', form='unformatted', iostat=ios)
    if (ios /= 0) call fatal('cannot open the molecular-axis binary input')
    read(input_unit, iostat=ios) input_magic
    if (ios /= 0) call fatal('cannot read the molecular-axis header')
    read(input_unit) input_version
    read(input_unit) input_marker
    read(input_unit) input_dtype_bytes
    read(input_unit) input_frames
    read(input_unit) input_molecules
    read(input_unit) input_components
    read(input_unit) input_header_bytes
    call validate_input_header()

    if (input_molecules > int(huge(molecules), int64)) then
        call fatal('molecule count is too large for this executable')
    end if
    molecules = int(input_molecules)
    allocate(axes(3, molecules))

    open(newunit=output_unit, file=trim(output_path), status='replace', &
        action='write', access='stream', form='unformatted', iostat=ios)
    if (ios /= 0) call fatal('cannot create the nematic-order binary output')
    output_magic = 'NEMORDR1'
    write(output_unit) output_magic
    write(output_unit) output_version
    write(output_unit) endian_marker
    write(output_unit) output_dtype_bytes
    write(output_unit) input_frames
    write(output_unit) 3_int32
    write(output_unit) 9_int32
    write(output_unit) output_record_values
    write(output_unit) output_header_bytes

    open(newunit=csv_unit, file=trim(csv_path), status='replace', &
        action='write', form='formatted', iostat=ios)
    if (ios /= 0) call fatal('cannot create the nematic-order CSV output')
    write(csv_unit, '(A)') &
        'frame,lambda_1,lambda_2,lambda_3,' // &
        'evec1_x,evec1_y,evec1_z,evec2_x,evec2_y,evec2_z,' // &
        'evec3_x,evec3_y,evec3_z'

    do frame = 1, int(input_frames)
        read(input_unit, iostat=ios) axes
        if (ios /= 0) call fatal('molecular-axis data ended before the advertised frame count')

        q = 0.0_real64
        do molecule = 1, molecules
            axis = real(axes(:, molecule), real64)
            norm_squared = sum(axis * axis)
            if (abs(norm_squared - 1.0_real64) > 1.0e-3_real64) then
                call fatal('input contains a non-unit molecular axis')
            end if
            q(1, 1) = q(1, 1) + axis(1) * axis(1)
            q(1, 2) = q(1, 2) + axis(1) * axis(2)
            q(1, 3) = q(1, 3) + axis(1) * axis(3)
            q(2, 2) = q(2, 2) + axis(2) * axis(2)
            q(2, 3) = q(2, 3) + axis(2) * axis(3)
            q(3, 3) = q(3, 3) + axis(3) * axis(3)
        end do
        q = 1.5_real64 * q / real(molecules, real64)
        q(1, 1) = q(1, 1) - 0.5_real64
        q(2, 2) = q(2, 2) - 0.5_real64
        q(3, 3) = q(3, 3) - 0.5_real64
        q(2, 1) = q(1, 2)
        q(3, 1) = q(1, 3)
        q(3, 2) = q(2, 3)

        call symmetric_eigensystem(q, eigenvalues, eigenvectors)
        record(1:3) = eigenvalues
        record(4:6) = eigenvectors(:, 1)
        record(7:9) = eigenvectors(:, 2)
        record(10:12) = eigenvectors(:, 3)
        write(output_unit) record
        write(csv_unit, '(I0,12(",",ES24.16E3))') frame, record
    end do

    close(csv_unit)
    close(output_unit)
    close(input_unit)

    open(newunit=metadata_unit, file=trim(metadata_path), status='replace', &
        action='write', form='formatted', iostat=ios)
    if (ios /= 0) call fatal('cannot create the nematic-order metadata file')
    call write_metadata(metadata_unit)
    close(metadata_unit)

    call system_clock(count=clock_end)
    if (clock_rate > 0_int64) then
        elapsed_seconds = real(clock_end - clock_start, real64) / real(clock_rate, real64)
    else
        elapsed_seconds = 0.0_real64
    end if
    if (elapsed_seconds > 0.0_real64) then
        frames_per_second = real(input_frames, real64) / elapsed_seconds
    else
        frames_per_second = 0.0_real64
    end if

    open(newunit=performance_unit, file=trim(performance_path), status='replace', &
        action='write', form='formatted', iostat=ios)
    if (ios /= 0) call fatal('cannot create the performance CSV file')
    write(performance_unit, '(A)') &
        'run_id,temperature_k,frames,molecules,extractor_wall_seconds,frames_per_second'
    write(performance_unit, '(A,",",F0.5,",",I0,",",I0,",",F0.6,",",F0.6)') &
        trim(run_id), temperature_k, input_frames, input_molecules, &
        elapsed_seconds, frames_per_second
    close(performance_unit)

    write(*, '(A,I0,A,I0,A)') 'Completed: calculated eigenpairs for ', &
        input_frames, ' frames of ', input_molecules, ' molecules.'

contains

    subroutine validate_configuration()
        if (len_trim(input_path) == 0) call fatal('input_path is empty')
        if (len_trim(output_path) == 0) call fatal('output_path is empty')
        if (len_trim(final_output_path) == 0) call fatal('final_output_path is empty')
        if (len_trim(csv_path) == 0) call fatal('csv_path is empty')
        if (len_trim(metadata_path) == 0) call fatal('metadata_path is empty')
        if (len_trim(performance_path) == 0) call fatal('performance_path is empty')
    end subroutine validate_configuration


    subroutine validate_input_header()
        if (input_magic /= 'NEMAXIS1') call fatal('input magic is not NEMAXIS1')
        if (input_version /= 1_int32) call fatal('unsupported molecular-axis version')
        if (input_marker /= endian_marker) call fatal('input byte order differs from this host')
        if (input_dtype_bytes /= 4_int32) call fatal('molecular-axis dtype is not float32')
        if (input_frames <= 0_int64) call fatal('input has no frames')
        if (input_molecules <= 0_int64) call fatal('input has no molecules')
        if (input_components /= 3_int32) call fatal('input does not have three components')
        if (input_header_bytes /= 44_int32) call fatal('unexpected molecular-axis header size')
    end subroutine validate_input_header


    subroutine symmetric_eigensystem(matrix, values, vectors)
        real(real64), intent(in) :: matrix(3, 3)
        real(real64), intent(out) :: values(3), vectors(3, 3)
        real(real64) :: work(3, 3), maximum, tau, tangent, cosine, sine
        real(real64) :: app, aqq, apq, arp, arq, vrp, vrq
        real(real64) :: temporary_value, temporary_vector(3), vector_norm
        integer :: sweep, p, q_index, row, first, second, largest_component

        work = matrix
        vectors = 0.0_real64
        vectors(1, 1) = 1.0_real64
        vectors(2, 2) = 1.0_real64
        vectors(3, 3) = 1.0_real64

        do sweep = 1, 50
            p = 1
            q_index = 2
            maximum = abs(work(1, 2))
            if (abs(work(1, 3)) > maximum) then
                p = 1
                q_index = 3
                maximum = abs(work(1, 3))
            end if
            if (abs(work(2, 3)) > maximum) then
                p = 2
                q_index = 3
                maximum = abs(work(2, 3))
            end if
            if (maximum < 1.0e-14_real64) exit

            apq = work(p, q_index)
            tau = (work(q_index, q_index) - work(p, p)) / (2.0_real64 * apq)
            if (tau >= 0.0_real64) then
                tangent = 1.0_real64 / (tau + sqrt(1.0_real64 + tau * tau))
            else
                tangent = -1.0_real64 / (-tau + sqrt(1.0_real64 + tau * tau))
            end if
            cosine = 1.0_real64 / sqrt(1.0_real64 + tangent * tangent)
            sine = tangent * cosine
            app = work(p, p)
            aqq = work(q_index, q_index)

            do row = 1, 3
                if (row == p .or. row == q_index) cycle
                arp = work(row, p)
                arq = work(row, q_index)
                work(row, p) = cosine * arp - sine * arq
                work(p, row) = work(row, p)
                work(row, q_index) = sine * arp + cosine * arq
                work(q_index, row) = work(row, q_index)
            end do
            work(p, p) = cosine * cosine * app - &
                2.0_real64 * sine * cosine * apq + sine * sine * aqq
            work(q_index, q_index) = sine * sine * app + &
                2.0_real64 * sine * cosine * apq + cosine * cosine * aqq
            work(p, q_index) = 0.0_real64
            work(q_index, p) = 0.0_real64

            do row = 1, 3
                vrp = vectors(row, p)
                vrq = vectors(row, q_index)
                vectors(row, p) = cosine * vrp - sine * vrq
                vectors(row, q_index) = sine * vrp + cosine * vrq
            end do
        end do

        values = [work(1, 1), work(2, 2), work(3, 3)]
        do first = 1, 2
            do second = first + 1, 3
                if (values(second) > values(first)) then
                    temporary_value = values(first)
                    values(first) = values(second)
                    values(second) = temporary_value
                    temporary_vector = vectors(:, first)
                    vectors(:, first) = vectors(:, second)
                    vectors(:, second) = temporary_vector
                end if
            end do
        end do

        do first = 1, 3
            vector_norm = sqrt(sum(vectors(:, first) * vectors(:, first)))
            if (vector_norm <= tiny(1.0_real64)) call fatal('eigensolver produced a zero vector')
            vectors(:, first) = vectors(:, first) / vector_norm
            largest_component = maxloc(abs(vectors(:, first)), dim=1)
            if (vectors(largest_component, first) < 0.0_real64) then
                vectors(:, first) = -vectors(:, first)
            end if
        end do
    end subroutine symmetric_eigensystem


    subroutine write_metadata(unit_number)
        integer, intent(in) :: unit_number

        write(unit_number, '(A)') 'format: nematics-nematic-order-v1'
        write(unit_number, '(A)') 'source:'
        write(unit_number, '(A,A,A)') "  molecular_axis: '", trim(input_path), "'"
        write(unit_number, '(A,A,A)') "  molecule: '", trim(molecule_name), "'"
        write(unit_number, '(A,A,A)') "  dataset: '", trim(dataset_name), "'"
        write(unit_number, '(A,A,A)') "  run_id: '", trim(run_id), "'"
        write(unit_number, '(A,F12.5)') '  temperature_k: ', temperature_k
        write(unit_number, '(A)') 'shape:'
        write(unit_number, '(A,I0)') '  frames: ', input_frames
        write(unit_number, '(A,I0)') '  molecules_per_frame: ', input_molecules
        write(unit_number, '(A)') 'calculation:'
        write(unit_number, '(A)') &
            "  Q: '(1/N) sum_i [(3/2) u_i u_i - (1/2) I]'"
        write(unit_number, '(A)') '  eigenpair_order: descending_eigenvalue'
        write(unit_number, '(A)') '  director: eigenvector_1'
        write(unit_number, '(A)') &
            '  eigenvector_sign: largest_absolute_component_positive'
        write(unit_number, '(A)') 'binary:'
        write(unit_number, '(A)') "  magic: 'NEMORDR1'"
        write(unit_number, '(A)') '  header_bytes: 44'
        write(unit_number, '(A)') '  dtype: float64'
        write(unit_number, '(A)') '  byte_order: native'
        write(unit_number, '(A)') &
            '  record: [lambda1, lambda2, lambda3, evec1_xyz, evec2_xyz, evec3_xyz]'
        write(unit_number, '(A,A,A)') "output: '", trim(final_output_path), "'"
        write(unit_number, '(A)') 'status: complete'
    end subroutine write_metadata


    subroutine fatal(message)
        character(len=*), intent(in) :: message
        write(error_unit, '(A)') 'nematic-order: ' // trim(message)
        error stop 1
    end subroutine fatal

end program nematics_nematic_order
