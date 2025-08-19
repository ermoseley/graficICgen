module grafic_io_test
  implicit none
  integer, parameter :: i4 = selected_int_kind(9)
  integer, parameter :: r4 = selected_real_kind(6)
  integer, parameter :: i8 = selected_int_kind(18)
contains

  subroutine read_ic_particle_ids(filename, ids, n1, n2, n3)
    character(len=*), intent(in) :: filename
    integer(i8), allocatable, intent(out) :: ids(:,:,:)
    integer(i4), intent(out) :: n1, n2, n3

    integer(i4) :: n1h, n2h, n3h
    real(r4) :: dx, x1, x2, x3, f1, f2, f3, f4
    integer :: unit, k, ios, i, j
    integer(i8), allocatable :: plane(:,:)

    unit = 33
    open(unit, file=filename, form='unformatted', access='sequential', &
         action='read', status='old', iostat=ios)
    if (ios /= 0) then
      write(*,*) 'Open failed for ', trim(filename), ' iostat=', ios
      stop 1
    end if

    ! Header: 3x int32 + 8x float32
    read(unit) n1h, n2h, n3h, dx, x1, x2, x3, f1, f2, f3, f4
    n1 = n1h; n2 = n2h; n3 = n3h

    allocate(ids(n1, n2, n3))
    allocate(plane(n1, n2))

    do k = 1, n3
      read(unit) ((plane(i,j), i=1,n1), j=1,n2)
      ids(:,:,k) = plane
    end do

    close(unit)
    deallocate(plane)
  end subroutine read_ic_particle_ids

#ifdef USE_MPI
  subroutine read_ic_particle_ids_mpi(filename, ids, n1, n2, n3)
    use mpi
    character(len=*), intent(in) :: filename
    integer(i8), allocatable, intent(out) :: ids(:,:,:)
    integer(i4), intent(out) :: n1, n2, n3

    integer :: unit, k, ios, i, j, ierr, rank
    integer :: n1h, n2h, n3h
    real(r4) :: dx, x1, x2, x3, f1, f2, f3, f4
    integer(i8), allocatable :: plane(:,:)
    integer :: id_type

    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)

    if (rank == 0) then
      unit = 33
      open(unit, file=filename, form='unformatted', access='sequential', &
           action='read', status='old', iostat=ios)
      if (ios /= 0) then
        write(*,*) 'Open failed for ', trim(filename), ' iostat=', ios
        call MPI_Abort(MPI_COMM_WORLD, 1, ierr)
      end if
      read(unit) n1h, n2h, n3h, dx, x1, x2, x3, f1, f2, f3, f4
    end if

    call MPI_Bcast(n1h, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
    call MPI_Bcast(n2h, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)
    call MPI_Bcast(n3h, 1, MPI_INTEGER, 0, MPI_COMM_WORLD, ierr)

    n1 = n1h; n2 = n2h; n3 = n3h

    allocate(ids(n1, n2, n3))
    allocate(plane(n1, n2))

    call MPI_Type_create_f90_integer(18, id_type, ierr)

    do k = 1, n3
      if (rank == 0) then
        read(unit) ((plane(i,j), i=1,n1), j=1,n2)
      else
        plane = 0_i8
      end if
      call MPI_Bcast(plane, n1*n2, id_type, 0, MPI_COMM_WORLD, ierr)
      ids(:,:,k) = plane
    end do

    call MPI_Type_free(id_type, ierr)

    if (rank == 0) then
      close(unit)
    end if

    deallocate(plane)
  end subroutine read_ic_particle_ids_mpi
#endif

end module grafic_io_test

program test_read_ids
#ifdef USE_MPI
  use mpi
#endif
  use grafic_io_test
  implicit none

  character(len=256) :: fname
  integer(i4) :: n1, n2, n3
  integer(i8), allocatable :: ids(:,:,:)
  integer(i8) :: smin, smax, sum_ids, expected_sum, ntotal
  integer :: i, j, k
#ifdef USE_MPI
  integer :: ierr, rank, nprocs
#endif
  logical :: has_zero, all_present
  logical, allocatable :: present(:)
  integer(i8) :: id_val
  integer(i8) :: num_duplicates, num_out_of_range

#ifdef USE_MPI
  call MPI_Init(ierr)
  call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
  call MPI_Comm_size(MPI_COMM_WORLD, nprocs, ierr)
#endif

  call get_command_argument(1, fname)
  if (len_trim(fname) == 0) then
    fname = 'ic_particle_ids'
  else
    fname = trim(fname)
  end if

#ifdef USE_MPI
  if (nprocs > 1) then
    call read_ic_particle_ids_mpi(fname, ids, n1, n2, n3)
  else
    call read_ic_particle_ids(fname, ids, n1, n2, n3)
  end if
#else
  call read_ic_particle_ids(fname, ids, n1, n2, n3)
#endif

  ntotal = int(n1, kind=i8) * int(n2, kind=i8) * int(n3, kind=i8)
  smin = ids(1,1,1)
  smax = ids(1,1,1)
  sum_ids = 0_i8
  has_zero = .false.

  do k = 1, n3
    do j = 1, n2
      do i = 1, n1
        id_val = ids(i,j,k)
        sum_ids = sum_ids + id_val
        if (id_val < smin) smin = id_val
        if (id_val > smax) smax = id_val
        if (id_val == 0_i8) has_zero = .true.
      end do
    end do
  end do

  expected_sum = ntotal * (ntotal + 1_i8) / 2_i8

#ifdef USE_MPI
  if (nprocs == 1 .or. rank == 0) then
#endif
    write(*,*) 'n1,n2,n3 = ', n1, n2, n3
    write(*,*) 'min,max   = ', smin, smax
    write(*,*) 'sum       = ', sum_ids
    write(*,*) 'expected  = ', expected_sum

    if (has_zero) then
      write(*,*) 'ERROR: At least one ID is zero.'
      stop 2
    end if

    if (smin /= 1_i8 .or. smax /= ntotal) then
      write(*,*) 'ERROR: ID range mismatch. Expected [1..', ntotal, ']'
      stop 2
    end if

    ! Full coverage check (duplicates and missing)
    allocate(present(ntotal))
    present = .false.
    num_duplicates = 0_i8
    num_out_of_range = 0_i8

    do k = 1, n3
      do j = 1, n2
        do i = 1, n1
          id_val = ids(i,j,k)
          if (id_val < 1_i8 .or. id_val > ntotal) then
            num_out_of_range = num_out_of_range + 1_i8
          else
            if (present(id_val)) then
              num_duplicates = num_duplicates + 1_i8
            else
              present(id_val) = .true.
            end if
          end if
        end do
      end do
    end do

    all_present = .true.
    do id_val = 1_i8, ntotal
      if (.not. present(id_val)) then
        all_present = .false.
        exit
      end if
    end do

    if (num_out_of_range /= 0_i8) then
      write(*,*) 'ERROR: Found IDs out of range: ', num_out_of_range
      stop 2
    end if

    if (num_duplicates /= 0_i8) then
      write(*,*) 'ERROR: Found duplicate IDs: ', num_duplicates
      stop 2
    end if

    if (.not. all_present) then
      write(*,*) 'ERROR: Missing some IDs in 1..N.'
      stop 2
    end if

    if (sum_ids == expected_sum) then
      write(*,*) 'OK: IDs look consistent (no zero, full coverage, correct sum).'
    else
      write(*,*) 'ERROR: Sum mismatch despite coverage checks.'
      stop 2
    end if

    deallocate(present)
#ifdef USE_MPI
  end if
#endif

#ifdef USE_MPI
  call MPI_Finalize(ierr)
#endif

end program test_read_ids
