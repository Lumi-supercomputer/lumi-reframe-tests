!    This file is part of ELPA.
!
!    The ELPA library was originally created by the ELPA consortium,
!    consisting of the following organizations:
!
!    - Max Planck Computing and Data Facility (MPCDF), formerly known as
!      Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
!    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
!      Informatik,
!    - Technische Universität München, Lehrstuhl für Informatik mit
!      Schwerpunkt Wissenschaftliches Rechnen ,
!    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
!    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
!      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
!      and
!    - IBM Deutschland GmbH
!
!
!    More information can be found here:
!    http://elpa.mpcdf.mpg.de/
!
!    ELPA is free software: you can redistribute it and/or modify
!    it under the terms of the version 3 of the license of the
!    GNU Lesser General Public License as published by the Free
!    Software Foundation.
!
!    ELPA is distributed in the hope that it will be useful,
!    but WITHOUT ANY WARRANTY; without even the implied warranty of
!    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
!    GNU Lesser General Public License for more details.
!
!    You should have received a copy of the GNU Lesser General Public License
!    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
!
!    ELPA reflects a substantial effort on the part of the original
!    ELPA consortium, and we ask you to respect the spirit of the
!    license that we chose: i.e., please contribute any changes you
!    may have back to the original ELPA library distribution, and keep
!    any derivatives of ELPA under the same license that we chose for
!    the original distribution, the GNU Lesser General Public License.
!
!

program test_elpa_cholesky
   use elpa
   use mpi

   implicit none

   ! matrix dimensions
   integer(kind=c_int32_t)                       :: na, nev, nblk

   ! mpi
   integer(kind=c_int32_t)                       :: myid, nprocs
   integer(kind=c_int32_t)                       :: na_cols, na_rows  ! local matrix size
   integer(kind=c_int32_t)                       :: np_cols, np_rows  ! number of MPI processes per column/row
   integer(kind=c_int32_t)                       :: mpierr

   ! The Matrix
   real(kind=C_DOUBLE), allocatable, target    :: a(:,:)
   real(kind=C_DOUBLE), allocatable, target    :: z(:,:)

   integer(kind=c_int32_t)             :: status
   integer(kind=c_int)                 :: error_elpa

   class(elpa_t), pointer              :: e

   integer(kind=c_int)                 :: kernel
   character(len=1)                    :: layout

   integer :: i


   ! default parameters
   na = 500
   nev = 150
   nblk = 16
   myid = 0
   nprocs = 1

   call mpi_init(mpierr)
   call mpi_comm_rank(mpi_comm_world, myid,  mpierr)
   call mpi_comm_size(mpi_comm_world, nprocs, mpierr)

   status = 0
   if (elpa_init(20241105) /= ELPA_OK) then
     print *, "ELPA API version not supported"
     stop 1
   endif

   if (myid == 0) then
     print '((a,i0))', 'Program validate_real_double_cholesky_1stage_gpu_random'
     print *, ""
   endif

   layout = 'C'

   np_rows = 1
   np_cols = 1 

   if (myid == 0) then
     print '((a,i0))', 'Matrix size: ', na
     print '((a,i0))', 'Num eigenvectors: ', nev
     print '((a,i0))', 'Blocksize: ', nblk
     print '((a,i0))', 'Num MPI proc: ', nprocs
     print '(3(a,i0))','Number of processor rows=',np_rows,', cols=',np_cols,', total=',nprocs
     print '(a)',      'Process layout: ' // layout
     print *,''
   endif

   ! Allocate the matrices needed for elpa

   na_rows = na
   na_cols = na

   allocate(a (na_rows,na_cols))
   allocate(z (na_rows,na_cols))

   a(:,:) = 0.0
   z(:,:) = 0.0

    call random_number(a)
    z = transpose(a)
    a = 0.5*(a+z)
    do i = 1, na
      a(i,i) = a(i,i) + na
    enddo

   e => elpa_allocate(error_elpa)

! Set parameters

   call e%set("na", int(na,kind=c_int), error_elpa)
   call e%set("nev", int(nev,kind=c_int), error_elpa)
   call e%set("local_nrows", na, error_elpa)
   call e%set("local_ncols", na, error_elpa)
   call e%set("nblk", int(nblk,kind=c_int), error_elpa)
   if (layout .eq. 'C') then
     call e%set("matrix_order",COLUMN_MAJOR_ORDER,error_elpa)
   else
     call e%set("matrix_order",ROW_MAJOR_ORDER,error_elpa)
   endif
   call e%set("mpi_comm_parent", MPI_COMM_WORLD, error_elpa)
   call e%set("debug", 1, error_elpa)
   call e%set("process_row", na, error_elpa)
   call e%set("process_col", na, error_elpa)
   call e%set("verbose", 1, error_elpa)
   call e%set("timings", 1, error_elpa)
   call e%set("solver", ELPA_SOLVER_2STAGE, error_elpa)
   call e%set("real_kernel", ELPA_2STAGE_REAL_AMD_GPU, error_elpa)
   call e%set("amd-gpu", 1, error_elpa)

   error_elpa = e%setup()

   call e%print_settings(error_elpa)

   !_____________________________________________________________________________________________________________________
   ! The actual solve step

     call e%timer_start("e%cholesky()")

     call e%cholesky(a, error_elpa)

     call e%timer_stop("e%cholesky()")

     if (myid .eq. 0) then
       call e%print_times("e%cholesky()")
     endif

   !_____________________________________________________________________________________________________________________
   ! Deallocate

   call elpa_deallocate(e, error_elpa)

   deallocate(a)
   deallocate(z)

   call elpa_uninit(error_elpa)

   call mpi_finalize(mpierr)

   call exit(status)

end program
