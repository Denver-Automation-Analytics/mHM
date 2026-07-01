!> \file mo_read_spatial_data.f90
!> \brief \copybrief mo_read_spatial_data
!> \details \copydetails mo_read_spatial_data

!> \brief Reads spatial input data.
!> \details This module is to read spatial input data, e.g. dem, aspect, flow direction.
!! The module provides a subroutine for ASCII files.
!! (Subroutine for NetCDF files will come with release 5.1).
!! The data are read from the specified directory.
!> \authors Juliane Mai
!> \date Dec 2012
!> \copyright Copyright 2005-\today, the mHM Developers, Luis Samaniego, Sabine Attinger: All rights reserved.
!! mHM is released under the LGPLv3+ license \license_note
!> \ingroup f_common
MODULE mo_read_spatial_data

  ! This module provides routines to read spatial data.

  ! Written  Juliane Mai, Jan 2013
  ! Modified

  USE mo_kind, ONLY : i4, dp
  USE mo_os, ONLY : check_path_isfile
  USE mo_message, only: error_message
  use mo_string_utils, only : num2str

  IMPLICIT NONE
  PUBLIC :: check_uniform_axis, get_header_info_from_nc, read_header_nc_or_ascii
  PUBLIC :: read_header_ascii             ! Reads header of ASCII files
  PUBLIC :: read_spatial_data_nc_or_ascii ! Reads netcdf and ASCII files
  PUBLIC :: read_spatial_data_nc          ! Reads netcdf files
  PUBLIC :: read_spatial_data_ascii       ! Reads ASCII files

  
  ! ------------------------------------------------------------------

  !    NAME
  !        read_spatial_data_nc_or_ascii

  !    PURPOSE
  !>       \brief Reads spatial data files of nc or ASCII format.

  !>       \details Reads spatial input data, e.g. dem, aspect, flow direction.

  !    HISTORY
  !>       \authors Simon Lüdke

  !>       \date June 2025

  INTERFACE  read_spatial_data_nc_or_ascii
    MODULE PROCEDURE read_spatial_data_nc_or_ascii_i4, read_spatial_data_nc_or_ascii_dp
  END INTERFACE read_spatial_data_nc_or_ascii

  ! ------------------------------------------------------------------

  !    NAME
  !        read_spatial_data_nc

  !    PURPOSE
  !>       \brief Reads spatial data files of nc format.

  !>       \details Reads spatial input data, e.g. dem, aspect, flow direction.

  !    HISTORY
  !>       \authors Simon Lüdke

  !>       \date June 2025

  INTERFACE  read_spatial_data_nc
    MODULE PROCEDURE read_spatial_data_nc_i4, read_spatial_data_nc_dp
  END INTERFACE read_spatial_data_nc

  ! ------------------------------------------------------------------

  !    NAME
  !        read_spatial_data_ascii

  !    PURPOSE
  !>       \brief Reads spatial data files of ASCII format.

  !>       \details Reads spatial input data, e.g. dem, aspect, flow direction.

  !    HISTORY
  !>       \authors Juliane Mai

  !>       \date Jan 2013

  ! Modifications:
  ! Matthias Zink  Feb 2013 - , added interface and routine for datatype i4
  ! David Schaefer Mar 2015 - , removed double allocation of temporary data
  ! Robert Schweppe Jun 2018 - refactoring and reformatting

  INTERFACE  read_spatial_data_ascii
    MODULE PROCEDURE read_spatial_data_ascii_i4, read_spatial_data_ascii_dp
  END INTERFACE read_spatial_data_ascii

  public :: check_header

  

  ! ------------------------------------------------------------------

  PRIVATE

  ! ------------------------------------------------------------------

CONTAINS

  ! ------------------------------------------------------------------

  !    NAME
  !        read_spatial_data_ascii_dp

  !    PURPOSE
  !>       \brief TODO: add description

  !>       \details TODO: add description

  !    INTENT(IN)
  !>       \param[in] "character(len = *) :: filename" filename with location
  !>       \param[in] "integer(i4) :: fileunit"        unit for opening the file
  !>       \param[in] "integer(i4) :: header_nCols"    number of columns of data fields:
  !>       \param[in] "integer(i4) :: header_nRows"    number of rows of data fields:
  !>       \param[in] "real(dp) :: header_xllcorner"   header read in lower left corner
  !>       \param[in] "real(dp) :: header_yllcorner"   header read in lower left corner
  !>       \param[in] "real(dp) :: header_cellsize"    header read in cellsize

  !    INTENT(OUT)
  !>       \param[out] "real(dp), dimension(:, :) :: data" data
  !>       \param[out] "logical, dimension(:, :) :: mask"  mask

  !    HISTORY
  !>       \authors Robert Schweppe

  !>       \date Jun 2018

  ! Modifications:

  subroutine read_spatial_data_ascii_dp(filename, fileunit, header_ncols, header_nrows, header_xllcorner, &
                                       header_yllcorner, header_cellsize, data, mask)
    implicit none

    ! filename with location
    character(len = *), intent(in) :: filename

    ! unit for opening the file
    integer(i4), intent(in) :: fileunit

    ! number of rows of data fields:
    integer(i4), intent(in) :: header_nRows

    ! number of columns of data fields:
    integer(i4), intent(in) :: header_nCols

    ! header read in lower left corner
    real(dp), intent(in) :: header_xllcorner

    ! header read in lower left corner
    real(dp), intent(in) :: header_yllcorner

    ! header read in cellsize
    real(dp), intent(in) :: header_cellsize

    ! data
    real(dp), dimension(:, :), allocatable, intent(out) :: data

    ! mask
    logical, dimension(:, :), allocatable, intent(out) :: mask

    ! number of rows of data fields:
    integer(i4) :: file_nRows

    ! number of columns of data fields:
    integer(i4) :: file_nCols

    ! file read in lower left corner
    real(dp) :: file_xllcorner

    ! file read in lower left corner
    real(dp) :: file_yllcorner

    ! file read in cellsize
    real(dp) :: file_cellsize

    ! file read in nodata value
    real(dp) :: file_nodata

    integer(i4) :: i, j

    ! data
    real(dp), dimension(:, :), allocatable :: tmp_data

    ! mask
    logical, dimension(:, :), allocatable :: tmp_mask

    call read_header_ascii(filename, fileunit, &
            file_ncols, file_nrows, &
            file_xllcorner, file_yllcorner, file_cellsize, file_nodata)
    call check_header(file_ncols, file_nrows, file_xllcorner, file_yllcorner, file_cellsize, & 
                      header_ncols, header_nrows, header_xllcorner, header_yllcorner, header_cellsize)
    ! allocation and initialization of matrices
    allocate(tmp_data(file_nrows, file_ncols))
    tmp_data = file_nodata
    allocate(tmp_mask(file_nrows, file_ncols))
    tmp_mask = .true.


    !checking whether the file exists
    call check_path_isfile(path = filename, raise=.true.)
    ! read in
    ! recl is only a rough estimate on bytes per line in the ascii
    ! default for nag: recl=1024(byte) which is not enough for 100s of columns
    open (unit = fileunit, file = filename, action = 'read', status = 'old', recl = 48 * file_ncols)
    ! (a) skip header
    do i = 1, 6
      read(fileunit, *)
    end do
    ! (b) read data
    do i = 1, file_nrows
      read(fileunit, *) (tmp_data(i, j), j = 1, file_ncols)
    end do
    close(fileunit)

    ! set mask .false. if nodata value appeared
    where (abs(tmp_data - file_nodata) .lt. tiny(1.0_dp))
      tmp_mask = .false.
    end where

    ! transpose of data due to longitude-latitude ordering
    allocate(data(file_ncols, file_nrows))
    data = transpose(tmp_data)
    deallocate(tmp_data)

    allocate(mask(file_ncols, file_nrows))
    mask = transpose(tmp_mask)
    deallocate(tmp_mask)

  end subroutine read_spatial_data_ascii_dp

  !    NAME
  !        read_spatial_data_ascii_i4

  !    PURPOSE
  !>       \brief TODO: add description

  !>       \details TODO: add description

  !    INTENT(IN)
  !>       \param[in] "character(len = *) :: filename" filename with location
  !>       \param[in] "integer(i4) :: fileunit"        unit for opening the file
  !>       \param[in] "integer(i4) :: header_nCols"    number of columns of data fields:
  !>       \param[in] "integer(i4) :: header_nRows"    number of rows of data fields:
  !>       \param[in] "real(dp) :: header_xllcorner"   header read in lower left corner
  !>       \param[in] "real(dp) :: header_yllcorner"   header read in lower left corner
  !>       \param[in] "real(dp) :: header_cellsize"    header read in cellsize

  !    INTENT(OUT)
  !>       \param[out] "integer(i4), dimension(:, :) :: data" data
  !>       \param[out] "logical, dimension(:, :) :: mask"     mask

  !    HISTORY
  !>       \authors Robert Schweppe

  !>       \date Jun 2018

  ! Modifications:

  subroutine read_spatial_data_ascii_i4(filename, fileunit, header_ncols, header_nrows, header_xllcorner, &
                                       header_yllcorner, header_cellsize, data, mask)
    implicit none

    ! filename with location
    character(len = *), intent(in) :: filename

    ! unit for opening the file
    integer(i4), intent(in) :: fileunit

    ! number of rows of data fields:
    integer(i4), intent(in) :: header_nRows

    ! number of columns of data fields:
    integer(i4), intent(in) :: header_nCols

    ! header read in lower left corner
    real(dp), intent(in) :: header_xllcorner

    ! header read in lower left corner
    real(dp), intent(in) :: header_yllcorner

    ! header read in cellsize
    real(dp), intent(in) :: header_cellsize

    ! data
    integer(i4), dimension(:, :), allocatable, intent(out) :: data

    ! mask
    logical, dimension(:, :), allocatable, intent(out) :: mask

    ! number of rows of data fields:
    integer(i4) :: file_nRows

    ! number of columns of data fields:
    integer(i4) :: file_nCols

    ! file read in lower left corner
    real(dp) :: file_xllcorner

    ! file read in lower left corner
    real(dp) :: file_yllcorner

    ! file read in cellsize
    real(dp) :: file_cellsize

    ! file read in nodata value
    real(dp) :: file_nodata

    integer(i4) :: i, j

    ! data
    integer(i4), dimension(:, :), allocatable :: tmp_data

    ! mask
    logical, dimension(:, :), allocatable :: tmp_mask

    call read_header_ascii(filename, fileunit, &
            file_ncols, file_nrows, &
            file_xllcorner, file_yllcorner, file_cellsize, file_nodata)
    call check_header(file_ncols, file_nrows, file_xllcorner, file_yllcorner, file_cellsize, &
                      header_ncols, header_nrows, header_xllcorner, header_yllcorner, header_cellsize)
    ! allocation and initialization of matrices
    allocate(tmp_data(file_nrows, file_ncols))
    tmp_data = int(file_nodata, i4)
    allocate(tmp_mask(file_nrows, file_ncols))
    tmp_mask = .true.

    !checking whether the file exists
    call check_path_isfile(path = filename, raise=.true.)
    ! read in
    ! recl is only a rough estimate on bytes per line in the ascii
    ! default for nag: recl=1024(byte) which is not enough for 100s of columns
    open (unit = fileunit, file = filename, action = 'read', status = 'old', recl = 48 * file_ncols)
    ! (a) skip header
    do i = 1, 6
      read(fileunit, *)
    end do
    ! (b) read data
    do i = 1, file_nrows
      read(fileunit, *) (tmp_data(i, j), j = 1, file_ncols)
    end do
    close(fileunit)

    ! set mask .false. if nodata value appeared
    where (tmp_data .EQ. int(file_nodata, i4))
      tmp_mask = .false.
    end where

    ! transpose of data due to longitude-latitude ordering
    allocate(data(file_ncols, file_nrows))
    data = transpose(tmp_data)
    deallocate(tmp_data)

    allocate(mask(file_ncols, file_nrows))
    mask = transpose(tmp_mask)
    deallocate(tmp_mask)

  end subroutine read_spatial_data_ascii_i4

  ! ------------------------------------------------------------------

  !    NAME
  !        read_header_ascii

  !    PURPOSE
  !>       \brief Reads header lines of ASCII files.

  !>       \details Reads header lines of ASCII files, e.g. dem, aspect, flow direction.

  !    INTENT(IN)
  !>       \param[in] "character(len = *) :: filename" Name of file and its location
  !>       \param[in] "integer(i4) :: fileunit"        File unit for open file

  !    INTENT(OUT)
  !>       \param[out] "integer(i4) :: header_nCols"  Reference number of columns
  !>       \param[out] "integer(i4) :: header_nRows"  Reference number of rows
  !>       \param[out] "real(dp) :: header_xllcorner" Reference lower left corner (x)
  !>       \param[out] "real(dp) :: header_yllcorner" Reference lower left corner (y)
  !>       \param[out] "real(dp) :: header_cellsize"  Reference cell size [m]
  !>       \param[out] "real(dp) :: header_nodata"    Reference nodata value

  !    HISTORY
  !>       \authors Juliane Mai

  !>       \date Jan 2013

  ! Modifications:
  ! Robert Schweppe Jun 2018 - refactoring and reformatting

  subroutine read_header_ascii(filename, fileunit, header_ncols, header_nrows, header_xllcorner, header_yllcorner, &
                              header_cellsize, header_nodata)
    use mo_common_constants, only : nodata_dp
    implicit none

    ! Name of file and its location
    character(len = *), intent(in) :: filename

    ! File unit for open file
    integer(i4), intent(in) :: fileunit

    ! Reference number of rows
    integer(i4), intent(out) :: header_nRows

    ! Reference number of columns
    integer(i4), intent(out) :: header_nCols

    ! Reference lower left corner (x)
    real(dp), intent(out) :: header_xllcorner

    ! Reference lower left corner (y)
    real(dp), intent(out) :: header_yllcorner

    ! Reference cell size [m]
    real(dp), intent(out) :: header_cellsize

    ! Reference nodata value
    real(dp), intent(out) :: header_nodata

    character(5) :: dummy
    integer(i4) :: io

    !checking whether the file exists
    call check_path_isfile(path = filename, raise=.true.)
    ! reading header from a file
    open (unit = fileunit, file = filename, status = 'old')
    read (fileunit, *) dummy, header_nCols
    read (fileunit, *) dummy, header_nRows
    read (fileunit, *) dummy, header_xllcorner
    read (fileunit, *) dummy, header_yllcorner
    read (fileunit, *) dummy, header_cellsize
    read (fileunit, *, iostat=io) dummy, header_nodata
    ! EOF reached (nodata not present, use default value)
    if (io < 0) header_nodata = nodata_dp
    close(fileunit)
    dummy = dummy // ''   ! only to avoid warning

  end subroutine read_header_ascii

  subroutine check_header(ncols, nrows, xllcorner, yllcorner, &
          cellsize, ref_ncols, ref_nrows, ref_xllcorner, ref_yllcorner, ref_cellsize, &
          tolerance, context)

    use mo_kind, only : dp, i4
    use mo_message, only : error_message
    use mo_string_utils, only : num2str
    use mo_common_constants, only: defaultTolerance_dp

    implicit none

    integer(i4), intent(in) :: ncols
    integer(i4), intent(in) :: nrows
    real(dp), intent(in) :: xllcorner
    real(dp), intent(in) :: yllcorner
    real(dp), intent(in) :: cellsize
    integer(i4), intent(in) :: ref_nrows
    integer(i4), intent(in) :: ref_ncols
    real(dp), intent(in) :: ref_xllcorner
    real(dp), intent(in) :: ref_yllcorner
    real(dp), intent(in) :: ref_cellsize
    real(dp), intent(in), optional :: tolerance
    character(*), intent(in), optional :: context

    real(dp) :: tol_local
    character(len=:), allocatable :: err_message

    if (present(tolerance)) then
      tol_local = tolerance
    else
      tol_local = defaultTolerance_dp
    end if

    err_message = ''
    if ((ref_ncols .ne. ncols)) then
      err_message = err_message // 'read_spatial_data_ascii: ncols mismatch file='// &
                    trim(num2str(ncols))//' expected='//trim(num2str(ref_ncols))// &
                    new_line('a')
    end if
    if ((nrows .ne. ref_nrows)) then
      err_message = err_message // 'read_spatial_data_ascii: nrows mismatch file='// &
                    trim(num2str(nrows))//' expected='//trim(num2str(ref_nrows))// &
                    new_line('a')
    end if
    if ((abs(xllcorner - ref_xllcorner) .gt. tol_local)) then
      err_message = err_message // 'read_spatial_data_ascii: xllcorner mismatch file='// &
                    trim(num2str(xllcorner, '(G32.16)'))//' expected='// &
                    trim(num2str(ref_xllcorner, '(G32.16)'))// &
                    new_line('a')
    end if
    if ((abs(yllcorner - ref_yllcorner) .gt. tol_local)) then
      err_message = err_message // 'read_spatial_data_ascii: yllcorner mismatch file='// &
                    trim(num2str(yllcorner, '(G32.16)'))//' expected='// &
                    trim(num2str(ref_yllcorner, '(G32.16)'))// &
                    new_line('a')
    end if
    if ((abs(cellsize - ref_cellsize) .gt. tol_local)) then
      err_message = err_message // 'read_spatial_data_ascii: cellsize mismatch file='// &
                    trim(num2str(cellsize, '(G32.16)'))//' expected='// &
                    trim(num2str(ref_cellsize, '(G32.16)'))// &
                    new_line('a')
    end if

    if (len_trim(err_message) > 0) then
      if (present(context)) then
        err_message = trim(context) // new_line('a') // err_message
      end if 
      call error_message(trim(err_message))
    end if

  end subroutine check_header

  !> \brief Read header information from either NetCDF or ASCII depending on file name/presence.
  !> \authors Simon Lüdke
  !> \date Feb 2025
  subroutine read_header_nc_or_ascii(filepath, fileunit, header_ncols, header_nrows, header_xllcorner, header_yllcorner, &
                                     header_cellsize, header_nodata, varName)
    use mo_os, only : path_root, path_isfile
    implicit none
    character(len=*), intent(in) :: filepath
    integer(i4), intent(in) :: fileunit
    integer(i4), intent(out) :: header_ncols, header_nrows
    real(dp), intent(out) :: header_xllcorner, header_yllcorner, header_cellsize, header_nodata
    character(len=*), intent(in), optional :: varName

    character(len=:), allocatable :: ncname
    character(len=:), allocatable :: trimmed
    integer(i4) :: len_path
    logical :: is_nc

    trimmed = trim(filepath)
    len_path = len_trim(trimmed)
    is_nc = .false.
    if (len_path >= 3) then
      if ((trimmed(len_path-2:len_path) == '.nc') .or. (trimmed(len_path-2:len_path) == '.NC')) is_nc = .true.
    end if

    ncname = trimmed
    if (.not. is_nc) ncname = path_root(trimmed) // '.nc'

    if (is_nc .or. path_isfile(ncname)) then
      call get_header_info_from_nc(ncname, fileunit, header_ncols, header_nrows, header_xllcorner, header_yllcorner, &
                                   header_cellsize, header_nodata, var=varName)
    else
      call read_header_ascii(trimmed, fileunit, header_ncols, header_nrows, header_xllcorner, header_yllcorner, &
                             header_cellsize, header_nodata)
    end if
  end subroutine read_header_nc_or_ascii

    !> \brief check if given axis is a uniform axis.
  !> \authors Sebastian Müller
  !> \date Mar 2024
  subroutine check_uniform_axis(var, cellsize, origin, increasing, tol)
    use mo_netcdf, only : NcVariable
    use mo_utils, only: is_close
    implicit none
    type(NcVariable), intent(in) :: var !< NetCDF variable for corresponding axis
    real(dp), optional, intent(out) :: cellsize !< cellsize of the uniform axis
    real(dp), optional, intent(out) :: origin !< origin of the axis vertices
    logical, optional, intent(out) :: increasing !< whether the axis has increasing values
    real(dp), intent(in), optional :: tol !< tolerance for cell size comparisson (default: 1.e-7)
    real(dp), dimension(:), allocatable :: axis
    real(dp), dimension(:,:), allocatable :: bounds
    real(dp) :: diff, tol_
    integer(i4) :: i_ub, i_lb
    logical :: has_bnds
    type(NcVariable) :: bnds
    character(len=256) :: name

    call var%getData(axis)
    if (var%hasAttribute("bounds")) then
      has_bnds = .true.
      call var%getAttribute("bounds", name)
      bnds = var%parent%getVariable(trim(name))
      call bnds%getData(bounds)
    else
      has_bnds = .false.
    end if
    ! store var name for error messages
    name = var%getName()

    tol_ = 1.e-7_dp
    if ( present(tol) ) tol_ = tol

    if (size(axis) == 0_i4) &
      call error_message("check_uniform_axis: axis is empty: ", name)

    if (size(axis) > 1_i4) then
      diff = (axis(size(axis)) - axis(1)) / real(size(axis) - 1_i4, dp)
      if (.not.all(is_close(axis(2:size(axis))-axis(1:size(axis)-1), diff, rtol=0.0_dp, atol=tol_))) &
        call error_message("check_uniform_axis: given axis is not uniform: ", name)
    else
      if (.not. has_bnds) &
        call error_message("check_uniform_axis: can't check axis of size 1 when no bounds are given: ", name)
      diff = bounds(2,1) - bounds(1,1)
    end if

    if (has_bnds) then
      ! be forgiving if the bounds don't have the same monotonicity as the axis (cf-convetion is hard)
      i_lb = 1
      i_ub = 2
      if (size(bounds, dim=2)>1) then
        if (.not. is_close(bounds(2,1), bounds(1,2), rtol=0.0_dp, atol=tol_) &
            .and. is_close(bounds(1,1), bounds(2,2), rtol=0.0_dp, atol=tol_)) then
          print *, "check_uniform_axis: bounds actually have wrong monotonicity: ", name
          i_lb = 2
          i_ub = 1
        end if
      end if
      if (.not.all(is_close(bounds(i_ub,:)-bounds(i_lb,:), diff, rtol=0.0_dp, atol=tol_))) &
        call error_message("check_uniform_axis: given bounds are not uniform: ", name)
      if (.not.all(is_close(axis(:)-bounds(i_lb,:), 0.5_dp*diff, rtol=0.0_dp, atol=tol_))) &
        call error_message("check_uniform_axis: given bounds are not centered around axis points: ", name)
    end if

    if ( present(cellsize) ) cellsize = abs(diff)
    if ( present(origin) ) origin = minval(axis) - 0.5_dp * abs(diff)
    if ( present(increasing) ) increasing = diff > 0.0_dp

  end subroutine check_uniform_axis

  !> \brief check if given variable is a x-axis.
  !> \return `logical :: is_x_axis`
  !> \authors Sebastian Müller
  !> \date Mar 2024
  logical function is_x_axis(var)
    use mo_netcdf, only : NcVariable
    implicit none
    type(NcVariable), intent(in) :: var !< NetCDF variable to check
    character(len=256) :: tmp_str

    is_x_axis = .false.
    if (var%hasAttribute("standard_name")) then
      call var%getAttribute("standard_name", tmp_str)
      if (trim(tmp_str) == "projection_x_coordinate") is_x_axis = .true.
    else if (var%hasAttribute("axis")) then
      call var%getAttribute("axis", tmp_str)
      if (trim(tmp_str) == "X") is_x_axis = .true.
    else if (var%hasAttribute("_CoordinateAxisType")) then
      call var%getAttribute("_CoordinateAxisType", tmp_str)
      if (trim(tmp_str) == "GeoX") is_x_axis = .true.
    end if
  end function is_x_axis

  !> \brief check if given variable is a y-axis.
  !> \return `logical :: is_y_axis`
  !> \authors Sebastian Müller
  !> \date Mar 2024
  logical function is_y_axis(var)
    use mo_netcdf, only : NcVariable
    implicit none
    type(NcVariable), intent(in) :: var !< NetCDF variable to check
    character(len=256) :: tmp_str

    is_y_axis = .false.
    if (var%hasAttribute("standard_name")) then
      call var%getAttribute("standard_name", tmp_str)
      if (trim(tmp_str) == "projection_y_coordinate") is_y_axis = .true.
    else if (var%hasAttribute("axis")) then
      call var%getAttribute("axis", tmp_str)
      if (trim(tmp_str) == "Y") is_y_axis = .true.
    else if (var%hasAttribute("_CoordinateAxisType")) then
      call var%getAttribute("_CoordinateAxisType", tmp_str)
      if (trim(tmp_str) == "GeoY") is_y_axis = .true.
    end if
  end function is_y_axis

!> \brief check if given variable is a lon coordinate.
  !> \return `logical :: is_lon_coord`
  !> \authors Sebastian Müller
  !> \date Mar 2024
  logical function is_lon_coord(var)
    use mo_netcdf, only : NcVariable
    implicit none
    type(NcVariable), intent(in) :: var !< NetCDF variable to check
    character(len=256) :: tmp_str

    is_lon_coord = .false.
    if (var%hasAttribute("standard_name")) then
      call var%getAttribute("standard_name", tmp_str)
      if (trim(tmp_str) == "longitude") is_lon_coord = .true.
    else if (var%hasAttribute("units")) then
      call var%getAttribute("units", tmp_str)
      if (trim(tmp_str) == "degreeE") is_lon_coord = .true.
      if (trim(tmp_str) == "degree_E") is_lon_coord = .true.
      if (trim(tmp_str) == "degree_east") is_lon_coord = .true.
      if (trim(tmp_str) == "degreesE") is_lon_coord = .true.
      if (trim(tmp_str) == "degrees_E") is_lon_coord = .true.
      if (trim(tmp_str) == "degrees_east") is_lon_coord = .true.
    else if (var%hasAttribute("_CoordinateAxisType")) then
      call var%getAttribute("_CoordinateAxisType", tmp_str)
      if (trim(tmp_str) == "Lon") is_lon_coord = .true.
    else if (var%hasAttribute("long_name")) then
      call var%getAttribute("long_name", tmp_str)
      if (trim(tmp_str) == "longitude") is_lon_coord = .true.
    end if

  end function is_lon_coord

  !> \brief check if given variable is a lat coordinate.
  !> \return `logical :: is_lat_coord`
  !> \authors Sebastian Müller
  !> \date Mar 2024
  logical function is_lat_coord(var)
    use mo_netcdf, only : NcVariable
    implicit none
    type(NcVariable), intent(in) :: var !< NetCDF variable to check
    character(len=256) :: tmp_str

    is_lat_coord = .false.
    if (var%hasAttribute("standard_name")) then
      call var%getAttribute("standard_name", tmp_str)
      if (trim(tmp_str) == "latitude") is_lat_coord = .true.
    else if (var%hasAttribute("units")) then
      call var%getAttribute("units", tmp_str)
      if (trim(tmp_str) == "degreeN") is_lat_coord = .true.
      if (trim(tmp_str) == "degree_N") is_lat_coord = .true.
      if (trim(tmp_str) == "degree_north") is_lat_coord = .true.
      if (trim(tmp_str) == "degreesN") is_lat_coord = .true.
      if (trim(tmp_str) == "degrees_N") is_lat_coord = .true.
      if (trim(tmp_str) == "degrees_north") is_lat_coord = .true.
    else if (var%hasAttribute("_CoordinateAxisType")) then
      call var%getAttribute("_CoordinateAxisType", tmp_str)
      if (trim(tmp_str) == "Lat") is_lat_coord = .true.
    else if (var%hasAttribute("long_name")) then
      call var%getAttribute("long_name", tmp_str)
      if (trim(tmp_str) == "latitude") is_lat_coord = .true.
    end if
  end function is_lat_coord


  !> \brief Read NetCDF header information (matching read_header_ascii signature)
  !> \details Reads grid metadata (ncols, nrows, corner, cellsize, nodata) from a NetCDF file.
  !> \authors Sebastian Müller, Simon Lüdke
  !> \date Mar 2024 / Feb 2025
  subroutine get_header_info_from_nc(ncname, fileunit, header_ncols, header_nrows, header_xllcorner, header_yllcorner, &
                                     header_cellsize, header_nodata, mask, var)
    use mo_netcdf, only : NcDataset, NcVariable, NcDimension
    use mo_utils, only : is_close
    use mo_string_utils, only : num2str
    implicit none
    character(*), intent(in) :: ncname !< NetCDF filename
    integer(i4), intent(in) :: fileunit !< Unused (kept for signature compatibility)
    integer(i4), intent(out) :: header_ncols !< number of columns
    integer(i4), intent(out) :: header_nrows !< number of rows
    real(dp), intent(out) :: header_xllcorner !< lower left x
    real(dp), intent(out) :: header_yllcorner !< lower left y
    real(dp), intent(out) :: header_cellsize !< cell size
    real(dp), intent(out) :: header_nodata !< nodata value
    logical, optional, allocatable, dimension(:,:), intent(out) :: mask !< mask 
    character(*), intent(in), optional :: var !< optional variable name

    type(NcDataset) :: nc
    type(NcVariable) :: ncvar, xvar, yvar
    type(NcDimension), dimension(:), allocatable :: dims
    integer(i4) :: rnk, y_dir
    integer(i4) :: bottom_up, top_down
    real(dp) ::  cs_x, cs_y, tol
    logical :: y_inc, x_sph, y_sph, x_cart, y_cart, flip_y
    integer(i4) :: keep_y
    character(len=256) :: varName

    keep_y = -1_i4 
    y_dir = keep_y

    ! set defaults
    tol = 1.e-7_dp
    bottom_up = 1_i4
    top_down = 0_i4

    nc = NcDataset(ncname, "r")

    if (present(var)) then
      varName = trim(var)
    else
      varName = determine_data_var_name(nc, ncname)
    end if

    ncvar = nc%getVariable(varName)
    rnk = ncvar%getRank()
    if (rnk < 2) call error_message("grid % from_netcdf: given variable has too few dimensions: ", trim(nc%fname), ":", trim(varName))

    dims = ncvar%getDimensions()
    header_ncols = dims(1)%getLength()
    header_nrows = dims(2)%getLength()
    xvar = nc%getVariable(trim(dims(1)%getName()))
    yvar = nc%getVariable(trim(dims(2)%getName()))

    ! check if x/y axis are x/y/lon/lat by standard_name, units, axistype or long_name
    if (is_x_axis(yvar).or.is_lon_coord(yvar).or.is_y_axis(xvar).or.is_lat_coord(xvar)) &
      call error_message("grid % from_netcdf: variable seems to have wrong axis order (not y-x): ", trim(nc%fname), ":", trim(varName))

    x_cart = is_x_axis(xvar)
    y_cart = is_y_axis(yvar)
    x_sph = is_lon_coord(xvar)
    y_sph = is_lat_coord(yvar)

    if (.not.(x_cart.or.x_sph)) &
      call error_message("grid % from_netcdf: can't determine coordinate system from x-axis: ", trim(nc%fname), ":", trim(varName))
    if (.not.(y_cart.or.y_sph)) &
      call error_message("grid % from_netcdf: can't determine coordinate system from y-axis: ", trim(nc%fname), ":", trim(varName))
    if (.not.(x_sph.eqv.y_sph)) &
      call error_message("grid % from_netcdf: x and y axis seem to have different coordinate systems: ", trim(nc%fname), ":", trim(varName))

    ! check axis uniformity and monotonicity
    call check_uniform_axis(xvar, cellsize=cs_x, origin=header_xllcorner, tol=tol)
    call check_uniform_axis(yvar, cellsize=cs_y, origin=header_yllcorner, increasing=y_inc, tol=tol)
    if (y_dir == keep_y) then
      y_dir = top_down
      if (y_inc) y_dir = bottom_up
    end if
    ! check y_dir
    if (.not.any(y_dir==[bottom_up, top_down])) &
      call error_message("grid % from_netcdf: y-direction not valid: ", trim(num2str(y_dir)))

    ! warn about flipping if present axis is not in desired direction
    flip_y = y_inc.neqv.(y_dir==bottom_up)
    if (flip_y) then
      print *, "grid % from_netcdf: y axis direction is oposite to desired one (inefficient flipping). ", &
                        "You could flip the file beforehand with: 'cdo invertlat <ifile> <ofile>'. ", trim(nc%fname), ":", trim(varName)
    end if
    ! check cellsize in x and y direction
    if (.not.is_close(cs_x, cs_y, rtol=0.0_dp, atol=tol)) &
      call error_message("grid % from_netcdf: x and y axis have different cell sizes: ", trim(nc%fname), ":", trim(varName))
    header_cellsize = cs_x

    ! Determine no-data value
    if (ncvar%hasAttribute("_FillValue")) then
      call ncvar%getAttribute("_FillValue", header_nodata)
    else if (ncvar%hasAttribute("missing_value")) then
      call ncvar%getAttribute("missing_value", header_nodata)
    else
      call error_message('***ERROR: read_nc_header: missing _FillValue or missing_value attribute')
    end if

    call nc%close()
  end subroutine get_header_info_from_nc

  function determine_data_var_name(nc, ncname) result(varName)
    use mo_netcdf, only : NcDataset, NcVariable
    use mo_message, only : error_message
    use mo_os, only : path_stem
    class(NcDataset), intent(in) :: nc
    character(len=*), intent(in) :: ncname
    character(len=256) :: varName

    type(NcVariable), dimension(:), allocatable :: vars
    character(len=256), dimension(:), allocatable :: data_var_names
    integer(i4) :: i, data_var_count, data_var_index
    character(len=256) :: stem

    varName = ''
    vars = nc%getVariables()

    data_var_count = 0
    do i = 1, size(vars)
      if (vars(i)%getNoDimensions() >= 2) data_var_count = data_var_count + 1
    end do

    if (data_var_count == 0) then
      call error_message('read_spatial_data_nc: could not determine variable name for file: ' // trim(ncname))
    end if

    allocate(data_var_names(data_var_count))
    data_var_index = 0
    do i = 1, size(vars)
      if (vars(i)%getNoDimensions() >= 2) then
        data_var_index = data_var_index + 1
        data_var_names(data_var_index) = trim(vars(i)%getName())
      end if
    end do

    if (data_var_count == 1) then
      varName = trim(data_var_names(1))
      return
    end if

    stem = trim(path_stem(ncname))
    do i = 1, data_var_count
      if ((trim(data_var_names(i)) == stem) .or. (trim(data_var_names(i)) == 'data')) then
        varName = trim(data_var_names(i))
        return
      end if
    end do

    call error_message('read_spatial_data_nc: variable name could not be determined for file: ' // trim(ncname))
  end function determine_data_var_name


  !    NAME
  !        read_spatial_data_nc_i4

  !    PURPOSE
  !>       \brief Read file from filepath. If there is a nc file with the correct name it is prefered over the asci file.

  !>       \details TODO: add description

  !    HISTORY
  !>       \authors Simon Lüdke

  !>       \date June 2025
  subroutine read_spatial_data_nc_i4(ncname, data, maskout, ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value, varName)
    use mo_netcdf, only : NcDataset, NcVariable
    use mo_message, only : error_message
    implicit none

    character(len=*), intent(in) :: ncname
    character(len=*), intent(in), optional :: varName
    integer(i4), dimension(:, :), allocatable, intent(out) :: data
    logical, optional, allocatable, dimension(:, :), intent(out) :: maskout
    integer(i4), intent(out) :: ncols, nrows
    real(dp), intent(out) :: xllcorner, yllcorner, cellsize
    real(dp), intent(out) :: nodata_value

    type(NcDataset) :: nc
    type(NcVariable) :: var
    integer(i4), allocatable :: var_shape(:), start(:), cnt(:)
    character(len=256) :: varName_local

    ! Open NetCDF dataset
    nc = NcDataset(ncname, "r")

    if (present(varName)) then
      varName_local = trim(varName)
    else
      varName_local = determine_data_var_name(nc, ncname)
    end if

    ! Extract header info
    call get_header_info_from_nc(ncname, 0_i4, ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value, var=varName_local)

    ! Retrieve variable
    var = nc%getVariable(trim(varName_local))

    ! Determine shape and prepare read extents
    var_shape = var%getShape()
    allocate(start(size(var_shape)), source=1_i4)
    allocate(cnt(size(var_shape)), source=1_i4)
    cnt(1:2) = var_shape(1:2)

    ! Determine no-data value
    if (var%hasAttribute("_FillValue")) then
      call var%getAttribute("_FillValue", nodata_value)
    else if (var%hasAttribute("missing_value")) then
      call var%getAttribute("missing_value", nodata_value)
    else
      call error_message('***ERROR: read_nc_i4_data: missing _FillValue or missing_value attribute')
    end if

    ! Allocate and read data
    allocate(data(var_shape(1), var_shape(2)))
    call var%getData(data, start=start, cnt=cnt)

    if (present(maskout)) then
      allocate(maskout(var_shape(1), var_shape(2)))
      maskout = abs(real(data, dp) - nodata_value) .gt. tiny(1.0_dp)
    end if

    call nc%close()
  end subroutine read_spatial_data_nc_i4
  
  
  !    NAME
  !        read_spatial_data_nc_dp

  !    PURPOSE
  !>       \brief Read file from filepath. If there is a nc file with the correct name it is prefered over the asci file.

  !>       \details TODO: add description

  !    HISTORY
  !>       \authors Simon Lüdke

  !>       \date June 2025
  subroutine read_spatial_data_nc_dp(ncname, data, maskout, ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value, varName)
    use mo_netcdf, only : NcDataset, NcVariable
    use mo_message, only : error_message
    implicit none

    character(len=*), intent(in) :: ncname
    character(len=*), intent(in), optional :: varName
    real(dp), dimension(:, :), allocatable, intent(out) :: data
    logical, optional, allocatable, dimension(:, :), intent(out) :: maskout
    integer(i4), intent(out) :: ncols, nrows
    real(dp), intent(out) :: xllcorner, yllcorner, cellsize
    real(dp), intent(out) :: nodata_value

    type(NcDataset) :: nc
    type(NcVariable) :: var
    integer(i4), allocatable :: var_shape(:), start(:), cnt(:)
    character(len=256) :: varName_local

    ! Open NetCDF dataset
    nc = NcDataset(ncname, "r")

    if (present(varName)) then
      varName_local = trim(varName)
    else
      varName_local = determine_data_var_name(nc, ncname)
    end if

    ! Extract header info
    call get_header_info_from_nc(ncname, 0_i4, ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value, var=varName_local)

    ! Retrieve variable
    var = nc%getVariable(trim(varName_local))

    ! Determine shape and prepare read extents
    var_shape = var%getShape()
    allocate(start(size(var_shape)), source=1_i4)
    allocate(cnt(size(var_shape)), source=1_i4)
    cnt(1:2) = var_shape(1:2)

    ! Determine no-data value
    if (var%hasAttribute("_FillValue")) then
      call var%getAttribute("_FillValue", nodata_value)
    else if (var%hasAttribute("missing_value")) then
      call var%getAttribute("missing_value", nodata_value)
    else
      call error_message('***ERROR: read_nc_i4_data: missing _FillValue or missing_value attribute')
    end if

    ! Allocate and read data
    allocate(data(var_shape(1), var_shape(2)))
    call var%getData(data, start=start, cnt=cnt)

    if (present(maskout)) then
      allocate(maskout(var_shape(1), var_shape(2)))
      maskout = abs(data - nodata_value) .gt. tiny(1.0_dp)
    end if

    call nc%close()
  end subroutine read_spatial_data_nc_dp

  
  
  !    NAME
  !        read_spatial_data_nc_or_ascii_dp

  !    PURPOSE
  !>       \brief Read file from filepath. If there is a nc file with the correct name it is prefered over the asci file.

  !>       \details TODO: add description

  !    HISTORY
  !>       \authors Simon Lüdke

  !>       \date June 2025

  subroutine read_spatial_data_nc_or_ascii_dp(filepath, fileunit, header_ncols, header_nrows, header_xllcorner, &
                                       header_yllcorner, header_cellsize, data, maskout, &
                                       out_nCols, out_nRows, out_xllcorner, out_yllcorner, out_cellsize, out_nodata_value)
    use mo_os, only : path_root, path_isfile
        implicit none

    ! filename with location
    character(len = *), intent(in) :: filepath

    ! unit for opening the file
    integer(i4), intent(in) :: fileunit

    ! number of rows of data fields:
    integer(i4), intent(in) :: header_nRows

    ! number of columns of data fields:
    integer(i4), intent(in) :: header_nCols

    ! header read in lower left corner
    real(dp), intent(in) :: header_xllcorner

    ! header read in lower left corner
    real(dp), intent(in) :: header_yllcorner

    ! header read in cellsize
    real(dp), intent(in) :: header_cellsize

    ! data
    real(dp), dimension(:, :), allocatable, intent(out) :: data

    ! mask
    logical, optional, dimension(:, :), allocatable, intent(out) :: maskout

    ! number of rows of data fields:
    integer(i4), optional, intent(out) :: out_nRows

    ! number of columns of data fields:
    integer(i4), optional, intent(out) :: out_nCols

    ! header read in lower left corner
    real(dp), optional, intent(out) :: out_xllcorner

    ! header read in lower left corner
    real(dp), optional, intent(out) :: out_yllcorner

    ! header read in cellsize
    real(dp), optional, intent(out) :: out_cellsize

    ! 
    real(dp), optional, intent(out) :: out_nodata_value

    ! file exists 
    real(dp) :: nodata_value
    integer(i4) :: nrows, ncols
    real(dp) :: xllcorner, yllcorner, cellsize
    character(len=:), allocatable :: ncname

    ! Prefer explicit NetCDF path if it already ends with .nc, otherwise use sidecar .nc
    if (len_trim(filepath) >= 3) then
      if ((filepath(len_trim(filepath)-2:len_trim(filepath)) == '.nc') .or. (filepath(len_trim(filepath)-2:len_trim(filepath)) == '.NC')) then
        ncname = trim(filepath)
      else
        ncname = path_root(filepath) // '.nc'
      end if
    else
      ncname = path_root(filepath) // '.nc'
    end if

    ! preferably use nc file if it exists alternatively the asci version
    ! print *, "Check if ", ncname, " existis: ", path_isfile(ncname)
    if (path_isfile(ncname)) then
      print *, "Read nc file: ", ncname
      call read_spatial_data_nc(ncname, data, maskout, ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value)
    else
    print *, "Read ascii file: ", filepath
      if ((header_ncols /= 0_i4) .and. (header_nrows /= 0_i4)) then
        ncols = header_ncols
        nrows = header_nrows
        yllcorner = header_yllcorner
        xllcorner = header_xllcorner
        cellsize = header_cellsize
      else 
        call read_header_ascii(filepath, fileunit, &
              ncols, nrows, xllcorner, &
              yllcorner, cellsize, nodata_value)

      end if
      ! print *, "run ascii dp reading with ", filepath
      call read_spatial_data_ascii(filepath, fileunit, ncols, nrows, xllcorner, &
                                    yllcorner, cellsize, data, maskout)
    end if 
    !  if header information is needed as output it is saved to the output variables
    if (present(out_nCols)) & 
      out_nCols = ncols
    if (present(out_nRows)) & 
      out_nRows = nrows
    if (present(out_xllcorner)) & 
      out_xllcorner = xllcorner
    if (present(out_yllcorner)) & 
      out_yllcorner = yllcorner
    if (present(out_cellsize)) & 
      out_cellsize = cellsize
    if (present(out_nodata_value)) & 
      out_nodata_value = nodata_value

  end subroutine read_spatial_data_nc_or_ascii_dp

  !    NAME
  !        read_spatial_data_nc_or_ascii_i4

  !    PURPOSE
  !>       \brief Read file from filepath. If there is a nc file with the correct name it is prefered over the asci file.

  !>       \details TODO: add description

  !    HISTORY
  !>       \authors Simon Lüdke

  !>       \date June 2025

  subroutine read_spatial_data_nc_or_ascii_i4(filepath, fileunit, header_ncols, header_nrows, header_xllcorner, &
                                       header_yllcorner, header_cellsize, data, maskout, &
                                       out_nCols, out_nRows, out_xllcorner, out_yllcorner, out_cellsize, out_nodata_value)
    use mo_os, only : path_root, path_isfile
    implicit none

    ! filename with location
    character(len = *), intent(in) :: filepath

    ! unit for opening the file
    integer(i4), intent(in) :: fileunit

    ! number of rows of data fields:
    integer(i4), intent(in) :: header_nRows

    ! number of columns of data fields:
    integer(i4), intent(in) :: header_nCols

    ! header read in lower left corner
    real(dp), intent(in) :: header_xllcorner

    ! header read in lower left corner
    real(dp), intent(in) :: header_yllcorner

    ! header read in cellsize
    real(dp),  intent(in) :: header_cellsize

    ! data
    integer(i4), dimension(:, :), allocatable, intent(out) :: data

    ! mask
    logical, optional, dimension(:, :), allocatable, intent(out) :: maskout

    ! number of rows of data fields:
    integer(i4), optional, intent(out) :: out_nRows

    ! number of columns of data fields:
    integer(i4), optional, intent(out) :: out_nCols

    ! header read in lower left corner
    real(dp), optional, intent(out) :: out_xllcorner

    ! header read in lower left corner
    real(dp), optional, intent(out) :: out_yllcorner

    ! header read in cellsize
    real(dp), optional, intent(out) :: out_cellsize
    
    real(dp), optional, intent(out) :: out_nodata_value

    ! file exists 
    real(dp) :: nodata_value
    integer(i4) :: nrows, ncols
    real(dp) :: xllcorner, yllcorner, cellsize
    character(len=:), allocatable :: ncname

    ! Prefer explicit NetCDF path if it already ends with .nc, otherwise use sidecar .nc
    if (len_trim(filepath) >= 3) then
      if ((filepath(len_trim(filepath)-2:len_trim(filepath)) == '.nc') .or. (filepath(len_trim(filepath)-2:len_trim(filepath)) == '.NC')) then
        ncname = trim(filepath)
      else
        ncname = path_root(filepath) // '.nc'
      end if
    else
      ncname = path_root(filepath) // '.nc'
    end if

    ! preferably use nc file if it exists alternatively the asci version
    ! print *, "Check if ", ncname, " existis: ", path_isfile(ncname)
    if (path_isfile(ncname)) then
      print *, "Read nc file: ", ncname
      call read_spatial_data_nc(ncname, data, maskout, ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value)
    else
      print *, "Read ascii file: ", filepath
      if ((header_ncols /= 0_i4) .and. (header_nrows /= 0_i4)) then
        ncols = header_ncols
        nrows = header_nrows
        yllcorner = header_yllcorner
        xllcorner = header_xllcorner
        cellsize = header_cellsize
      else 
        call read_header_ascii(filepath, fileunit, &
              ncols, nrows, xllcorner, &
              yllcorner, cellsize, nodata_value)

      end if
      ! print *, "run ascii i4 reading with ", filepath, " as ", ncname, " does not exists"
      call read_spatial_data_ascii(filepath, fileunit, ncols, nrows, xllcorner, &
                                    yllcorner, cellsize, data, maskout)
    end if 
    !  if header information is needed as output it is saved to the output variables
    if (present(out_nCols)) & 
      out_nCols = ncols
    if (present(out_nRows)) & 
      out_nRows = nrows
    if (present(out_xllcorner)) & 
      out_xllcorner = xllcorner
    if (present(out_yllcorner)) & 
      out_yllcorner = yllcorner
    if (present(out_cellsize)) & 
      out_cellsize = cellsize
    if (present(out_nodata_value)) & 
      out_nodata_value = nodata_value
  end subroutine read_spatial_data_nc_or_ascii_i4


END MODULE mo_read_spatial_data
