module paircorr_fort_mod
  implicit none
contains

  subroutine paircorr_update(pos, yax, zax, box, glob_dir, close_images, dzz, &
                             par_min, par_max, per_min, per_max, rad_min, rad_max, &
                             hist_par100, histy_par100, hist_par200, histy_par200, &
                             hist_par300, histy_par300, hist_parall, histy_parall, &
                             hist_per, histy_per, hist_rad, histy_rad)
    implicit none

    real(8), intent(in) :: pos(:,:), yax(:,:), zax(:,:), box(3), glob_dir(3)
    real(8), intent(in) :: close_images(:,:)
    real(8), intent(in) :: dzz
    real(8), intent(in) :: par_min, par_max, per_min, per_max, rad_min, rad_max

    real(8), intent(inout) :: hist_par100(:), histy_par100(:,:)
    real(8), intent(inout) :: hist_par200(:), histy_par200(:,:)
    real(8), intent(inout) :: hist_par300(:), histy_par300(:,:)
    real(8), intent(inout) :: hist_parall(:), histy_parall(:,:)
    real(8), intent(inout) :: hist_per(:), histy_per(:,:)
    real(8), intent(inout) :: hist_rad(:), histy_rad(:,:)

    integer :: nmm, i, j, k
    real(8) :: rho, dnz, dnr, dnr2
    real(8) :: rij0(3), rij(3), rijsq, rz, rv, abs_rz
    real(8) :: ww(6), ww1, ww2, ww3
    logical :: have_ww

    nmm = size(pos, dim=1)
    if (nmm <= 1) return

    rho = dble(nmm) * dble(nmm - 1) / (box(1) * box(2) * box(3))

    dnz = rho * acos(-1.0d0) * (par_max - par_min) / dble(size(hist_parall))
    dnr = 0.5d0 * rho * acos(-1.0d0) * 2.0d0 * (2.0d0 * dzz) * (per_max - per_min) / dble(size(hist_per))
    dnr2 = 0.5d0 * rho * acos(-1.0d0) * 4.0d0 * (rad_max - rad_min) / dble(size(hist_rad))

    do i = 1, nmm - 1
      do j = i + 1, nmm
        rij0(:) = pos(j, :) - pos(i, :)
        have_ww = .false.

        do k = 1, size(close_images, dim=1)
          rij(:) = rij0(:) + close_images(k, :) * box(:)
          rijsq = rij(1) * rij(1) + rij(2) * rij(2) + rij(3) * rij(3)
          if (rijsq > 2.0d0 * par_max * par_max) cycle

          rz = rij(1) * glob_dir(1) + rij(2) * glob_dir(2) + rij(3) * glob_dir(3)
          rv = sqrt(max(0.0d0, rijsq - rz * rz))

          if (.not. have_ww) then
            ww1 = zax(i,1) * zax(j,1) + zax(i,2) * zax(j,2) + zax(i,3) * zax(j,3)
            ww2 = (zax(i,1) * glob_dir(1) + zax(i,2) * glob_dir(2) + zax(i,3) * glob_dir(3)) * &
                  (zax(j,1) * glob_dir(1) + zax(j,2) * glob_dir(2) + zax(j,3) * glob_dir(3))
            ww3 = yax(i,1) * yax(j,1) + yax(i,2) * yax(j,2) + yax(i,3) * yax(j,3)

            ww(1) = 1.0d0
            ww(2) = ww1
            ww(3) = ww1 * ww1
            ww(4) = ww2
            ww(5) = ww2 * ww2
            ww(6) = ww3
            have_ww = .true.
          end if

          abs_rz = abs(rz)

          if (rv < 1.0d0) call update_hist(abs_rz, dnz * 1.0d0, ww, par_min, par_max, hist_par100, histy_par100)
          if (rv < 2.0d0) call update_hist(abs_rz, dnz * 4.0d0, ww, par_min, par_max, hist_par200, histy_par200)
          if (rv < 3.0d0) call update_hist(abs_rz, dnz * 9.0d0, ww, par_min, par_max, hist_par300, histy_par300)
          if (rv < 5.0d0) call update_hist(abs_rz, dnz * 25.0d0, ww, par_min, par_max, hist_parall, histy_parall)

          if (abs_rz < dzz) call update_hist(rv, dnr * rv, ww, per_min, per_max, hist_per, histy_per)

          call update_hist(sqrt(rijsq), dnr2 * rijsq, ww, rad_min, rad_max, hist_rad, histy_rad)
        end do
      end do
    end do
  end subroutine paircorr_update

  subroutine update_hist(x, mult, ww, minv, maxv, hist, histy)
    implicit none
    real(8), intent(in) :: x, mult, ww(:), minv, maxv
    real(8), intent(inout) :: hist(:), histy(:,:)

    integer :: ib, nbins, b
    real(8) :: dw

    if (x < minv .or. x > maxv) return

    nbins = size(hist)
    dw = maxv - minv
    ib = 1 + int(dble(nbins) * (x - minv) / dw)
    if (ib > nbins) ib = nbins
    if (ib < 1) return

    hist(ib) = hist(ib) + 1.0d0
    if (mult <= 0.0d0) return

    do b = 1, min(size(ww), size(histy, dim=2))
      histy(ib, b) = histy(ib, b) + ww(b) / mult
    end do
  end subroutine update_hist

end module paircorr_fort_mod
