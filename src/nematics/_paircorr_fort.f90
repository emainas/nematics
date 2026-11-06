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


  subroutine report_paircorr_update(molpos, molz, dpos, ddir, box, glob_dir, &
                                    close_images, dzz, &
                                    global_par100, global_par200, global_par300, &
                                    local_par100, local_par200, local_par300, &
                                    global_per, local_per, cn_rad)
    ! Accumulate the correlation functions shown in the 5CB section of
    ! try_to_reproduce.pdf.  Every histogram stores four channels:
    !   1 raw pair count, 2 g0 numerator, 3 g1 numerator, 4 g2 numerator.
    ! The g0/g1/g2 channels include their ideal-gas geometric normalization.
    implicit none

    real(8), intent(in) :: molpos(:,:), molz(:,:), dpos(:,:), ddir(:,:)
    real(8), intent(in) :: box(3), glob_dir(3), close_images(:,:), dzz
    real(8), intent(inout) :: global_par100(:,:), global_par200(:,:), global_par300(:,:)
    real(8), intent(inout) :: local_par100(:,:), local_par200(:,:), local_par300(:,:)
    real(8), intent(inout) :: global_per(:,:), local_per(:,:), cn_rad(:,:)

    integer :: nmm, i, j, k
    real(8) :: rho, pi, par_dw, local_dw, radial_dw
    real(8) :: rij0(3), rij(3), dij0(3), dij(3)
    real(8) :: rijsq, dijsq, rz, rv, local_z, local_r
    real(8) :: dotmol, p2mol, dotdip, p2dip
    real(8) :: global_par_norm, local_par_norm
    real(8) :: global_per_norm, local_per_norm, radial_norm

    nmm = size(molpos, dim=1)
    if (nmm <= 1) return

    pi = acos(-1.0d0)
    rho = dble(nmm) * dble(nmm - 1) / (box(1) * box(2) * box(3))
    par_dw = 5.0d0 / dble(size(global_par100, dim=1))
    local_dw = 10.0d0 / dble(size(local_par100, dim=1))
    radial_dw = 5.0d0 / dble(size(global_per, dim=1))

    do i = 1, nmm - 1
      do j = i + 1, nmm
        rij0(:) = molpos(j,:) - molpos(i,:)
        dij0(:) = dpos(j,:) - dpos(i,:)

        dotmol = dot_product(molz(i,:), molz(j,:))
        p2mol = 1.5d0 * dotmol * dotmol - 0.5d0
        dotdip = dot_product(ddir(i,:), ddir(j,:))
        p2dip = 1.5d0 * dotdip * dotdip - 0.5d0

        do k = 1, size(close_images, dim=1)
          rij(:) = rij0(:) + close_images(k,:) * box(:)
          rijsq = dot_product(rij, rij)

          ! The largest molecular selection is |z|<5 nm and r_perp<3 nm.
          if (rijsq <= 34.0d0) then
            rz = dot_product(rij, glob_dir)
            rv = sqrt(max(0.0d0, rijsq - rz * rz))

            if (abs(rz) < 5.0d0) then
              if (rv < 1.0d0) then
                global_par_norm = rho * pi * 1.0d0 * par_dw
                call update_report_hist(abs(rz), 0.0d0, 5.0d0, global_par_norm, &
                                        dotmol, p2mol, global_par100)
              end if
              if (rv < 2.0d0) then
                global_par_norm = rho * pi * 4.0d0 * par_dw
                call update_report_hist(abs(rz), 0.0d0, 5.0d0, global_par_norm, &
                                        dotmol, p2mol, global_par200)
              end if
              if (rv < 3.0d0) then
                global_par_norm = rho * pi * 9.0d0 * par_dw
                call update_report_hist(abs(rz), 0.0d0, 5.0d0, global_par_norm, &
                                        dotmol, p2mol, global_par300)
              end if
            end if

            if (abs(rz) < dzz .and. rv < 5.0d0) then
              global_per_norm = 2.0d0 * pi * rho * dzz * radial_dw * rv
              call update_report_hist(rv, 0.0d0, 5.0d0, global_per_norm, &
                                      dotmol, p2mol, global_per)
            end if

            ! Local molecular-axis correlations are ordered: both i-centered
            ! and j-centered descriptions of every unordered pair are added.
            local_z = dot_product(rij, molz(i,:))
            local_r = sqrt(max(0.0d0, rijsq - local_z * local_z))
            if (abs(local_z) < 5.0d0) then
              if (local_r < 1.0d0) then
                local_par_norm = rho * pi * 1.0d0 * local_dw
                call update_report_hist(local_z, -5.0d0, 5.0d0, local_par_norm, &
                                        dotmol, p2mol, local_par100)
              end if
              if (local_r < 2.0d0) then
                local_par_norm = rho * pi * 4.0d0 * local_dw
                call update_report_hist(local_z, -5.0d0, 5.0d0, local_par_norm, &
                                        dotmol, p2mol, local_par200)
              end if
              if (local_r < 3.0d0) then
                local_par_norm = rho * pi * 9.0d0 * local_dw
                call update_report_hist(local_z, -5.0d0, 5.0d0, local_par_norm, &
                                        dotmol, p2mol, local_par300)
              end if
            end if
            if (abs(local_z) < dzz .and. local_r < 5.0d0) then
              local_per_norm = 4.0d0 * pi * rho * dzz * radial_dw * local_r
              call update_report_hist(local_r, 0.0d0, 5.0d0, local_per_norm, &
                                      dotmol, p2mol, local_per)
            end if

            local_z = dot_product(-rij, molz(j,:))
            local_r = sqrt(max(0.0d0, rijsq - local_z * local_z))
            if (abs(local_z) < 5.0d0) then
              if (local_r < 1.0d0) then
                local_par_norm = rho * pi * 1.0d0 * local_dw
                call update_report_hist(local_z, -5.0d0, 5.0d0, local_par_norm, &
                                        dotmol, p2mol, local_par100)
              end if
              if (local_r < 2.0d0) then
                local_par_norm = rho * pi * 4.0d0 * local_dw
                call update_report_hist(local_z, -5.0d0, 5.0d0, local_par_norm, &
                                        dotmol, p2mol, local_par200)
              end if
              if (local_r < 3.0d0) then
                local_par_norm = rho * pi * 9.0d0 * local_dw
                call update_report_hist(local_z, -5.0d0, 5.0d0, local_par_norm, &
                                        dotmol, p2mol, local_par300)
              end if
            end if
            if (abs(local_z) < dzz .and. local_r < 5.0d0) then
              local_per_norm = 4.0d0 * pi * rho * dzz * radial_dw * local_r
              call update_report_hist(local_r, 0.0d0, 5.0d0, local_per_norm, &
                                      dotmol, p2mol, local_per)
            end if
          end if

          dij(:) = dij0(:) + close_images(k,:) * box(:)
          dijsq = dot_product(dij, dij)
          if (dijsq < 25.0d0) then
            radial_norm = 2.0d0 * pi * rho * radial_dw * dijsq
            call update_report_hist(sqrt(dijsq), 0.0d0, 5.0d0, radial_norm, &
                                    dotdip, p2dip, cn_rad)
          end if
        end do
      end do
    end do
  end subroutine report_paircorr_update


  subroutine update_report_hist(x, minv, maxv, mult, dotval, p2val, accum)
    implicit none
    real(8), intent(in) :: x, minv, maxv, mult, dotval, p2val
    real(8), intent(inout) :: accum(:,:)
    integer :: ib, nbins

    if (x < minv .or. x >= maxv .or. mult <= 0.0d0) return
    nbins = size(accum, dim=1)
    ib = 1 + int(dble(nbins) * (x - minv) / (maxv - minv))
    if (ib < 1 .or. ib > nbins) return

    accum(ib,1) = accum(ib,1) + 1.0d0
    accum(ib,2) = accum(ib,2) + 1.0d0 / mult
    accum(ib,3) = accum(ib,3) + dotval / mult
    accum(ib,4) = accum(ib,4) + p2val / mult
  end subroutine update_report_hist

end module paircorr_fort_mod
