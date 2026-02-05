module commondata
    implicit none
    integer :: NAtPerMol
    !parameter (NAtPerMol=79)   !C4OPHODBcP
    parameter (NAtPerMol=38)   !5CB
    
    real :: AtPos(1000, NAtPerMol, 3)   ! Positions of all atoms (C, N)
    real :: CBCM(1000,3)    ! CM of the CB units
    real :: CBDIRZ(1000,3), CBDIRY(1000,3), CBDIRX(1000,3) !Axes frame of the cb units
    real :: MolXax(1000, 3),MolYax(1000, 3),MolZax(1000, 3), MolCM(1000,3)
    real :: BOX(3)
    real :: CNDIR(1000,3)    ! Dir of the cn bonds
    real :: CNCM(1000,3)    ! Dir of the cn bonds
    real :: CHBOND(1000, 30, 3), CHBONDLEN(1000, 30)    !CH BONDS
    real :: dpos(1000,3), ddir(1000,3)
    
    real :: MSQ(10)
    
    REAL, ALLOCATABLE :: Hist(:,:), AVHist(:,:), MC_PerBin(:)
    real DC
    INTEGER :: NHIST, ibin
    real om(3,3), avom(3,3), evalz(3)
    REAL Pi, TwoPi, FourPi, Pi_over_4, Pi_over_2

    integer NMOLS, NMM
    Integer :: CloseIm(27,3)

    !!!!!!!!!!!!!!!!!!!!!!!!!
    REAL :: GlobDir(3), AV_GlobDir(3)
    REAL :: DeltaZ=0.05D0, DeltaZ2=0.1D0, DeltaR=0.04D0, DeltaR2=0.02D0
    REAL, ALLOCATABLE:: HISTZ(:,:), AV_HISTZ(:,:)
    REAL, ALLOCATABLE:: HISTR(:,:), AV_HISTR(:,:)

    REAL, ALLOCATABLE:: HISTRMOL(:,:), AV_HISTRMOL(:,:)
    REAL, ALLOCATABLE:: HISTZMOL(:,:), AV_HISTZMOL(:,:)

    REAL, ALLOCATABLE:: HISTTT(:,:,:), AV_HISTTT(:,:,:)
    REAL, ALLOCATABLE:: HISTTg(:,:,:), AV_HISTTg(:,:,:)
    
    REAL, ALLOCATABLE:: HISTR2(:,:), AV_HISTR2(:,:)
    REAL, ALLOCATABLE:: HISTRg(:,:), AV_HISTRg(:,:)

    INTEGER, ALLOCATABLE :: NmolsAtZ(:), MolIdAtZ(:,:)
    REAL, ALLOCATABLE :: Slab(:,:)
    INTEGER :: NBINZ, NBINZ2, NBINR, NBINR2
    !REAL :: HistSamples = 0.0D0, HistSamplesMOL=0.0D0
    CHARACTER(180) :: RDFILE, RDFILE2, RDFILEMOL, RDFILE2MOL
    REAL :: HistSamples = 0.0D0, MOLHistSamples=0.0D0

    !!!!!!!!!!!!!!!!!!!!!!!!

    
    end module
    
        