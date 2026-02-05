!  ChiDoonCB7CBMD.f90 
!
!  FUNCTIONS:
!  ChiDoonCB7CBMD - Entry point of console application.
!

!****************************************************************************
!
!  PROGRAM: ChiDoonCB7CBMD
!
!  PURPOSE:  Entry point for the console application.
!
!****************************************************************************

    program ChiDoon5CB
    use commondata
    use EIGEN_3x3
    !use pafanalysis
    !use class_Autocorrelation
    
    implicit none
    !type(snaptype) :: sn
    !type(AutoCorrel) :: Corr

    
    
    character(280) TrajPath, TrajPa(20), fname, FF, SPfname
    character(4)   SnapId
    character(5)   ch1
    character(3)   ch2, tn(24)
    character(11)  fn(24)
    
    character(3) TempCh(20)
    character(90) resfile1, resfile2
    
    
    integer i, snap, groAtoms, molid, atid, atid2, tt, Ntemps
    real    pos(3), bb(3), e1(3), e2(3), ee(3), ss, nsnaps
    real evec(3,3), eval(3), dirs(1000,3), CdOP(14)   !, dirsall(9000,3)
    real evecz(3,3)  !, evalz(3)
    real evecy(3,3), evaly(3)
    real evecx(3,3), evalx(3)
    real EvecBiax(3,3), EvalBiax(3)
    real zpaf(3), ypaf(3), xpaf(3)  
    real :: opz=0.0D0, opzb=0.0D0, opy=0.0D0, opyb=0.0D0, opx=0.0D0, opxb=0.0D0
    real :: op1(3)=0.0d0
    integer j, ib, k, k1, k2, k3, i1, j1, ii, jj
    real :: nmrspec(0:200, 4) = 0.0d0
    real :: DeltaS 
    real :: vv(3), avgpar, avgtot, avgper, avmm(3,3), dd(3,3)
    
    real*8 :: mdotm_intra, mdotm_inter, mn_mn_intra, mn_mn_inter, mnSQ, gpar, gper
    real*8 :: qq, qq0, qqintra, qqinter, ww0, wwintra, wwinter, vv0, vvintra, vvinter
    real*8 rij(3), CNHistSamples, RadHistSamples, gRadHistSamples, avdir(3)

!!! Create CloseIm array, index runs over 26 neighbors + self (k=1)
    k=1
    CloseIm(k,1)=0
    CloseIm(k,2)=0
    CloseIm(k,3)=0
    !k=0
    do k1 = -1, 1
        do k2 = -1, 1
            do k3 = -1, 1
                if (k1==0 .AND. k2==0 .AND. k3==0) CYCLE
                k= k + 1
                CloseIm(k,1)=k1
                CloseIm(k,2)=k2
                CloseIm(k,3)=k3
            enddo
        enddo
    enddo
    IF (k .ne. 27) STOP(' CHECK CloseIm ')
!!! End Create CloseIm array

    DeltaS = 1.0D0/200
    Pi        = 4.00*ATAN(1.00)
    TwoPi     = 2.00*Pi
    FourPi    = 4.00*Pi
    Pi_over_2 = Pi/2.0d0
    Pi_over_4 = Pi/4.0d0
    
    !Dc = 2.0D0*0.02
    !NHIST = NINT(2.0D0/Dc)
    !DC = 1.0D0/NHIST
    !print*, NHIST
    !!stop
    !ALLOCATE( Hist(-NHIST:NHIST, 0:10) )
    !ALLOCATE( AVHist(-NHIST:NHIST, 0:10) )
    !HIST   = 0.0
    !AVHIST = 0.0
    !FF = 'GAFF'
    !if( ff=='AMOE' ) THEN
    !    Ntemps = 14
    !    fn(Ntemps:1:-1)=['k-418g', 'k-417g', 'k-416g', 'k-415g', 'k-414g', 'k-413g', 'k-412g', 'k-411g', 'k-409g', 'k-408g', 'k-406k', 'k-405j', 'k-407g', 'k-410g']    
    !    tn(Ntemps:1:-1)=[  '394',  '390',    '386',    '382',    '378',    '374',    '370',    '366',    '362',    '358',    '354',    '350',    '346',    '342'] 
    !else
    !    Ntemps = 16
    !    tn(1:Ntemps)=['402', '406', '410', '414', '418', '422', '426', '430', '434', '438', '442', '446', '450', '454', '458', '462']
    !    fn(1:Ntemps)=['212g','213g','214g','215g','216g','217h','218g','219g','220g','221g','222g','223g','224h','225g','226g','227g']
    !endif
    
NMOLS=1000
!NAtPerMol=38

i=0
i=i+1; TempCh(i)='390'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\037c_p-md-390-0.00\gro\'
i=i+1; TempCh(i)='388'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\035c_p-md-388-0.00\gro\'
i=i+1; TempCh(i)='386'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\025b_p-md-386-0.00\gro\'
i=i+1; TempCh(i)='384'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\033c_p-md-384-0.00\gro\'
i=i+1; TempCh(i)='382'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\032c_p-md-382-0.00\gro\'
i=i+1; TempCh(i)='380'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\026c_p-md-380-0.00\gro\'
i=i+1; TempCh(i)='378'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\025b_p-md-378-0.08\gro\'
i=i+1; TempCh(i)='376'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\024b_p-md-376-0.08\gro\'
i=i+1; TempCh(i)='374'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\023b_p-md-374-0.08\gro\'
i=i+1; TempCh(i)='372'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\022b_p-md-372-0.08\gro\'
i=i+1; TempCh(i)='370'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\021b_p-md-370-0.08\gro\'
i=i+1; TempCh(i)='368'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\020b_p-md-368-0.08\gro\'
i=i+1; TempCh(i)='366'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\019c_p-md-366-0.08\gro\'
i=i+1; TempCh(i)='364'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\018b_p-md-364-0.08\gro\'
i=i+1; TempCh(i)='362'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\015b_p-md-362-0.08\gro\'
i=i+1; TempCh(i)='360'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\014b_p-md-360-0.08\gro\'
i=i+1; TempCh(i)='358'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\016b_p-md-358-0.08\gro\'
i=i+1; TempCh(i)='356'; TrajPa(i) ='E:\Chi-Duen\5cb-x1000-gaff-88x88x94-zalign\017b_p-md-356-0.08\gro\'
Ntemps = I


do tt = 1, 1, 1  !loop over the available or the desired temperatures 
    

    !HERE WE GIVE THE PATH TO THE TRAJECTORY FILES
    TrajPath = TrajPa(tt)
    !TrajPath ='C:\Users\agvan\Downloads\ChiDoon_C4OPHODBP\c4ophodbp-x1000-gaff-70x70x290-zalign-007a-508-gro\gro\'
    !TempCh = '508'

    
    !RDFILE   = '5CB/'//TempCh(tt)//'_2.0.dat'
    !RDFILE2  = '5CB/'//TempCh(tt)//'_rpc.dat'
    
    nsnaps = 0.0

    HistSamples = 0.0D0
    MolHistSamples = 0.0D0
    CNHistSamples = 0.0d0
    RadHistSamples = 0.0D0
    gRadHistSamples = 0.0D0
    CdOP=0.0D0
    MSQ = 0.0D0
    !call sn%init(Nmols=1000, dim=3)
    !call Corr%init(TotSnaps=800, Every=1, CorrelToBeCalculated=8, sn=sn)
    opz=0.0D0; opzb=0.0D0; opy=0.0D0; opyb=0.0D0; opx=0.0D0; opxb=0.0D0
    op1=0.0d0
    gpar = 0.0
    avgpar = 0.0
    avgper = 0.0
    avgtot = 0.0
    avmm = 0.0d0
    
    do snap = 1, 1000  ! loop over the snapshots

        call ReadFrame_GRO( TrajPath, snap-1 )
        nsnaps = nsnaps + 1.0        
        ! AT THIS POINT THE Atpos(1000, NAtPerMol, 3) ARRAY IS POPULATED
        
        call MakeAlldirsOfSnap() ! this call populates CNDIR(1000,3) and CNCM(1000,3)
                                 ! here cndir and cncm are refered to the whole 5cb molecule
        
        call Make_PAF()          ! calculates the PAF based on MolZax
        ! at this point evecz(:,1) is the eigenvector of the PAF corresponding to the largest eigenvalue
        ! at this point evecz(:,2) is the eigenvector of the PAF corresponding to the middle eigenvalue
        ! at this point evecz(:,3) is the eigenvector of the PAF corresponding to the smalest eigenvalue
        ! evalz(1) is the largest eigenvalue of the PAF, evalz(2) the middle, evalz(3) the smallest
        
        GlobDir(:) = Evecz(:,1)

        
        
         !CALL GRadPairCor(gRadHistSamples, MolCM, MolXax, MolYax, MolZax)
         !CALL  RadPairCor(RadHistSamples, MolCM, MolXax, MolYax, MolZax)
         !CALL PairCor_Along_GlobDir(CNHistSamples, MolCM, MolXax, MolYax, MolZax, 2.0D0)
         !CALL PairCor_Along_Mol(   MolHistSamples, MolCM, MolXax, MolYax, MolZax, 2.0D0)

        !if (MOD(snap,100)==0 ) then
        !    if (MolHistSamples >0.5) call PRINT_PairCor_Along_Mol(MolHistSamples)
        !    if (CNHistSamples >0.5)  call PRINT_PairCor_Along_GlobDir(CNHistSamples)
        !    if (gRadHistSamples>0.5)  call PRINT_GRadPairCor(gRadHistSamples)
        !    if (RadHistSamples>0.5)  call PRINT_RadPairCor(RadHistSamples)
        !endif
        
        
    if (MOD(snap,50) == 0) then 
        print('(I5, 2x, ''-- '', A3,'' --'')'), snap, TempCh(tt)
        print*, ' eigevals (OPs) and eigevecs of PAF'
        print('(4F12.6)'), evalz(1), Evecz(:,1)
        print('(4F12.6)'), evalz(2), Evecz(:,2)
        print('(4F12.6)'), evalz(3), Evecz(:,3)
    endif
    
    enddo

    !if (Corr%CURRENT_N_OF_COR_SNAPS>0.5D0) then  
    !    open(unit=120, file=TRIM(TempCh(tt)//'_CorrFuncs.dat'))
    !        do i = 0, corr%Tot_COR_SNAPS-1
    !            write(120,'(20F14.7)') I*1.0, Corr%CorrHist(i,1:corr%CorrelToBeCalculated)/MAX(1.0,Corr%CorrHist(i,0))
    !        ENDDO
    !    close(120)
    !endif
    
    !open(unit=120, file=TRIM(TempCh(tt)//'.Ops'))
    !    write(120,'(A3, 3x, 20F14.7)') TempCh(tt), avdir/nsnaps, opz/nsnaps, opzb/nsnaps, opy/nsnaps, opyb/nsnaps, opx/nsnaps, opxb/nsnaps, op1/nsnaps
    !close(120)
    !
    !if (MolHistSamples >0.5) call PRINT_PairCor_Along_Mol(MolHistSamples)
    !if (CNHistSamples >0.5)  call PRINT_PairCor_Along_GlobDir(CNHistSamples)
    !if (RadHistSamples>0.5)  call PRINT_RadPairCor(RadHistSamples)
!stop    
    
    
enddo    
    contains
    
   function int2str( i, size ) result(str)
   ! Converts the integer "I" into a string with length "size"
   IMPLICIT NONE
   INTEGER, Intent(IN) :: I
   INTEGER, Intent(IN), OPTIONAL  :: size
   
   CHARACTER*12 :: tmp, str
   CHARACTER(LEN=99) :: FMT
   CHARACTER*1 :: ch
   
   INTEGER :: k, j 
   
!   if (present(size) ) THEN
!      WRITE( ch, '(I1)' ) size
!      FMT = '"(I'//ch//')"'
!      WRITE( tmp, FMT ) I
!   else
      WRITE( tmp, "(I12)" ) I
!   endif  
   
   if (present(size) ) THEN
     do k = 12, 12-size+1, -1
       if (tmp(k:k)=='' ) tmp(k:k)='0'
     enddo
   endif
   
   j=0
   do k = 1, 12
      str(k:k) = ''
      if (tmp(k:k)=='' ) cycle
      j = j + 1
      str(j:j) = tmp(k:k)
   enddo
   END FUNCTION
   
   
   subroutine ReadFrame_GRO( TrajPath, snap )
   implicit none
   integer, intent(in) :: snap
   integer stat, atid2, molid
   character(5)   ch1
   character(8)   ch2
   character(2)   ch3
   character(200) fname, TrajPath

    SnapId = int2str(snap)
    fname = trim(trim(TrajPath)//'frame'//trim(SnapId)//'.gro')
    open(unit=333,file=fname,status='old')
        read(333,*)
        read(333,*) groAtoms
        !print*, groAtoms
        do i=1, groatoms
           read(333,'(a5,a8,a2,I5, 3f8.3,3f8.4)') ch1(1:5), ch2, ch3, atid2, pos
           read(ch1,*,iostat=stat) molid 
           !print*, ch1(1:5), molid
           read(ch3,*,iostat=stat) atid 
            !read*
           Atpos(molid, atid2-(molid-1)*NAtPerMol, :) = pos
           !print*, molid, atid2-(molid-1)*NAtPerMol, atid2, pos
           !read*
        enddo

        read(333,*) BOX
        !print*, box
    close(333)
    !stop
   end subroutine
   
   
Subroutine MOL_AXES_FRAME_1(i)
! Build the molecular axes frame for molecule i
implicit none
integer i
real e1(3), e2(3), ee(3)
        molcm(I,:) = ( Atpos(i, 8, :)+Atpos(i, 10, :)+&
                       Atpos(i,12, :) )/3.0D0
         
        e1 = unitvec( Atpos(i,9, :)-Atpos(i,10, :) )

        dirs(i  ,:) = e1

        molzax(I,:) = e1   ! the long molecular axis
        ! at the moment we dont use in the calculations the y and x axes. 
        ! we polpulate them without any specific reason. 
        ! Update this in the future 
        molyax(I,:) = unitvec( e1 )
        ee = e1  !crossProd(molyax(I,:), molzax(I,:))
        molxax(I,:) = unitvec( ee )

        ! POSITION AN ORIENTATION OF THE DIPOLE
        dpos(I,:) = (Atpos(i, 3, :)+Atpos(i, 4, :) )/2.0d0 ! podition of the cn point dipole
        ddir(i,:) = unitvec( Atpos(i,4, :)-Atpos(i,3, :) ) ! direction of the cn dipole
end subroutine 


subroutine MakeAlldirsOfSnap()
implicit none
real :: e1(3), e2(3), ss, ee(3)
integer i
    do I=1, 1000
        CALL MOL_AXES_FRAME_1(i)
    enddo
end subroutine

Subroutine Make_PAF()
implicit none
integer I(1)
    CALL Grobal_PAF(MolZax, Evecz, Evalz )
    Zpaf = Evecz(:,1)
    Ypaf = Evecz(:,2)
    Xpaf = Evecz(:,3)
    return
end subroutine


!subroutine MakeCHbondsOfSnap()
!implicit none
!integer :: i, j
!real :: ss
!real :: ee(3)
!    do I=1, 500
!        ! CH bonds close to CN 
!        CHbond(I, 01, :) = Atpos(i, 58, :)-Atpos(i, 21, :)
!        CHbond(I, 02, :) = Atpos(i, 54, :)-Atpos(i, 13, :)
!        CHbond(I, 03, :) = Atpos(i, 61, :)-Atpos(i, 24, :)
!        CHbond(I, 04, :) = Atpos(i, 57, :)-Atpos(i, 16, :)
!
!        CHbond(I, 05, :) = Atpos(i, 62, :)-Atpos(i, 25, :)
!        CHbond(I, 06, :) = Atpos(i, 50, :)-Atpos(i, 09, :)
!        CHbond(I, 07, :) = Atpos(i, 65, :)-Atpos(i, 28, :)
!        CHbond(I, 08, :) = Atpos(i, 53, :)-Atpos(i, 12, :)
!
!        CHbond(I, 09, :) = Atpos(i, 51, :)-Atpos(i, 10, :)
!        CHbond(I, 10, :) = Atpos(i, 63, :)-Atpos(i, 26, :)
!        CHbond(I, 11, :) = Atpos(i, 56, :)-Atpos(i, 15, :)
!        CHbond(I, 12, :) = Atpos(i, 60, :)-Atpos(i, 23, :)
!
!        CHbond(I, 13, :) = Atpos(i, 55, :)-Atpos(i, 14, :)
!        CHbond(I, 14, :) = Atpos(i, 59, :)-Atpos(i, 22, :)
!        CHbond(I, 15, :) = Atpos(i, 64, :)-Atpos(i, 27, :)
!        CHbond(I, 16, :) = Atpos(i, 52, :)-Atpos(i, 11, :)
!
!        CHbond(I, 17, :) = Atpos(i, 38, :)-Atpos(i, 29, :)
!        CHbond(I, 18, :) = Atpos(i, 36, :)-Atpos(i, 29, :)
!        CHbond(I, 19, :) = Atpos(i, 48, :)-Atpos(i, 35, :)
!        CHbond(I, 20, :) = Atpos(i, 49, :)-Atpos(i, 35, :)
!
!        CHbond(I, 21, :) = Atpos(i, 37, :)-Atpos(i, 30, :)
!        CHbond(I, 22, :) = Atpos(i, 39, :)-Atpos(i, 30, :)
!        CHbond(I, 23, :) = Atpos(i, 46, :)-Atpos(i, 34, :)
!        CHbond(I, 24, :) = Atpos(i, 47, :)-Atpos(i, 34, :)
!
!        CHbond(I, 25, :) = Atpos(i, 42, :)-Atpos(i, 31, :)
!        CHbond(I, 26, :) = Atpos(i, 40, :)-Atpos(i, 31, :)
!        CHbond(I, 27, :) = Atpos(i, 43, :)-Atpos(i, 33, :)
!        CHbond(I, 28, :) = Atpos(i, 45, :)-Atpos(i, 33, :)
!
!        CHbond(I, 29, :) = Atpos(i, 41, :)-Atpos(i, 32, :)
!        CHbond(I, 30, :) = Atpos(i, 44, :)-Atpos(i, 32, :)
!        do j=1, 30
!            ss = SQRT(SUM(CHbond(I, j, :)**2))
!            CHbondLen(I, j) = ss
!            CHbond(I, j, :) = CHbond(I, j, :)/ss
!        enddo
!        
!    enddo
!end subroutine


SUBROUTINE Grobal_PAF( DIRs,  Evec, Eval, ns, om0 )
        IMPLICIT NONE
        INTEGER, OPTIONAL :: NS
        REAL, INTENT(IN) :: Dirs(:,:)
        !REAL, Optional, INTENT(IN) :: Dirs2(:,:)
        REAL, INTENT(OUT)  :: Eval(:), Evec(:,:)
        INTEGER :: nn, i
        REAL, OPTIONAL :: om0(:,:)
        REAL :: om(3,3)
        
            IF ( PRESENT(ns)) THEN
                nn=ns
            ELSE
                nn = size(dirs, DIM=1)
            ENDIF
            if( nn==0 ) Then
                Evec = 0.0D0
                Eval = 0.0D0
                return
            endif
            om = 0.0D0
            
            !print*, nn
            DO i = 1, nn
                om(1,1) = om(1,1) + dirs(i,1)**2 
                om(2,2) = om(2,2) + dirs(i,2)**2
                om(3,3) = om(3,3) + dirs(i,3)**2
                om(1,2) = om(1,2) + dirs(i,1)*dirs(i,2) 
                om(1,3) = om(1,3) + dirs(i,1)*dirs(i,3) 
                om(2,3) = om(2,3) + dirs(i,2)*dirs(i,3) 
            ENDDO
            !om(3,3) = -om(1,1)-om(2,2)
            om(2,1) = om(1,2)
            om(3,1) = om(1,3)
            om(3,2) = om(2,3)
            
            !om = 1.5*om
            
            om(1,1) = om(1,1) - nn/3.0D0 
            om(2,2) = om(2,2) - nn/3.0D0 
            om(3,3) = om(3,3) - nn/3.0D0 
            !print*, 1.5*(om(1,1)+om(2,2)+om(3,3))
            if (PRESENT(om0)) om0 = om
            call DSYEVH3(om, evec, eval)
            eval = 1.5D0*eval/nn
RETURN            
            DO i = 1, nn
                om(1,1) = om(1,1) + 1.50D0*dirs(i,1)**2-0.50D0 
                om(2,2) = om(2,2) + 1.50D0*dirs(i,2)**2-0.50D0 
                om(3,3) = om(3,3) + 1.50D0*dirs(i,3)**2-0.50D0 
                om(1,2) = om(1,2) + 1.50D0*dirs(i,1)*dirs(i,2) 
                om(1,3) = om(1,3) + 1.50D0*dirs(i,1)*dirs(i,3) 
                om(2,3) = om(2,3) + 1.50D0*dirs(i,2)*dirs(i,3) 
            ENDDO
            !om(3,3) = -om(1,1)-om(2,2)
            om(2,1) = om(1,2)
            om(3,1) = om(1,3)
            om(3,2) = om(2,3)
            om = om/nn
            call DSYEVH3(om, evec, eval)

            RETURN
            
        END SUBROUTINE    
        
SUBROUTINE Grobal_PAF_Biax( DIRs1, DIRs2, Evec, Eval, ns, om0 )
        IMPLICIT NONE
        INTEGER, OPTIONAL :: NS
        REAL, INTENT(IN) :: Dirs1(:,:),Dirs2(:,:)
        !REAL, Optional, INTENT(IN) :: Dirs2(:,:)
        REAL, INTENT(OUT)  :: Eval(:), Evec(:,:)
        INTEGER :: nn, i
        REAL, OPTIONAL :: om0(:,:)
        REAL :: om(3,3)
        
            IF ( PRESENT(ns)) THEN
                nn=ns
            ELSE
                nn = size(dirs1, DIM=1)
            ENDIF
            if( nn==0 ) Then
                Evec = 0.0D0
                Eval = 0.0D0
                return
            endif
            om = 0.0D0
            
            !print*, nn
            DO i = 1, nn
                om(1,1) = om(1,1) + dirs1(i,1)**2-dirs2(i,1)**2 
                om(2,2) = om(2,2) + dirs1(i,2)**2-dirs2(i,2)**2
                om(3,3) = om(3,3) + dirs1(i,3)**2-dirs2(i,3)**2
                om(1,2) = om(1,2) + dirs1(i,1)*dirs1(i,2)-dirs2(i,1)*dirs2(i,2) 
                om(1,3) = om(1,3) + dirs1(i,1)*dirs1(i,3)-dirs2(i,1)*dirs2(i,3) 
                om(2,3) = om(2,3) + dirs1(i,2)*dirs1(i,3)-dirs2(i,2)*dirs2(i,3) 
            ENDDO
            om(2,1) = om(1,2)
            om(3,1) = om(1,3)
            om(3,2) = om(2,3)
            
            if (PRESENT(om0)) om0 = om
            call DSYEVH3(om, evec, eval)
            eval = 1.5D0*eval/nn
RETURN            
            
END SUBROUTINE    

Function CrossProd( v, s ) Result (c)
    implicit none
    REAL :: v(:),s(:), c(3)
       c(1) = s(3)*v(2) - s(2)*v(3)
       c(2) =-s(3)*v(1) + s(1)*v(3)
       c(3) = s(2)*v(1) - s(1)*v(2)
    end function CrossProd
    
    Function UnitVec( v ) Result (c)
    implicit none
    REAL :: v(:), c(3), ss 
       ss=SQRT(SUM(v*v))
       c = v/ss
    end function UnitVec

    end program ChiDoon5CB

