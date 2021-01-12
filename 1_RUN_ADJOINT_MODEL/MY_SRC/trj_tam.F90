MODULE trj_tam
#ifdef key_tam
   !!======================================================================
   !!                       ***  MODULE trj_tam ***
   !! NEMOVAR trajectory: Allocate and read the trajectory for linearzation
   !!======================================================================

   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !!   bkg_init : Initialize the background fields from disk
   !!----------------------------------------------------------------------
   !! * Modules used
   USE par_oce
   USE tamtrj             ! Parameters for the assmilation interface
   USE in_out_manager
   USE oce                ! Model variables
   USE zdf_oce            ! Vertical mixing variables
   USE zdfddm             ! Double diffusion mixing parameterization
   USE zdfbfr
   USE trc_oce
   USE ldftra_oce         ! Lateral tracer mixing coefficient defined in memory
   USE ldfslp             ! Slopes of neutral surfaces
   USE tradmp             ! Tracer damping
   USE sbc_oce            ! Ocean surface boundary conditions
   USE iom                ! Library to read input files
   USE zdfmxl
   USE divcur             ! horizontal divergence and relative vorticity
   USE sshwzv
   USE oce_tam
   !!! 20191004B - by SAM - combining before and now variables into one output
   USE eosbn2
   USE lbclnk_tam
   USE wrk_nemo 
   !!! /20191004B
   IMPLICIT NONE

   !! * Routine accessibility
   PRIVATE
   PUBLIC &
      & trj_rea,     &   !: Read trajectory at time step kstep into now fields
      & trj_rd_spl,  &   !: Read simple data (without interpolation)
      & trj_wri_spl, &   !: Write simple data (without interpolation)
      & tl_trj_wri,  &   !: Write simple linear-tangent data
      & tl_trj_ini,  &   !: initialize the model-tangent state trajectory
      & trj_deallocate   !: Deallocate all the saved variable

   LOGICAL, PUBLIC :: &
      & ln_trjwri_tan = .FALSE.   !: No output of the state trajectory fields

   CHARACTER (LEN=40), PUBLIC :: &
      & cn_tantrj                                  !: Filename for storing the
                                                   !: linear-tangent trajectory
   INTEGER, PUBLIC :: &
      & nn_ittrjfrq_tan         !: Frequency of trajectory output for linear-tangent

   !! * Module variables
   LOGICAL, SAVE :: &
      & ln_mem = .FALSE.      !: Flag for allocation
   INTEGER, SAVE :: inumtrj1 = -1, inumtrj2 = -1
   REAL(wp), SAVE :: &
      & stpr1, &
      & stpr2
   REAL(wp), ALLOCATABLE, DIMENSION(:,:), SAVE :: &
      & empr1,    &
      & empsr1,   &
      & empr2,    &
      & empsr2,   &
      & bfruar1,  &
      & bfrvar1,  &
      & bfruar2,  &
      & bfrvar2
#if defined key_traldf_eiv
#if defined key_traldf_c3d
   REAL(wp), ALLOCATABLE, DIMENSION(:,:,:), SAVE :: &
#elif defined key_traldf_c2d
   REAL(wp), ALLOCATABLE, DIMENSION(:,:), SAVE :: &
#elif defined key_traldf_c1d
   REAL(wp), ALLOCATABLE, DIMENSION(:), SAVE :: &
#else
   REAL(wp) ::
#endif
      & aeiur1,   &
      & aeivr1,   &
      & aeiwr1,   &
      & aeiur2,   &
      & aeivr2,   &
      & aeiwr2
#endif
  REAL(wp), ALLOCATABLE, DIMENSION(:,:,:), SAVE :: &
      & unr1,     &
      & vnr1,     &
      & tnr1,     &
      & snr1,     &
      & avmur1,   &
      & avmvr1,   &
      & avtr1,    &
      & uslpr1,   &
      & vslpr1,   &
      & wslpir1,  &
      & wslpjr1,  &
      & avsr1,    &
      & etot3r1,  &
      & unr2,     &
      & vnr2,     &
      & tnr2,     &
      & snr2,     &
      & avmur2,   &
      & avmvr2,   &
      & avtr2,    &
      & uslpr2,   &
      & vslpr2,   &
      & wslpir2,  &
      & wslpjr2,  &
      & avsr2,    &
      & etot3r2
  REAL(wp), ALLOCATABLE, DIMENSION(:,:), SAVE :: &
      & hmlp1,    &
      & hmlp2,    &
!!! 20191004C add 'sshn' to nonlinear trajectory
      & sshnr1,   & 
      & sshnr2
!!! /20191004C

CONTAINS

   SUBROUTINE tl_trj_ini
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE tl_trj_ini ***
      !!
      !! ** Purpose : initialize the model-tangent state trajectory
      !!
      !! ** Method  :
      !!
      !! ** Action  :
      !!
      !! References :
      !!
      !! History :
      !!        ! 10-07 (F. Vigilant)
      !!-----------------------------------------------------------------------

      IMPLICIT NONE

      !! * Modules used
      NAMELIST/namtl_trj/ nn_ittrjfrq_tan, ln_trjwri_tan, cn_tantrj

      ln_trjwri_tan = .FALSE.
      nn_ittrjfrq_tan = 1
      cn_tantrj = 'tl_trajectory'
      REWIND ( numnam )
      READ   ( numnam, namtl_trj )

      ! Control print
      IF(lwp) THEN
         WRITE(numout,*)
         WRITE(numout,*) 'tl_trj_ini : Linear-Tagent Trajectory handling:'
         WRITE(numout,*) '~~~~~~~~~~~~'
         WRITE(numout,*) '          Namelist namtl_trj : set trajectory parameters'
         WRITE(numout,*) '             Logical switch for writing out state trajectory         ', &
            &            ' ln_trjwri_tan = ', ln_trjwri_tan
         WRITE(numout,*) '             Frequency of trajectory output                          ', &
            &            ' nn_ittrjfrq_tan = ', nn_ittrjfrq_tan
      END IF
   END SUBROUTINE tl_trj_ini

   SUBROUTINE trj_rea( kstp, kdir )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE trj_reat  ***
      !!
      !! ** Purpose : Read from file the trjectory from the outer loop
      !!
      !! ** Method  : IOM
      !!
      !! ** Action  :
      !!
      !! References :
      !!
      !! History :
      !!        ! 08-05 (K. Mogensen) Initial version
      !!        ! 09-03 (F.Vigilant) Add reading of hmlp and calls (divcur, wzvmod)
      !!        ! 2010-04 (F. Vigilant) converison to 3.2
      !!        ! 2012-07 (P.-A. Bouttier) converison to 3.4
      !!-----------------------------------------------------------------------
      !! * Modules used
      !! * Arguments
      INTEGER, INTENT(in) :: &
         & kstp, &           ! Step for requested trajectory
         & kdir              ! Direction for stepping (1 forward, -1 backward)
      !! * Local declarations
      CHARACTER (LEN=100) :: &
         & cl_dirtrj
      INTEGER :: &
         & inrcm,  &
         & inrcp,  &
         & inrc,   &
         & istpr1, &
         & istpr2, &
	 & it
      REAL(KIND=wp) :: &
         & zwtr1, &
         & zwtr2, &
         & zden,  &
         & zstp
      ! Initialize data and open file
      !! if step time is corresponding to a saved state


      IF ( ( MOD( kstp - nit000 + 1, nn_ittrjfrq ) == 0 )  ) THEN

!!! 20191004R - trajectory offset option
         !it = kstp - nit000 + 1
         it = ((kstp - nit000 + 1 + nn_ittrjoffset) - MOD(kstp - nit000 + 1, nn_ittrjfrq))
!!!/20191004R

         IF ( inumtrj1 == -1 ) THEN
            
            ! Define the input file
            !!! 20191004D
            !WRITE(cl_dirtrj, FMT='(I5.5,A,A,".nc")' ) it, '_', TRIM( cn_dirtrj )
            WRITE(cl_dirtrj, FMT='(A,A,I0.8,".nc")' ) TRIM( cn_dirtrj ), '_', it
            !         WRITE(cl_dirtrj, FMT='(A,".nc")' ) TRIM( c_dirtrj )
            !!! /20191004D

            cl_dirtrj = TRIM( cl_dirtrj )

            IF(lwp) THEN

               WRITE(numout,*)
               WRITE(numout,*)'Reading non-linear fields from : ',TRIM(cl_dirtrj)
               WRITE(numout,*)

            ENDIF
            CALL iom_open( cl_dirtrj, inumtrj1 )
            if ( inumtrj1 == -1)  CALL ctl_stop( 'No tam_trajectory cl_amstrj found' )
            IF ( .NOT. ln_mem ) THEN
               ALLOCATE( &
                  & empr1(jpi,jpj),  &
                  & empsr1(jpi,jpj), &
                  & empr2(jpi,jpj),  &
                  & empsr2(jpi,jpj), &
                  & bfruar1(jpi,jpj),&
                  & bfrvar1(jpi,jpj),&
                  & bfruar2(jpi,jpj),&
                  & bfrvar2(jpi,jpj) &
                  & )

               ALLOCATE( &
                  & unr1(jpi,jpj,jpk),     &
                  & vnr1(jpi,jpj,jpk),     &
                  & tnr1(jpi,jpj,jpk),     &
                  & snr1(jpi,jpj,jpk),     &
                  & avmur1(jpi,jpj,jpk),   &
                  & avmvr1(jpi,jpj,jpk),   &
                  & avtr1(jpi,jpj,jpk),    &
                  & etot3r1(jpi,jpj,jpk),  &
                  & unr2(jpi,jpj,jpk),     &
                  & vnr2(jpi,jpj,jpk),     &
                  & tnr2(jpi,jpj,jpk),     &
                  & snr2(jpi,jpj,jpk),     &
                  & avmur2(jpi,jpj,jpk),   &
                  & avmvr2(jpi,jpj,jpk),   &
                  & avtr2(jpi,jpj,jpk),    &
                  & etot3r2(jpi,jpj,jpk),  &
!!! 20191004C Addd "sshn" to trajectory
                  & sshnr1(jpi,jpj),       &
                  & sshnr2(jpi,jpj)        &
!!! /20191004C
                  & )
#if defined key_traldf_eiv
#if defined key_traldf_c3d
#elif defined key_traldf_c2d
               ALLOCATE( &
                  & aeiur1(jpi,jpj), &
                  & aeivr1(jpi,jpj), &
                  & aeiwr1(jpi,jpj), &
                  & aeiur2(jpi,jpj), &
                  & aeivr2(jpi,jpj), &
                  & aeiwr2(jpi,jpj)  &
                  & )
#elif defined key_traldf_c1d
#endif
#endif

#if defined key_ldfslp
               ALLOCATE( &
                  & uslpr1(jpi,jpj,jpk),   &
                  & vslpr1(jpi,jpj,jpk),   &
                  & wslpir1(jpi,jpj,jpk),  &
                  & wslpjr1(jpi,jpj,jpk),  &
                  & uslpr2(jpi,jpj,jpk),   &
                  & vslpr2(jpi,jpj,jpk),   &
                  & wslpir2(jpi,jpj,jpk),  &
                  & wslpjr2(jpi,jpj,jpk)   &
                  & )
#endif

#if defined key_zdfddm
               ALLOCATE( &
                  & avsr1(jpi,jpj,jpk),    &
                  & avsr2(jpi,jpj,jpk)     &
                  & )
#endif

#if defined key_tradmp
               ALLOCATE( &
                  & hmlp1(jpi,jpj),    &
                  & hmlp2(jpi,jpj)     &
                  & )
#endif
               ln_mem = .TRUE.
            ENDIF
         ENDIF


      ! Read records

         inrcm = INT( ( kstp - nit000 + 1 ) / nn_ittrjfrq ) + 1

         ! Copy record 1 into record 2

         IF ( ( kstp /= nitend )         .AND. &
            & ( kstp - nit000 + 1 /= 0 ) .AND. &
            & ( kdir == -1 ) ) THEN

            stpr2           = stpr1

            empr2   (:,:)   = empr1   (:,:)
            empsr2  (:,:)   = empsr1  (:,:)
            bfruar2  (:,:)  = bfruar1 (:,:)
            bfrvar2  (:,:)  = bfrvar1 (:,:)

            unr2    (:,:,:) = unr1    (:,:,:)
            vnr2    (:,:,:) = vnr1    (:,:,:)
            tnr2    (:,:,:) = tnr1    (:,:,:)
            snr2    (:,:,:) = snr1    (:,:,:)
            avmur2  (:,:,:) = avmur1  (:,:,:)
            avmvr2  (:,:,:) = avmvr1  (:,:,:)
            avtr2   (:,:,:) = avtr1   (:,:,:)
!!! 20191004C Adding sshn to nonlinear trajectory
            sshnr2  (:,:)   = sshnr1    (:,:)
!!! /20191004C
#if defined key_ldfslp
            uslpr2  (:,:,:) = uslpr1  (:,:,:)
            vslpr2  (:,:,:) = vslpr1  (:,:,:)
            wslpir2 (:,:,:) = wslpir1 (:,:,:)
            wslpjr2 (:,:,:) = wslpjr1 (:,:,:)
#endif
#if defined key_zdfddm
            avsr2   (:,:,:) = avsr1   (:,:,:)
#endif
            etot3r2 (:,:,:) = etot3r1 (:,:,:)
#if defined key_tradmp
            hmlp1   (:,:)   = hmlp2   (:,:)
#endif
#if defined key_traldf_eiv
#if defined key_traldf_c3d
            aeiur2  (:,:,:) = aeiur1  (:,:,:)
            aeivr2  (:,:,:) = aeivr1  (:,:.:)
            aeiwr2  (:,:,:) = aeiwr1  (:,:.:)
#elif defined key_traldf_c2d
            aeiur2  (:,:)   = aeiur1  (:,:)
            aeivr2  (:,:)   = aeivr1  (:,:)
            aeiwr2  (:,:)   = aeiwr1  (:,:)
#elif defined key_traldf_c1d
            aeiur2  (:)     = aeiur1  (:)
            aeivr2  (:)     = aeivr1  (:)
            aeiwr2  (:)     = aeiwr1  (:)
#else
            aeiur2          = aeiur1
            aeivr2          = aeivr1
            aeiwr2          = aeiwr1
#endif
#endif

            istpr1 = INT( stpr1 )

            IF(lwp) WRITE(numout,*) &
               &                 '    Trajectory record copy time step = ', istpr1

         ENDIF

         IF ( ( kstp - nit000 + 1 /= 0 ) .AND. ( kdir == -1 ) ) THEN
            ! We update the input filename

!!!20191004D expanding I/O to allow up to 100e6 time steps
            !WRITE(cl_dirtrj, FMT='(I5.5,A,A,".nc")' ) (it-nn_ittrjfrq), '_', TRIM(cn_dirtrj )
            WRITE(cl_dirtrj, FMT='(A,A,I0.8,".nc")' ) TRIM(cn_dirtrj ), '_', (it-nn_ittrjfrq)
! /20191004D            

            cl_dirtrj = TRIM( cl_dirtrj )
            IF(lwp) THEN
               WRITE(numout,*)
               WRITE(numout,*)'Reading non-linear fields from : ',TRIM(cl_dirtrj)
               WRITE(numout,*)
            ENDIF
         ENDIF

         ! Read record 1

         IF ( ( kstp - nit000 + 1 == 0 ) .AND.( kdir == 1           ) .OR. &
            & ( kstp - nit000 + 1 /= 0 ) .AND.( kdir == -1          ) ) THEN

            IF ( kdir == -1 ) inrcm = inrcm - 1
!            inrc = inrcm
            ! temporary fix: currently, only one field by step time
            inrc = 1
            stpr1 = (inrcm - 1) * nn_ittrjfrq

            ! bug fixed to read several time the initial data
            IF ( ( kstp - nit000 + 1 == 0 ) .AND. ( kdir == 1 ) ) THEN
               ! Define the input file
!!! 20191004D expanding I/O to allow up to 100e6 time steps
               !WRITE(cl_dirtrj, FMT='(I5.5, A,A,".nc")' ) it, '_', TRIM( cn_dirtrj )
            WRITE(cl_dirtrj, FMT='(A,A,I0.8, ".nc")' ) TRIM( cn_dirtrj ), '_', it
!!! /20191004D

               cl_dirtrj = TRIM( cl_dirtrj )

               IF(lwp) THEN
                  WRITE(numout,*)
                  WRITE(numout,*)'Reading non-linear fields from : ',TRIM(cl_dirtrj)
                  WRITE(numout,*)
               ENDIF
            END IF
            IF ( inumtrj1 /= -1 )   CALL iom_open( cl_dirtrj, inumtrj1 )

            CALL iom_get( inumtrj1, jpdom_autoglo, 'emp'   , empr1   , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'emps'  , empsr1  , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'un'    , unr1    , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'vn'    , vnr1    , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'tn'    , tnr1    , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'sn'    , snr1    , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'avmu'  , avmur1  , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'avmv'  , avmvr1  , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'avt'   , avtr1   , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'bfrua' , bfruar1 , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'bfrva' , bfrvar1 , inrc )
!!! 20191004C - adding sshn to nonlinear trajectory
            CALL iom_get( inumtrj1, jpdom_autoglo, 'sshn'  , sshnr1  , inrc )
!!! /20191004C
#if defined key_ldfslp
            CALL iom_get( inumtrj1, jpdom_autoglo, 'uslp'  , uslpr1  , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'vslp'  , vslpr1  , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'wslpi' , wslpir1 , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'wslpj' , wslpjr1 , inrc )
#endif
#if defined key_zdfddm
            CALL iom_get( inumtrj1, jpdom_autoglo, 'avs'   , avsr1   , inrc )
#endif
            CALL iom_get( inumtrj1, jpdom_autoglo, 'etot3' , etot3r1 , inrc )
#if defined key_tradmp
            CALL iom_get( inumtrj1, jpdom_autoglo, 'hmlp'  , hmlp1   , inrc )
#endif
#if defined key_traldf_eiv
            CALL iom_get( inumtrj1, jpdom_autoglo, 'aeiu'  , aeiur1  , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'aeiv'  , aeivr1  , inrc )
            CALL iom_get( inumtrj1, jpdom_autoglo, 'aeiw'  , aeiwr1  , inrc )
#endif
!!!20191129A adding ice fraction to tam output
            CALL iom_get( inumtrj1, jpdom_autoglo, 'ice_fraction', fr_i , inrc)
!!!/20191129A           

            CALL iom_close( inumtrj1 )

            istpr1 = INT( stpr1 )
            IF(lwp)WRITE(numout,*) '   trajectory read time step = ', istpr1,&
               &                   '  record = ', inrc

         ENDIF


         ! Copy record 2 into record 1

         IF ( ( kstp - nit000 + 1 /= 0 ) .AND. &
            & ( kstp /= nitend         ) .AND. &
            & ( kdir == 1              ) ) THEN

            stpr1           = stpr2
            empr1   (:,:)   = empr2   (:,:)
            empsr1  (:,:)   = empsr2  (:,:)
            bfruar1 (:,:)   = bfruar2 (:,:)
            bfrvar1 (:,:)   = bfrvar2 (:,:)
            unr1    (:,:,:) = unr2    (:,:,:)
            vnr1    (:,:,:) = vnr2    (:,:,:)
            tnr1    (:,:,:) = tnr2    (:,:,:)
            snr1    (:,:,:) = snr2    (:,:,:)
            avmur1  (:,:,:) = avmur2  (:,:,:)
            avmvr1  (:,:,:) = avmvr2  (:,:,:)
            avtr1   (:,:,:) = avtr2   (:,:,:)
!!! 20191004C - added sshn to trajectory
            sshnr1  (:,:)   = sshnr2  (:,:)
!!!/20191004C
#if defined key_ldfslp
            uslpr1  (:,:,:) = uslpr2  (:,:,:)
            vslpr1  (:,:,:) = vslpr2  (:,:,:)
            wslpir1 (:,:,:) = wslpir2 (:,:,:)
            wslpjr1 (:,:,:) = wslpjr2 (:,:,:)
#endif
#if defined key_zdfddm
            avsr1   (:,:,:) = avsr2   (:,:,:)
#endif
            etot3r1 (:,:,:) = etot3r2 (:,:,:)
#if defined key_tradmp
            hmlp1   (:,:)   = hmlp2   (:,:)
#endif
#if defined key_traldf_eiv
#if defined key_traldf_c3d
            aeiur1  (:,:,:) = aeiur2  (:,:,:)
            aeivr1  (:,:,:) = aeivr2  (:,:.:)
            aeiwr1  (:,:,:) = aeiwr2  (:,:.:)
#elif defined key_traldf_c2d
            aeiur1  (:,:)   = aeiur2  (:,:)
            aeivr1  (:,:)   = aeivr2  (:,:)
            aeiwr1  (:,:)   = aeiwr2  (:,:)
#elif defined key_traldf_c1d
            aeiur1  (:)     = aeiur2  (:)
            aeivr1  (:)     = aeivr2  (:)
            aeiwr1  (:)     = aeiwr2  (:)
#else
            aeiur1          = aeiur2
            aeivr1          = aeivr2
            aeiwr1          = aeiwr2
#endif
#endif

            istpr1 = INT( stpr1 )
            IF(lwp) WRITE(numout,*) &
               &                 '   Trajectory record copy time step = ', istpr1

         ENDIF

         ! Read record 2

         IF ( ( ( kstp /= nitend ) .AND. ( kdir == 1  )) .OR. &
            &   ( kstp == nitend ) .AND.(  kdir == -1   ) ) THEN

               ! Define the input file
               IF  (  kdir == -1   ) THEN
!!!20191004D - expanding I/O filenames to allow up to 100e6 time steps
                     WRITE(cl_dirtrj, FMT='(A,A,I0.8,".nc")' ) TRIM( cn_dirtrj ), '_', it
                  !WRITE(cl_dirtrj, FMT='(I5.5,A,A,".nc")' ) it, '_', TRIM( cn_dirtrj )
!!!/20191004D
               ELSE
!!!20191004D - expanding I/O filenames to allow up to 100e6 time steps
                  !WRITE(cl_dirtrj, FMT='(I5.5,A,A,".nc")' ) (it+nn_ittrjfrq), '_', TRIM( cn_dirtrj )
                     WRITE(cl_dirtrj, FMT='(A,A,I0.8,".nc")' ) TRIM( cn_dirtrj ), '_', (it+nn_ittrjfrq)
!!!/20191004D
               ENDIF
               cl_dirtrj = TRIM( cl_dirtrj )

               IF(lwp) THEN
                  WRITE(numout,*)
                  WRITE(numout,*)'Reading non-linear fields from : ',TRIM(cl_dirtrj)
                  WRITE(numout,*)
               ENDIF

               CALL iom_open( cl_dirtrj, inumtrj2 )


            inrcp = inrcm + 1
            !            inrc  = inrcp
            inrc = 1  ! temporary  fix

            stpr2 = (inrcp - 1) * nn_ittrjfrq
            CALL iom_get( inumtrj2, jpdom_autoglo, 'emp'   , empr2   , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'emps'  , empsr2  , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'un'    , unr2    , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'vn'    , vnr2    , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'tn'    , tnr2    , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'sn'    , snr2    , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'avmu'  , avmur2  , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'avmv'  , avmvr2  , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'avt'   , avtr2   , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'bfrua' , bfruar2 , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'bfrva' , bfrvar2 , inrc )
!!! 20191004C add sshn to nonlinear trajectory
            CALL iom_get( inumtrj2, jpdom_autoglo, 'sshn'  , sshnr2  , inrc )
!!! /20191004C
#if defined key_ldfslp
            CALL iom_get( inumtrj2, jpdom_autoglo, 'uslp'  , uslpr2  , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'vslp'  , vslpr2  , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'wslpi' , wslpir2 , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'wslpj' , wslpjr2 , inrc )
#endif
#if defined key_zdfddm
            CALL iom_get( inumtrj2, jpdom_autoglo, 'avs'   , avsr2   , inrc )
#endif
            CALL iom_get( inumtrj2, jpdom_autoglo, 'etot3' , etot3r2 , inrc )
#if defined key_tradmp
            CALL iom_get( inumtrj2, jpdom_autoglo, 'hmlp'  , hmlp2   , inrc )
#endif
#if defined key_traldf_eiv
            CALL iom_get( inumtrj2, jpdom_autoglo, 'aeiu'  , aeiur2  , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'aeiv'  , aeivr2  , inrc )
            CALL iom_get( inumtrj2, jpdom_autoglo, 'aeiw'  , aeiwr2  , inrc )
#endif
            CALL iom_close( inumtrj2 )

            istpr2 = INT( stpr2 )
            IF(lwp)WRITE(numout,*) '   trajectory read2 time step = ', istpr2,&
               &                   '  record = ', inrc
         ENDIF

      ENDIF

      ! Add warning for user
      IF ( (kstp == nitend) .AND. ( MOD( kstp - nit000 + 1, nn_ittrjfrq ) /= 0 )  ) THEN
          IF(lwp) WRITE(numout,*) '   Warning ! nitend (=',nitend, ')', &
               &                  ' and saving frequency (=',nn_ittrjfrq,') not compatible.'
      ENDIF

      ! Linear interpolate to the current step

      IF(lwp)WRITE(numout,*) '   linear interpolate to current', &
         &                   ' time step = ', kstp

      ! Interpolation coefficients

      zstp = kstp - nit000 + 1
      zden   = 1.0 / ( stpr2 - stpr1 )

      zwtr1  = ( stpr2 - zstp      ) * zden
      zwtr2  = ( zstp  - stpr1     ) * zden

      IF(lwp)WRITE(numout,*) '   linear interpolate coeff.', &
         &                   '  = ', zwtr1, zwtr2

!!! 20191004E - SAM: correct transition of b->n for 'kdir==-1'
      !IF ( kstp /= nit000-1 ) THEN
      IF ( ( kstp /= nit000-1 ).AND.( kdir == 1 ) ) THEN
!!! /20191004E
         tsb(:,:,:,:) = tsn(:,:,:,:)
         ub(:,:,:) = un(:,:,:)
         vb(:,:,:) = vn(:,:,:)
      END IF
      emp(:,:)      = zwtr1 * empr1   (:,:)   + zwtr2 * empr2   (:,:)
      emps(:,:)     = zwtr1 * empsr1  (:,:)   + zwtr2 * empsr2  (:,:)
      bfrua(:,:)    = zwtr1 * bfruar1 (:,:)   + zwtr2 * bfruar2 (:,:)
      bfrva(:,:)    = zwtr1 * bfrvar1 (:,:)   + zwtr2 * bfrvar2 (:,:)
      un(:,:,:)     = zwtr1 * unr1    (:,:,:) + zwtr2 * unr2    (:,:,:)
      vn(:,:,:)     = zwtr1 * vnr1    (:,:,:) + zwtr2 * vnr2    (:,:,:)
      tsn(:,:,:,jp_tem)     = zwtr1 * tnr1    (:,:,:) + zwtr2 * tnr2    (:,:,:)
      tsn(:,:,:,jp_sal)     = zwtr1 * snr1    (:,:,:) + zwtr2 * snr2    (:,:,:)
!!! 20191004C Added 'sshn' to trajectory
      sshn(:,:)     = zwtr1 * sshnr1    (:,:) + zwtr2 * sshnr2    (:,:)
!!! /20191004C

!!! 20191004E - SAM: correct transition of b->n for 'kdir==-1'; Note, zstp should always be at leas stpr1+1
      IF ( kdir == -1 ) THEN
!!! /20191004E
!!! 20191004F - SAM: corrected 'stpr2 - zstp -1 ' to 'stpr2 - zstp + 1'
         zwtr1  = ( stpr2 - zstp + 1  ) * zden
         zwtr2  = ( zstp - 1 - stpr1 ) * zden
         ub(:,:,:)     = zwtr1 * unr1    (:,:,:) + zwtr2 * unr2    (:,:,:)
         vb(:,:,:)     = zwtr1 * vnr1    (:,:,:) + zwtr2 * vnr2    (:,:,:)
         tsb(:,:,:,jp_tem)     = zwtr1 * tnr1    (:,:,:) + zwtr2 * tnr2    (:,:,:)
         tsb(:,:,:,jp_sal)     = zwtr1 * snr1    (:,:,:) + zwtr2 * snr2    (:,:,:)
!!! /20191004F

!!! 20191004C Added 'sshb' to trajectory
         sshb(:,:)     = zwtr1 * sshnr1    (:,:) + zwtr2 * sshnr2    (:,:)
!!! /20191004C

!!! 20191004G - SAM: adjusted outputting of interpolation coefficients
         IF(lwp)WRITE(numout,*) '   interp. coef. for "before" time lev.: ', &
            & zwtr1, zwtr2
         zwtr1  = ( stpr2 - zstp      ) * zden
         zwtr2  = ( zstp  - stpr1     ) * zden      
      END IF
!!! /20191004G

      IF ( kstp == nit000-1 ) THEN
         tsb(:,:,:,:) = tsn(:,:,:,:)
         ub(:,:,:) = un(:,:,:)
         vb(:,:,:) = vn(:,:,:)
      END IF
      avmu(:,:,:)   = zwtr1 * avmur1  (:,:,:) + zwtr2 * avmur2  (:,:,:)
      avmv(:,:,:)   = zwtr1 * avmvr1  (:,:,:) + zwtr2 * avmvr2  (:,:,:)
      avt(:,:,:)    = zwtr1 * avtr1   (:,:,:) + zwtr2 * avtr2   (:,:,:)
#if defined key_ldfslp
      uslp(:,:,:)   = zwtr1 * uslpr1  (:,:,:) + zwtr2 * uslpr2  (:,:,:)
      vslp(:,:,:)   = zwtr1 * vslpr1  (:,:,:) + zwtr2 * vslpr2  (:,:,:)
      wslpi(:,:,:)  = zwtr1 * wslpir1 (:,:,:) + zwtr2 * wslpir2 (:,:,:)
      wslpj(:,:,:)  = zwtr1 * wslpjr1 (:,:,:) + zwtr2 * wslpjr2 (:,:,:)
#endif
#if defined key_zdfddm
      avs(:,:,:)    = zwtr1 * avsr1   (:,:,:) + zwtr2 * avsr2   (:,:,:)
#endif
      etot3(:,:,:)  = zwtr1 * etot3r1 (:,:,:) + zwtr2 * etot3r2 (:,:,:)
#if defined key_tradmp
      hmlp(:,:)     = zwtr1 * hmlp1(:,:)    + zwtr2 * hmlp2(:,:)
#endif
#if defined key_traldf_eiv
#if defined key_traldf_c3d
      aeiu(:,:,:)   = zwtr1 * aeiur1  (:,:,:) + zwtr2 * aeiur2  (:,:,:)
      aeiv(:,:,:)   = zwtr1 * aeivr1  (:,:,:) + zwtr2 * aeivr2  (:,:.:)
      aeiw(:,:,:)   = zwtr1 * aeiwr1  (:,:,:) + zwtr2 * aeiwr2  (:,:.:)
#elif defined key_traldf_c2d
      aeiu(:,:)     = zwtr1 * aeiur1  (:,:)   + zwtr2 * aeiur2  (:,:)
      aeiv(:,:)     = zwtr1 * aeivr1  (:,:)   + zwtr2 * aeivr2  (:,:)
      aeiw(:,:)     = zwtr1 * aeiwr1  (:,:)   + zwtr2 * aeiwr2  (:,:)
#elif defined key_traldf_c1d
      aeiu(:)       = zwtr1 * aeiur1  (:)     + zwtr2 * aeiur2  (:)
      aeiv(:)       = zwtr1 * aeivr1  (:)     + zwtr2 * aeivr2  (:)
      aeiw(:)       = zwtr1 * aeiwr1  (:)     + zwtr2 * aeiwr2  (:)
#else
      aeiu          = zwtr1 * aeiur1          + zwtr2 * aeiur2
      aeiv          = zwtr1 * aeivr1          + zwtr2 * aeivr2
      aeiw          = zwtr1 * aeiwr1          + zwtr2 * aeiwr2
#endif
#endif

      CALL ssh_wzv( kstp )

   END SUBROUTINE trj_rea


   SUBROUTINE trj_wri_spl(filename)
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE trj_wri_spl ***
      !!
      !! ** Purpose : Write SimPLe data to file the model state trajectory
      !!
      !! ** Method  :
      !!
      !! ** Action  :
      !!
      !! History :
      !!        ! 09-07 (F. Vigilant)
      !!-----------------------------------------------------------------------
      !! *Module udes
      USE iom
      USE sol_oce, ONLY : & ! solver variables
      & gcb, gcx
      !! * Arguments
      !! * Local declarations
      INTEGER :: &
         & inum, &                  ! File unit number
         & fd                       ! field number
      CHARACTER (LEN=50) :: &
         & filename

      fd=1
      WRITE(filename, FMT='(A,A)' ) TRIM( filename ), '.nc'
      filename = TRIM( filename )
      CALL iom_open( filename, inum, ldwrt = .TRUE., kiolib = jprstlib)

      ! Output trajectory fields
      CALL iom_rstput( fd, fd, inum, 'un'   , un   )
      CALL iom_rstput( fd, fd, inum, 'vn'   , vn   )
      CALL iom_rstput( fd, fd, inum, 'tn'   , tsn(:,:,:,jp_tem)   )
      CALL iom_rstput( fd, fd, inum, 'sn'   , tsn(:,:,:,jp_sal)   )
      CALL iom_rstput( fd, fd, inum, 'sshn' , sshn )
      CALL iom_rstput( fd, fd, inum, 'wn'   , wn   )
      CALL iom_rstput( fd, fd, inum, 'tb'   , tsb(:,:,:,jp_tem)   )
      CALL iom_rstput( fd, fd, inum, 'sb'   , tsb(:,:,:,jp_sal)   )
      CALL iom_rstput( fd, fd, inum, 'ua'   , ua   )
      CALL iom_rstput( fd, fd, inum, 'va'   , va   )
      CALL iom_rstput( fd, fd, inum, 'ta'   , tsa(:,:,:,jp_tem)   )
      CALL iom_rstput( fd, fd, inum, 'sa'   , tsa(:,:,:,jp_sal)   )
      CALL iom_rstput( fd, fd, inum, 'sshb' , sshb )
      CALL iom_rstput( fd, fd, inum, 'rhd'  , rhd  )
      CALL iom_rstput( fd, fd, inum, 'rhop' , rhop )
      CALL iom_rstput( fd, fd, inum, 'gtu'  , gtsu(:,:,jp_tem)  )
      CALL iom_rstput( fd, fd, inum, 'gsu'  , gtsu(:,:,jp_sal)  )
      CALL iom_rstput( fd, fd, inum, 'gru'  , gru  )
      CALL iom_rstput( fd, fd, inum, 'gtv'  , gtsv(:,:,jp_tem)  )
      CALL iom_rstput( fd, fd, inum, 'gsv'  , gtsv(:,:,jp_sal)  )
      CALL iom_rstput( fd, fd, inum, 'grv'  , grv  )
      CALL iom_rstput( fd, fd, inum, 'rn2'  , rn2  )
      CALL iom_rstput( fd, fd, inum, 'gcb'  , gcb  )
      CALL iom_rstput( fd, fd, inum, 'gcx'  , gcx  )

      CALL iom_close( inum )

   END SUBROUTINE trj_wri_spl

   SUBROUTINE trj_rd_spl(filename)
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE asm_trj__wop_rd ***
      !!
      !! ** Purpose : Read SimPLe data from file the model state trajectory
      !!
      !! ** Method  :
      !!
      !! ** Action  :
      !!
      !! History :
      !!        ! 09-07 (F. Vigilant)
      !!-----------------------------------------------------------------------
      !! *Module udes
      USE iom                 ! I/O module
      USE sol_oce, ONLY : & ! solver variables
      & gcb, gcx
      !! * Arguments
      !! * Local declarations
      INTEGER :: &
         & inum, &                  ! File unit number
         & fd                       ! field number
      CHARACTER (LEN=50) :: &
         & filename

      fd=1
      WRITE(filename, FMT='(A,A)' ) TRIM( filename ), '.nc'
      filename = TRIM( filename )
      CALL iom_open( filename, inum)

      ! Output trajectory fields
      CALL iom_get( inum, jpdom_autoglo, 'un'   , un,   fd )
      CALL iom_get( inum, jpdom_autoglo, 'vn'   , vn,   fd )
      CALL iom_get( inum, jpdom_autoglo, 'tn'   , tsn(:,:,:,jp_tem),   fd )
      CALL iom_get( inum, jpdom_autoglo, 'sn'   , tsn(:,:,:,jp_sal),   fd )
      CALL iom_get( inum, jpdom_autoglo, 'sshn' , sshn, fd )
      CALL iom_get( inum, jpdom_autoglo, 'wn'   , wn,   fd )
      CALL iom_get( inum, jpdom_autoglo, 'tb'   , tsb(:,:,:,jp_tem),   fd )
      CALL iom_get( inum, jpdom_autoglo, 'sb'   , tsb(:,:,:,jp_sal),   fd )
      CALL iom_get( inum, jpdom_autoglo, 'ua'   , ua,   fd )
      CALL iom_get( inum, jpdom_autoglo, 'va'   , va,   fd )
      CALL iom_get( inum, jpdom_autoglo, 'ta'   , tsa(:,:,:,jp_tem),   fd )
      CALL iom_get( inum, jpdom_autoglo, 'sa'   , tsa(:,:,:,jp_sal),   fd )
      CALL iom_get( inum, jpdom_autoglo, 'sshb' , sshb, fd )
      CALL iom_get( inum, jpdom_autoglo, 'rhd'  , rhd,  fd )
      CALL iom_get( inum, jpdom_autoglo, 'rhop' , rhop, fd )
      CALL iom_get( inum, jpdom_autoglo, 'gtu'  , gtsu(:,:,jp_tem),  fd )
      CALL iom_get( inum, jpdom_autoglo, 'gsu'  , gtsu(:,:,jp_sal),  fd )
      CALL iom_get( inum, jpdom_autoglo, 'gru'  , gru,  fd )
      CALL iom_get( inum, jpdom_autoglo, 'gtv'  , gtsv(:,:,jp_tem),  fd )
      CALL iom_get( inum, jpdom_autoglo, 'gsv'  , gtsv(:,:,jp_sal),  fd )
      CALL iom_get( inum, jpdom_autoglo, 'grv'  , grv,  fd )
      CALL iom_get( inum, jpdom_autoglo, 'rn2'  , rn2,  fd )
      CALL iom_get( inum, jpdom_autoglo, 'gcb'  , gcb,  fd )
      CALL iom_get( inum, jpdom_autoglo, 'gcx'  , gcx,  fd )

      CALL iom_close( inum )

   END SUBROUTINE trj_rd_spl


!!! 20191004H - switch to allow adjoint output
   SUBROUTINE tl_trj_wri(kstp, kadj)
   !SUBROUTINE tl_trj_wri(kstp)
!!!/20191004H

      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE tl_trj_wri ***
      !!
      !! ** Purpose : Write SimPLe data to file the model state trajectory
      !!
      !! ** Method  :
      !!
      !! ** Action  :
      !!
      !! History :
      !!        ! 10-07 (F. Vigilant)
      !!-----------------------------------------------------------------------
      !! *Module udes
      USE iom

      !! * Arguments
      INTEGER, INTENT(in) :: &
         & kstp           ! Step for requested trajectory

!!! 20191004H switch to allow adjoint output
      INTEGER, INTENT(in), OPTIONAL :: kadj
!!! /20191004H

      !! * Local declarations
      INTEGER :: &
         & inum           ! File unit number
      INTEGER :: &
         & it
      CHARACTER (LEN=50) :: &
         & filename
      CHARACTER (LEN=100) :: &
         & cl_tantrj
!!! 20191004B - single variable TAM output: temporary variable to sum "b" and "n" variables
      REAL(wp), POINTER, DIMENSION(:,:,:) :: zicapprox3d 
!!! /20191004B

      ! Initialize data and open file
      !! if step time is corresponding to a saved state
      IF ( ( MOD( kstp - nit000 + 1, nn_ittrjfrq_tan ) == 0 )  ) THEN
!!! 20191004B - single variable TAM output: temporary variable to sum "b" and "n" variables
         CALL wrk_alloc(jpi, jpj, jpk, zicapprox3d)
!!! /20191004B
         it = kstp - nit000 + 1

            ! Define the input file
!!! 20191004D adjust filename to accommodate longer runs (100e6 time steps)
            !WRITE(cl_tantrj, FMT='(I5.5, A,A,".nc")' ) it, '_', TRIM( cn_tantrj )
            WRITE(cl_tantrj, FMT='(A,A,I0.8,".nc")' ) TRIM( cn_tantrj ), '_', it
!!! /20191004D
            cl_tantrj = TRIM( cl_tantrj )

            IF(lwp) THEN
               WRITE(numout,*)
               WRITE(numout,*)'Writing linear-tangent fields from : ',TRIM(cl_tantrj)
               WRITE(numout,*)
            ENDIF

            CALL iom_open( cl_tantrj, inum, ldwrt = .TRUE., kiolib = jprstlib)

            ! Output trajectory fields
            CALL iom_rstput( it, it, inum, 'un'   , un   )
            CALL iom_rstput( it, it, inum, 'vn'   , vn   )
            CALL iom_rstput( it, it, inum, 'wn'   , wn   )
            CALL iom_rstput( it, it, inum, 'tn'   , tsn(:,:,:,jp_tem)   )
            CALL iom_rstput( it, it, inum, 'sn'   , tsn(:,:,:,jp_sal)   )
            CALL iom_rstput( it, it, inum, 'sshn' , sshn )
!!! 20191004H - switch to allow adjoint output 
            IF (.not.PRESENT(kadj)) THEN
!!! /20191004H

               CALL iom_rstput( it, it, inum, 'tn_tl'   , tsn_tl(:,:,:,jp_tem)   )
               CALL iom_rstput( it, it, inum, 'sn_tl'   , tsn_tl(:,:,:,jp_sal)   )
               CALL iom_rstput( it, it, inum, 'un_tl'   , un_tl   )               
               CALL iom_rstput( it, it, inum, 'vn_tl'   , vn_tl   )
               CALL iom_rstput( it, it, inum, 'wn_tl'   , wn_tl   )
!!! 20191004B - also limiting TLM output variables
               !CALL iom_rstput( it, it, inum, 'hdivn_tl', hdivn_tl)
               !CALL iom_rstput( it, it, inum, 'rotn_tl' , rotn_tl )
               !CALL iom_rstput( it, it, inum, 'rhd_tl' , rhd_tl )
               !CALL iom_rstput( it, it, inum, 'rhop_tl' , rhop_tl )
!!! /20191004B
               CALL iom_rstput( it, it, inum, 'sshn_tl' , sshn_tl )
!!! 20191004H - switch to allow adjoint output
            ELSE
               CALL iom_rstput(it, it, inum, 'wn_ad',    wn_ad)
!!! 20191004B - also limiting output variables for brevity
               !CALL iom_rstput(it, it, inum, 'hdivn_ad', hdivn_ad)
               !CALL iom_rstput(it, it, inum, 'rotn_ad',  rotn_ad)
               !CALL iom_rstput(it, it, inum, 'rhd_ad',   rhd_ad)
               !CALL iom_rstput(it, it, inum, 'rhop_ad',  rhop_ad)
!!! /20191004B
               CALL iom_rstput(it, it, inum, 'sshn_ad',  sshn_ad)
!!! 20191004B - combining multiple variables into a single output variable
               zicapprox3d(:,:,:) = tsn_ad(:,:,:,jp_tem) + tsb_ad(:,:,:,jp_tem)
               CALL lbc_lnk_adj(zicapprox3d(:,:,:), 'T', 1.0_wp)
               CALL iom_rstput( it, it, inum, 't_ad', zicapprox3d)
               zicapprox3d(:,:,:) = tsn_ad(:,:,:,jp_sal) + tsb_ad(:,:,:,jp_sal)
               CALL lbc_lnk_adj(zicapprox3d(:,:,:), 'T', 1.0_wp)
               CALL iom_rstput( it, it, inum, 's_ad', zicapprox3d)
               zicapprox3d(:,:,:) = vn_ad(:,:,:) + vb_ad(:,:,:)
               CALL lbc_lnk_adj(zicapprox3d(:,:,:), 'V', -1.0_wp)
               CALL iom_rstput( it, it, inum, 'v_ad', zicapprox3d)
               zicapprox3d(:,:,:) = un_ad(:,:,:) + ub_ad(:,:,:)
               CALL lbc_lnk_adj(zicapprox3d(:,:,:), 'U', -1.0_wp)
               CALL iom_rstput( it, it, inum, 'u_ad', zicapprox3d)                        
            END IF
!!!20191129A added trajectory ice fraction to output of adjoint
               CALL iom_rstput( it, it, inum, 'ice_fraction', fr_i )
!!!/20191129A

!!! /20191004B

!!! /20191004H
            CALL iom_close( inum )
!!! 20191004B Combining multiple variables into a single output variable
            CALL wrk_dealloc(jpi, jpj, jpk, zicapprox3d)
!!! /20191004B
         ENDIF

   END SUBROUTINE tl_trj_wri


   SUBROUTINE trj_deallocate
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE trj_deallocate ***
      !!
      !! ** Purpose : Deallocate saved trajectory arrays
      !!
      !! ** Method  :
      !!
      !! ** Action  :
      !!
      !! History :
      !!        ! 2010-06 (A. Vidard)
      !!-----------------------------------------------------------------------

         IF ( ln_mem ) THEN
            DEALLOCATE(  &
               & empr1,  &
               & empsr1, &
               & empr2,  &
               & empsr2, &
               & bfruar1,&
               & bfrvar1,&
               & bfruar2,&
               & bfrvar2 &
               & )

            DEALLOCATE(    &
               & unr1,     &
               & vnr1,     &
               & tnr1,     &
               & snr1,     &
               & avmur1,   &
               & avmvr1,   &
               & avtr1,    &
               & etot3r1,  &
               & unr2,     &
               & vnr2,     &
               & tnr2,     &
               & snr2,     &
               & avmur2,   &
               & avmvr2,   &
               & avtr2,    &
               & etot3r2,  &
!!! 20191004C adding ssh to nonlinear trajectory
               & sshnr1,   & 
               & sshnr2    &
!!! /20191004C
               & )

#if defined key_traldf_eiv
#if defined key_traldf_c3d
#elif defined key_traldf_c2d
            DEALLOCATE(  &
               & aeiur1, &
               & aeivr1, &
               & aeiwr1, &
               & aeiur2, &
               & aeivr2, &
               & aeiwr2  &
               & )
#elif defined key_traldf_c1d
#endif
#endif

#if defined key_ldfslp
            DEALLOCATE(    &
               & uslpr1,   &
               & vslpr1,   &
               & wslpir1,  &
               & wslpjr1,  &
               & uslpr2,   &
               & vslpr2,   &
               & wslpir2,  &
               & wslpjr2   &
               & )
#endif

#if defined key_zdfddm
            DEALLOCATE(    &
               & avsr1,    &
               & avsr2     &
               & )
#endif

#if defined key_tradmp
            DEALLOCATE(    &
               & hmlp1,    &
               & hmlp2     &
               & )
#endif
	 ENDIF
         END SUBROUTINE trj_deallocate
#endif
END MODULE trj_tam
