!
MODULE tamtrj
   !!======================================================================
   !!                       ***  MODULE tamtrj  ***
   !! Tangent and adjoint  trajectory interface: Write to file
   !!                                    the model state trajectory
   !!======================================================================
   !!----------------------------------------------------------------------
   !!   tam_trj_ini  : init: read the namelist
   !!   tam_trj_wri  : Write out the model state trajectory
   !!----------------------------------------------------------------------
   !! * Modules used
   USE oce                ! Dynamics and active tracers defined in memory
   USE sbc_oce            ! Ocean surface boundary conditions
   USE zdf_oce            ! Vertical mixing variables
#if defined key_zdfddm
   USE zdfddm             ! Double diffusion mixing parameterization
#endif
   USE ldftra_oce         ! Lateral tracer mixing coefficient defined in memory
#if   defined key_ldfslp
   USE ldfslp
#endif
#if defined key_traldf_c2d
   USE ldfeiv             ! eddy induced velocity coef.      (ldf_eiv routine)
#endif
#if defined key_zdftke
   USE zdftke             ! TKE vertical physics
#endif
   USE eosbn2             ! Equation of state (eos_bn2 routine)
   USE zdfbfr
   USE tradmp             ! Tracer damping
   USE sol_oce
   USE trc_oce
   USE in_out_manager
   USE dom_oce
   USE iom                 ! I/O module
   USE zdfmxl
   IMPLICIT NONE

   !! * Routine accessibility
   PRIVATE
   PUBLIC tam_trj_init, &  !: Write out the background state
      &   tam_trj_wri      !: Write out the background state

   LOGICAL, PUBLIC :: &
      & ln_trjhand = .FALSE.   !: No output of the state trajectory fields

   LOGICAL :: &
      & ln_trj_spl            !: save the trajectory at simple precision

   CHARACTER (LEN=40), PUBLIC :: &
      & cn_dirtrj                                  !: Filename for storing the
                                                   !: reference trajectory
   INTEGER, PUBLIC :: &
      & nn_ittrjfrq, &         !: Frequency of trajectory output for 4D-VAR
!!! 20191004R - trajectory offsetting
      & nn_ittrjoffset
!!! /20191004R


CONTAINS
   SUBROUTINE tam_trj_init
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE tam_trj_init ***
      !!
      !! ** Purpose : initialize the model state trajectory
      !!
      !! ** Method  :
      !!
      !! ** Action  :
      !!
      !! References :
      !!
      !! History :
      !!        ! 10-01 (A. Vidard)
      !!-----------------------------------------------------------------------

      IMPLICIT NONE

      !! * Modules used
      NAMELIST/namtrj/ nn_ittrjfrq, ln_trjhand, cn_dirtrj, ln_trj_spl, &
!!! 20191004R - trajectory offsetting
      & nn_ittrjoffset
!!! /20191004R
      cn_dirtrj   = 'tam_trajectory'
      ln_trjhand = .FALSE.
      nn_ittrjfrq = 1
      ln_trj_spl  = .TRUE.
!!! 20191004R - trajectory offsetting
      nn_ittrjoffset = 0
!!! /20191004R

      READ   ( numnam, namtrj )

      ! Control print
      IF(lwp) THEN
         WRITE(numout,*)
         WRITE(numout,*) 'tam_trj_ini : Trajectory handling:'
         WRITE(numout,*) '~~~~~~~~~~~~'
         WRITE(numout,*) '          Namelist namtam : set trajectory parameters'
         WRITE(numout,*) '             Logical switch for handling state trajectory         ', &
            &            ' ln_trjhand = ', ln_trjhand
         WRITE(numout,*) '             Logical switch for handling it at simple precision       ', &
            &            ' ln_trj_spl  = ', ln_trj_spl
         WRITE(numout,*) '             Frequency of trajectory output (or input for TAM)                          ', &
            &            ' nn_ittrjfrq = ', nn_ittrjfrq
!!! 20191004R - trajectory offsetting
         WRITE(numout,*) '             Offset used in labelling trajectory output ', &
            &            ' nn_ittrjoffset = ', nn_ittrjoffset
!!! /20191004R
      END IF
   END SUBROUTINE tam_trj_init
   SUBROUTINE tam_trj_wri( kt )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE tam_trj_wri ***
      !!
      !! ** Purpose : Write to file the model state trajectory
      !!
      !! ** Method  :
      !!
      !! ** Action  :
      !!
      !! References :
      !!
      !! History :
      !!        ! 2007-04 (A. Weaver)
      !!        ! 2009-03 (F. Vigilant) Add hmlp (zdfmxl) for no tracer nmldp=2
      !!        ! 2009-06 (F. Vigilant) special case when kt=nit000-1
      !!        ! 2009-07 (F. Vigilant) add computation of eiv at restart
      !!        ! 2010-01 (A. Vidard) asm_trj_wri->tam_trj_wri
      !!        ! 2010-04 (F. Vigilant) converison to 3.2
      !!-----------------------------------------------------------------------

      !! * Arguments
      INTEGER, INTENT( IN ) :: &
         & kt                    ! Current time-step

      !! * Local declarations
      INTEGER :: &
         & inum                  ! File unit number
      INTEGER :: &
         & it
      INTEGER :: &
         & ntype
      CHARACTER (LEN=50) :: &
         & cl_dirtrj
      REAL(wp) :: &
         & zdate            ! Date

      IF ( ln_trj_spl ) THEN ; ntype = jp_r4
                        ELSE ; ntype = jp_r8 ; ENDIF
      !------------------------------------------------------------------------
      ! Write a single file for each trajectory time step
      !------------------------------------------------------------------------
      !------------------------------------------------------------------------
      ! Write a single file for each trajectory time step
      !------------------------------------------------------------------------
      IF( ( MOD( kt - nit000 + 1, nn_ittrjfrq ) == 0 ) .OR. ( kt == nitend ) ) THEN

         IF( kt == nit000 - 1 ) THEN         ! Treat special case when kt = nit000-1
            !
#if defined key_zdftke
            IF(lwp) WRITE(numout,*) ' Computing  zdf_tke coeff. form restart...'
            ! Compute the vertical eddy viscosity and diffusivity coefficients
            CALL zdf_tke( nit000 )
#endif
#if defined key_zdfddm
            IF(lwp) WRITE(numout,*) ' Computing zdf_ddm coeff. from restart...'
            ! Compute the vertical eddy viscosity and diffusivity coefficients (salt effect)
            CALL zdf_ddm( nit000 )
#endif
            IF(lwp) WRITE(numout,*) ' Computing zdf_mxl coeff. from restart...'
            ! Compute the turbocline depth and the mixed layer depth
            CALL zdf_mxl( nit000 )
#if defined key_ldfslp
            IF(lwp) WRITE(numout,*) ' Compute the slopes of neutral surface...'
            CALL bn2( tsb, rn2 )
            CALL ldf_slp( nit000, rhd, rn2 )
#endif
#if defined key_traldf_c2d
            IF(lwp) WRITE(numout,*) ' Computing ldf_eiv coeff. from restart...'
            ! Compute eddy induced velocity coefficient
            IF( lk_traldf_eiv )   CALL ldf_eiv( nit000 )
#endif
         ENDIF
         !
!!! 20191004R - trajectory offsetting
         !it = kt - nit000 + 1
         it = kt - nit000 + 1 + nn_ittrjoffset
!!! /20191004R
         !
         ! Define the output file

!!!20191004D expanding filenames to allow longer runs
         !WRITE(cl_dirtrj, FMT='(I5.5,A,A)' ) it, '_', TRIM( cn_dirtrj )
         WRITE(cl_dirtrj, FMT='(A,A,I0.8)' ) TRIM( cn_dirtrj ), '_', it 
!!!/20191004D

         cl_dirtrj = TRIM( cl_dirtrj )
         CALL iom_open( cl_dirtrj, inum, ldwrt = .TRUE., kiolib = jprstlib)

         ! Output trajectory fields
         CALL iom_rstput( it, it, inum, 'emp'   , emp   , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'emps'  , emps  , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'un'    , un    , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'vn'    , vn    , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'tn'    , tsn(:,:,:,jp_tem)    , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'sn'    , tsn(:,:,:,jp_sal)    , ktype = ntype )
!!! 20191004C added sshn to trajectory
         CALL iom_rstput( it, it, inum, 'sshn'  , sshn  , ktype = ntype )
!!! /20191004C
         CALL iom_rstput( it, it, inum, 'avmu'  , avmu  , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'avmv'  , avmv  , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'avt'   , avt   , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'bfrua' , bfrua , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'bfrva' , bfrva , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'etot3' , etot3 , ktype = ntype )
#if defined key_ldfslp
         CALL iom_rstput( it, it, inum, 'uslp'  , uslp  , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'vslp'  , vslp  , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'wslpi' , wslpi , ktype = ntype )
         CALL iom_rstput( it, it, inum, 'wslpj' , wslpj , ktype = ntype )
#endif
#if defined key_zdfddm
         CALL iom_rstput( it, it, inum, 'avs'   , avs , ktype = ntype )
#endif
         IF( ln_tradmp ) THEN
            CALL iom_rstput( it, it, inum, 'hmlp'  , hmlp  , ktype = ntype )
            CALL iom_rstput( it, it, inum, 'strdmp', strdmp, ktype = ntype )
         END IF
         CALL iom_rstput( it, it, inum, 'aeiu'  , aeiu, ktype = ntype )
         CALL iom_rstput( it, it, inum, 'aeiv'  , aeiv, ktype = ntype )
         CALL iom_rstput( it, it, inum, 'aeiw'  , aeiw, ktype = ntype )
!!!20191129A adding ice fraction to trajectory and adjoint output
         CALL iom_rstput( it, it, inum, 'ice_fraction', fr_i, ktype = ntype )
!!!/20191129A
         CALL iom_close( inum )

   ENDIF

 END SUBROUTINE tam_trj_wri
END MODULE tamtrj
