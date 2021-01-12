!
MODULE traadv_tam
#if defined key_tam
   !!==============================================================================
   !!                       ***  MODULE  traadv_tam  ***
   !! Ocean active tracers:  advection trend -
   !!                        Tangent and Adjoint Module
   !!==============================================================================
   !! History of the direct routine:
   !!            9.0  !  05-11  (G. Madec)  Original code
   !! History of the TAM:
   !!                 !  08-06  (A. Vidard) Skeleton
   !!                 !  09-03  (F. Vigilant) Add tra_adv_ctl_tam routine
   !!                                         Allow call to tra_eiv_tam/adj
   !!                 ! 12-07   (P.-A. Bouttier) Phase with 3.4 version
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   tra_adv      : compute ocean tracer advection trend
   !!   tra_adv_init  : control the different options of advection scheme
   !!----------------------------------------------------------------------
   USE par_oce
   USE oce_tam
   USE oce
   USE ldftra_oce
   USE dom_oce         ! ocean space and time domain
   USE traadv_cen2_tam
   USE traadv_tvd 
   USE in_out_manager  ! I/O manager
   USE prtctl          ! Print control
   USE cla_tam
   USE iom
   USE lib_mpp
   USE wrk_nemo
   USE timing
!!! 20191004S,  20191004T - namelist parameters for advection schemes
   USE tamctl
!!! /20191004S, /20191004T
!!! 20191004W - addition of EIV to trajectory fields
   USE traadv_eiv
!!! /20191004W

   IMPLICIT NONE
   PRIVATE

   PUBLIC   tra_adv_tan     ! routine called by steptan module
   PUBLIC   tra_adv_adj     ! routine called by stepadj module
   PUBLIC   tra_adv_init_tam ! routine called by stepadj module
   PUBLIC   tra_adv_adj_tst ! routine called by tst module

   !!* Namelist nam_traadv
   LOGICAL, PUBLIC ::   ln_traadv_cen2   = .TRUE.       ! 2nd order centered scheme flag
   LOGICAL, PUBLIC ::   ln_traadv_tvd    = .FALSE.      ! TVD scheme flag

!!!20191004S remove references to incompatible advection schemes
   !LOGICAL, PUBLIC ::   ln_traadv_muscl  = .FALSE.      ! MUSCL scheme flag
   !LOGICAL, PUBLIC ::   ln_traadv_muscl2 = .FALSE.      ! MUSCL2 scheme flag
   !LOGICAL, PUBLIC ::   ln_traadv_ubs    = .FALSE.      ! UBS scheme flag
   !LOGICAL, PUBLIC ::   ln_traadv_qck    = .FALSE.      ! QUICKEST scheme flag
!!!/20191004S

!!!20191004S, 20191004T - add weighted-mean and trajectory-upstream schemes
   REAL,    PUBLIC :: rn_traadv_weight_h = 0._wp
   REAL,    PUBLIC :: rn_traadv_weight_v = 0._wp
!!! 20191004S, 20191004T

   INTEGER ::   nadv   ! choice of the type of advection scheme

   !! * Substitutions
#  include "domzgr_substitute.h90"
#  include "vectopt_loop_substitute.h90"

CONTAINS

   SUBROUTINE tra_adv_tan( kt )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE tra_adv_tan  ***
      !!
      !! ** Purpose of the direct routine:
      !!              compute the ocean tracer advection trend.
      !!
      !! ** Method  : - Update (ua,va) with the advection term following nadv
      !!----------------------------------------------------------------------
      REAL(wp), POINTER, DIMENSION(:,:,:) ::   zuntl, zvntl, zwntl   ! effective velocity
      REAL(wp), POINTER, DIMENSION(:,:,:) ::   zun, zvn, zwn         ! effective velocity
      !!
      INTEGER, INTENT( in ) ::   kt   ! ocean time-step index
      INTEGER               ::   jk   ! dummy loop index
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )  CALL timing_start('tra_adv_tan')
      !
      CALL wrk_alloc( jpi, jpj, jpk, zun, zvn, zwn )
      CALL wrk_alloc( jpi, jpj, jpk, zuntl, zvntl, zwntl )
      !
      IF( neuler == 0 .AND. kt == nit000 ) THEN     !at nit000
         r2dtra(:) =  rdttra(:)                     !     = rdtra (restarting with Euler time stepping)
      ELSEIF( kt <= nit000 + 1) THEN                !at nit000 or nit000+1
         r2dtra(:) = 2._wp * rdttra(:)              !     = 2 rdttra (leapfrog)
      ENDIF
      !
      IF( nn_cla == 1 )   CALL cla_traadv_tan( kt )
      !
      !                                               !==  effective transport  ==!
      DO jk = 1, jpkm1
         zun(:,:,jk) = e2u(:,:) * fse3u(:,:,jk) * un(:,:,jk)                  ! eulerian transport only
         zvn(:,:,jk) = e1v(:,:) * fse3v(:,:,jk) * vn(:,:,jk)
         zwn(:,:,jk) = e1t(:,:) * e2t(:,:)      * wn(:,:,jk)
         zuntl(:,:,jk) = e2u(:,:) * fse3u(:,:,jk) * un_tl(:,:,jk)                  ! eulerian transport only
         zvntl(:,:,jk) = e1v(:,:) * fse3v(:,:,jk) * vn_tl(:,:,jk)
         zwntl(:,:,jk) = e1t(:,:) * e2t(:,:)      * wn_tl(:,:,jk)
      END DO
      zun(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      zvn(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      zwn(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      zuntl(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      zvntl(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      zwntl(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      !
!!! 20191004W - addition of EIV to trajectory velocity      
      IF (ln_tl_eiv) THEN
      ! next two lines are a copy of the corresponding call from file OPA_SRC/TRA/traadv.F90 (lines 96 and 97)
      IF( lk_traldf_eiv .AND. .NOT. ln_traldf_grif )   &
         &              CALL tra_adv_eiv( kt, nit000, zun, zvn, zwn, 'TRA' )
      END IF 
!!! /20191004W
      !
      IF ( kt == nit000 ) THEN
            nadv = 2
            IF(lwp) WRITE(numout,*) ' tra_adv_tam: 2nd order scheme is forced in TAM'
            nadv = 1 ! force tra_adv_cen2 for tangent
      END IF 

      SELECT CASE ( nadv )                           ! compute advection trend and add it to general trend
      CASE ( 1 ) ;
!!! 20191004S, 20191004T - addition of weighted-mean and upstream schemes
         !CALL tra_adv_cen2_tan ( kt, nit000, zun, zvn, zwn, tsn, zuntl, zvntl, zwntl, tsn_tl, tsa_tl, jpts )    ! 2nd order centered scheme
         CALL tra_adv_cen2_tan ( kt, nit000, zun, zvn, zwn, tsn, zuntl, zvntl, zwntl, tsn_tl, tsa_tl, jpts, rn_traadv_weight_h,rn_traadv_weight_v )    ! 2nd order centered scheme
!!! /201910004S, /20191004T
      END SELECT
      !
      IF( nn_timing == 1 )  CALL timing_stop('tra_adv_tan')
      !
      CALL wrk_dealloc( jpi, jpj, jpk, zun, zvn, zwn, zuntl, zvntl, zwntl )
      !
   END SUBROUTINE tra_adv_tan

   SUBROUTINE tra_adv_adj( kt )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE tra_adv_adj  ***
      !!
      !! ** Purpose of the direct routine:
      !!              compute the ocean tracer advection trend.
      !!
      !! ** Method  : - Update (ua,va) with the advection term following nadv
      !!----------------------------------------------------------------------
      REAL(wp), POINTER, DIMENSION(:,:,:) ::   zunad, zvnad, zwnad   ! effective velocity
      REAL(wp), POINTER, DIMENSION(:,:,:) ::   zun, zvn, zwn   ! effective velocity
      INTEGER :: jk
      !!
      INTEGER, INTENT( in ) ::   kt   ! ocean time-step index
      !!----------------------------------------------------------------------
      IF( nn_timing == 1 )  CALL timing_start('tra_adv_adj')
      !
      CALL wrk_alloc( jpi, jpj, jpk, zun, zvn, zwn )
      CALL wrk_alloc( jpi, jpj, jpk, zunad, zvnad, zwnad )
      !
      zunad(:,:,:) = 0._wp                                                     ! no transport trough the bottom
      zvnad(:,:,:) = 0._wp                                                     ! no transport trough the bottom
      zwnad(:,:,:) = 0._wp
      !
      IF( neuler == 0 .AND. kt == nit000 ) THEN     ! at nit000
         r2dtra(:) =  rdttra(:)                          ! = rdtra (restarting with Euler time stepping)
      ELSEIF( kt <= nit000 + 1) THEN                ! at nit000 or nit000+1
         r2dtra(:) = 2._wp * rdttra(:)                   ! = 2 rdttra (leapfrog)
      ENDIF
      !
      DO jk = 1, jpkm1
         zun(:,:,jk) = e2u(:,:) * fse3u(:,:,jk) * un(:,:,jk)                  ! eulerian transport only
         zvn(:,:,jk) = e1v(:,:) * fse3v(:,:,jk) * vn(:,:,jk)
         zwn(:,:,jk) = e1t(:,:) * e2t(:,:)      * wn(:,:,jk)
      END DO
      zun(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      zvn(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      zwn(:,:,jpk) = 0._wp                                                     ! no transport trough the bottom
      !
!!! 20191004W - incorporate EIV into trajectory velocity fields
      IF (ln_tl_eiv) THEN
      ! the next lines are a copy of the corresponding call from file OPA_SRC/TRA/traadv.F90 (lines 96 and 97)
      IF( lk_traldf_eiv .AND. .NOT. ln_traldf_grif )   &
         &              CALL tra_adv_eiv( kt, nit000, zun, zvn, zwn, 'TRA' )
      END IF
!!! /20191004W
      !
      IF ( kt == nitend ) THEN
         IF(lwp) WRITE(numout,*) ' tra_adv_tam: 2nd order scheme is forced in TAM'
         nadv = 1 ! force tra_adv_cen2 for adjoint
      END IF
      !
      SELECT CASE ( nadv )                           ! compute advection trend and add it to general trend
!!! 20191004S , 20191004T - addition of weighting parameters for upstream / weighted-mean schemes
      !CASE ( 1 ) ; CALL tra_adv_cen2_adj ( kt, nit000, zun, zvn, zwn, tsn, zunad, zvnad, zwnad, tsn_ad, tsa_ad, jpts )    ! 2nd order centered scheme
      CASE ( 1 ) ; CALL tra_adv_cen2_adj ( kt, nit000, zun, zvn, zwn, tsn, zunad, zvnad, zwnad, tsn_ad, tsa_ad, jpts, rn_traadv_weight_h,rn_traadv_weight_v )    ! 2nd order centered scheme
!!! /20191004S, /20191004T
      END SELECT
      !
      DO jk = jpkm1, 1, -1
         un_ad(:,:,jk) = un_ad(:,:,jk) + e2u(:,:) * fse3u(:,:,jk) *  zunad(:,:,jk)
         vn_ad(:,:,jk) = vn_ad(:,:,jk) + e1v(:,:) * fse3v(:,:,jk) *  zvnad(:,:,jk)
         wn_ad(:,:,jk) = wn_ad(:,:,jk) + e1t(:,:) * e2t(:,:) *  zwnad(:,:,jk)
      END DO
      !
      IF( nn_cla == 1 )   CALL cla_traadv_adj( kt )
      !
      !
      IF( nn_timing == 1 )  CALL timing_stop('tra_adv_adj')
      !
      CALL wrk_dealloc( jpi, jpj, jpk, zun, zvn, zwn, zunad, zvnad, zwnad )
      !
   END SUBROUTINE tra_adv_adj
   SUBROUTINE tra_adv_adj_tst( kumadt )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE example_adj_tst ***
      !!
      !! ** Purpose : Test the adjoint routine.
      !!
      !! ** Method  : Verify the scalar product
      !!
      !!                 ( L dx )^T W dy  =  dx^T L^T W dy
      !!
      !!              where  L   = tangent routine
      !!                     L^T = adjoint routine
      !!                     W   = diagonal matrix of scale factors
      !!                     dx  = input perturbation (random field)
      !!                     dy  = L dx
      !!
      !!
      !! History :
      !!        ! 08-08 (A. Vidard)
      !!-----------------------------------------------------------------------
      !! * Modules used

      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt             ! Output unit

      CALL tra_adv_cen2_adj_tst( kumadt )

   END SUBROUTINE tra_adv_adj_tst

   SUBROUTINE tra_adv_init_tam
      !!---------------------------------------------------------------------
      !!                  ***  ROUTINE tra_adv_ctl_tam  ***
      !!
      !! ** Purpose :   Control the consistency between namelist options for
      !!              tracer advection schemes and set nadv
      !!----------------------------------------------------------------------
      INTEGER ::   ioptio

      NAMELIST/namtra_adv_tam/ ln_traadv_cen2 , &
!!!20191004S, 20191004T - introduction of weighting parameters for upwind and weighted-mean scheme
         &                 rn_traadv_weight_h, rn_traadv_weight_v 
         !&                 ln_traadv_muscl, ln_traadv_muscl2, &
         !&                 ln_traadv_ubs  , ln_traadv_qck
!!! /20191004S, /20191004T
      !!----------------------------------------------------------------------

      REWIND( numnam )               ! Read Namelist nam_traadv : tracer advection scheme
      READ(numnam, namtra_adv_tam)

      IF(lwp) THEN                    ! Namelist print
         WRITE(numout,*)
         WRITE(numout,*) 'tra_adv_init_tam : choice/control of the tracer advection scheme'
         WRITE(numout,*) '~~~~~~~~~~~'
         WRITE(numout,*) '       Namelist nam_traadv_tam : chose a advection scheme for tracers'
         WRITE(numout,*) '          2nd order advection scheme     ln_traadv_cen2   = ', ln_traadv_cen2
         WRITE(numout,*) '          TVD advection scheme           ln_traadv_tvd    = ', ln_traadv_tvd
!!! 20191004T, 20191004S - addition of weighted mean scheme and varying weight upstream scheme
         !WRITE(numout,*) '          MUSCL  advection scheme        ln_traadv_muscl  = ', ln_traadv_muscl
         !WRITE(numout,*) '          MUSCL2 advection scheme        ln_traadv_muscl2 = ', ln_traadv_muscl2
         !WRITE(numout,*) '          UBS    advection scheme        ln_traadv_ubs    = ', ln_traadv_ubs
         !WRITE(numout,*) '          QUICKEST advection scheme      ln_traadv_qck    = ', ln_traadv_qck
         IF( rn_traadv_weight_h < 0.0_wp) THEN 
            WRITE(numout,*) '         Weighted-mean horizontal advection scheme selected '
         ELSE
            WRITE(numout,*) 'weighted centred v. upstream advection scheme, horizontal upstream weighting = ', rn_traadv_weight_h*100, '%'
         ENDIF
         IF( rn_traadv_weight_v < 0.0_wp) THEN 
            WRITE(numout,*) '         Weighted-mean vertical advection scheme selected '
         ELSE
            WRITE(numout,*) 'weighted centred v. upstream advection scheme, vertical upstream weighting = ', rn_traadv_weight_v*100, '%'
         ENDIF
!!!/20191004T, /20191004S


    ENDIF

      ioptio = 0                      ! Parameter control
      IF( ln_traadv_cen2   )   ioptio = ioptio + 1
      IF( ln_traadv_tvd    )   ioptio = ioptio + 1
!!!20191004T - remove references to incompatible schemes
      !IF( ln_traadv_muscl  )   ioptio = ioptio + 1
      !IF( ln_traadv_muscl2 )   ioptio = ioptio + 1
      !IF( ln_traadv_ubs    )   ioptio = ioptio + 1
      !IF( ln_traadv_qck    )   ioptio = ioptio + 1
!!! /20191004T
      IF( lk_esopa         )   ioptio =          1

      IF( ioptio /= 1 )   CALL ctl_stop( 'Choose ONE advection scheme in namelist nam_traadv' )

      IF( nn_cla == 1 .AND. .NOT. ln_traadv_cen2 )   &
         &                CALL ctl_stop( 'cross-land advection only with 2nd order advection scheme' )

      !                              ! Set nadv
      IF( ln_traadv_cen2   )   nadv =  1
      IF( ln_traadv_tvd    )   nadv =  2
!!! 20191004T remove references to incompatible schemes
      !IF( ln_traadv_muscl  )   nadv =  3
      !IF( ln_traadv_muscl2 )   nadv =  4
      !IF( ln_traadv_ubs    )   nadv =  5
      !IF( ln_traadv_qck    )   nadv =  6
!!! /20191004T
      IF( lk_esopa         )   nadv = -1

      IF(lwp) THEN                   ! Print the choice
         WRITE(numout,*)
         IF( nadv ==  1 )   WRITE(numout,*) '         2nd order scheme is used'
!!! 20191004T remove references to incompatible schemes
         !IF( nadv ==  3 )   WRITE(numout,*) '         MUSCL     Not Available in NEMO TAM'
         !IF( nadv ==  4 )   WRITE(numout,*) '         MUSCL2    Not Available in NEMO TAM'
         !IF( nadv ==  5 )   WRITE(numout,*) '         UBS       Not Available in NEMO TAM'
         !IF( nadv ==  6 )   WRITE(numout,*) '         QUICKEST  Not Available in NEMO TAM'
!!! /20191004T
         IF( nadv == -1 )   WRITE(numout,*) '         esopa test: Not Available in NEMO TAM'
      ENDIF
      !
   END SUBROUTINE tra_adv_init_tam
#endif


  !!======================================================================
END MODULE traadv_tam
