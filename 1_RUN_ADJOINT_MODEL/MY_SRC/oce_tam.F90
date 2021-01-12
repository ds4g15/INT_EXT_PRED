!!! 20191004N - essential modifications throughout: building of {t,u,v,f}msk_i variables using dom_uniq
!See also http://forge.ipsl.jussieu.fr/nemo/ticket/1499
!!! /20191004N
MODULE oce_tam
   !!----------------------------------------------------------------------
   !!    This software is governed by the CeCILL licence (Version 2)
   !!----------------------------------------------------------------------
   !!======================================================================
   !!                       ***  MODULE oce_tam ***
   !! NEMOVAR Tangent linear and Adjoint Model variables :
   !!
   !!    Allocate tangent linear and adjoint fields for the inner loop
   !!
   !!======================================================================
   !! History of the direct module:
   !!   8.5  !  02-11  (G. Madec)  F90: Free form and module
   !!   9.0  !  05-11  (V. Garnier) Surface pressure gradient organization
   !! History of the TAM module:
   !!   9.0  !  07-07 (K. Mogensen) Initial version
   !!   9.0  !  08-03 (A. Vidard) Add variables
   !!   9.0  !  09-03 (A. Weaver) Nemo v3 compatible, merge tl_init/ad_init
   !!        ! 2011-07 (D. Lea) Add altimeter bias and sea ice
   !! NEMO 3.4  ! 2012-04 (P.-A. Bouttier) update 3.4
   !!           ! 2012-09 (A. Vidard) Deallocating and initialising options
   !!----------------------------------------------------------------------
   !!   oce_tam_init : Allocate and initialize the TAM fields
   !!----------------------------------------------------------------------
   !! * Modules used
   USE par_oce
   USE lib_mpp
   USE dom_oce
   USE domwri

   IMPLICIT NONE

   !! * Routine accessibility
   PRIVATE

   !! dynamics and tracer fields                            ! before ! now    ! after  ! the after trends becomes the fields
   !! --------------------------                            ! fields ! fields ! trends ! only after tra_zdf and dyn_spg
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   ub_tl   ,  un_tl    , ua_tl     !: i-horizontal velocity        [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   vb_tl   ,  vn_tl    , va_tl     !: j-horizontal velocity        [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::              wn_tl                !: vertical velocity            [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   rotb_tl ,  rotn_tl              !: relative vorticity           [s-1]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   hdivb_tl,  hdivn_tl             !: horizontal divergence        [s-1]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::   tsb_tl  ,  tsn_tl               !: 4D T-S fields        [Celcius,psu]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   rn2b_tl ,  rn2_tl               !: brunt-vaisala frequency**2   [s-2]
   !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::  tsa_tl                           !: 4D T-S trends fields & work array
   !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   rhd_tl                            !: in situ density anomalie rhd=(rho-rau0)/rau0  [no units]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   rhop_tl                           !: potential volumic mass                           [kg/m3]

   !! free surface                                      !  before  ! now    ! after  !
   !! ------------                                      !  fields  ! fields ! trends !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:), TARGET ::   sshb_tl   , sshn_tl   , ssha_tl   !: sea surface height at t-point [m]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)         ::   sshu_b_tl , sshu_n_tl , sshu_a_tl !: sea surface height at u-point [m]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)         ::   sshv_b_tl , sshv_n_tl , sshv_a_tl !: sea surface height at u-point [m]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)         ::            sshf_n_tl                !: sea surface height at f-point [m]
   !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   spgu_tl, spgv_tl                          !: horizontal surface pressure gradient

   !! interpolated gradient (only used in zps case)
   !! ---------------------
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   gtsu_tl, gtsv_tl   !: horizontal gradient of T, S bottom u-point
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   gru_tl , grv_tl    !: horizontal gradient of rd at bottom u-point
   !
   LOGICAL, SAVE, PRIVATE :: ll_alloctl = .FALSE.
   !
   !! dynamics and tracer fields                            ! before ! now    ! after  ! the after trends becomes the fields
   !! --------------------------                            ! fields ! fields ! trends ! only after tra_zdf and dyn_spg
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   ub_ad   ,  un_ad    , ua_ad     !: i-horizontal velocity        [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   vb_ad   ,  vn_ad    , va_ad     !: j-horizontal velocity        [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::              wn_ad                !: vertical velocity            [m/s]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   rotb_ad ,  rotn_ad              !: relative vorticity           [s-1]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   hdivb_ad,  hdivn_ad             !: horizontal divergence        [s-1]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::   tsb_ad  ,  tsn_ad               !: 4D T-S fields        [Celcius,psu]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:)   ::   rn2b_ad ,  rn2_ad               !: brunt-vaisala frequency**2   [s-2]
   !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:,:) ::  tsa_ad                           !: 4D T-S trends fields & work array
   !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   rhd_ad                            !: in situ density anomalie rhd=(rho-rau0)/rau0  [no units]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   rhop_ad                           !: potential volumic mass                           [kg/m3]

   !! free surface                                      !  before  ! now    ! after  !
   !! ------------                                      !  fields  ! fields ! trends !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:), TARGET ::   sshb_ad   , sshn_ad   , ssha_ad   !: sea surface height at t-point [m]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)         ::   sshu_b_ad , sshu_n_ad , sshu_a_ad !: sea surface height at u-point [m]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)         ::   sshv_b_ad , sshv_n_ad , sshv_a_ad !: sea surface height at u-point [m]
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)         ::            sshf_n_ad                !: sea surface height at f-point [m]
   !
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:) ::   spgu_ad, spgv_ad                          !: horizontal surface pressure gradient

   !! interpolated gradient (only used in zps case)
   !! ---------------------
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::   gtsu_ad, gtsv_ad   !: horizontal gradient of T, S bottom u-point
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:)   ::   gru_ad , grv_ad    !: horizontal gradient of rd at bottom u-point

   LOGICAL, PRIVATE, SAVE :: ll_allocad = .FALSE.
#if defined key_zdfddm
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!  AW: The declaration/allocation/initialization of these variables
!!!!      should be moved to a new module zdf_ddm_tam_init to be consistent
!!!!      with NEMO.
   REAL(wp), PUBLIC, DIMENSION(:,:,:), ALLOCATABLE :: &
      & rrau_tl,  & !: Tangent Linear of heat/salt buoyancy flux ratio
      & rrau_ad     !: Adjoint of heat/salt buoyancy flux ratio
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#endif
   REAL(wp), PUBLIC, ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: &
        & tmsk_i, &
        & umsk_i, &
        & vmsk_i, &
        & fmsk_i

   !!----------------------------------------------------------------------
   !! NEMO/OPA 4.0 , NEMO Consortium (2011)
   !! $Id$
   !! Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
   !!----------------------------------------------------------------------

   PUBLIC oce_alloc_tam
   PUBLIC oce_dealloc_tam
   PUBLIC oce_tam_init

CONTAINS
   INTEGER FUNCTION oce_alloc_tam( kmode )
      !!----------------------------------------------------------------------
      !!                   ***  FUNCTION oce_alloc  ***
      !!----------------------------------------------------------------------
      INTEGER, OPTIONAL :: kmode
      INTEGER :: ierr(5)
      INTEGER :: jmode
      INTEGER :: jk
      REAL(wp), DIMENSION(jpi,jpj) :: &
           & z_tmsk_i, &
           & z_umsk_i, &
           & z_vmsk_i, &
           & z_fmsk_i

      !!----------------------------------------------------------------------
      !
      IF ( PRESENT(kmode) ) THEN
         jmode = kmode
      ELSE
         jmode = 0
      END IF

      IF (.NOT. ALLOCATED ( tmsk_i ) ) THEN
         ALLOCATE ( tmsk_i (jpi, jpj, jpk), &
              &       umsk_i (jpi, jpj, jpk), &
              &       vmsk_i (jpi, jpj, jpk), &
              &       fmsk_i (jpi, jpj, jpk) )
         
         CALL dom_uniq( z_tmsk_i, 'T', 1 )
         CALL dom_uniq( z_umsk_i, 'U', 1 )
         CALL dom_uniq( z_vmsk_i, 'V', 1 )
         CALL dom_uniq( z_fmsk_i, 'F', 1 )
         
         DO jk = 1, jpk
            
            tmsk_i(:,:,jk) = tmask(:,:,jk) * z_tmsk_i(:,:)
            umsk_i(:,:,jk) = umask(:,:,jk) * z_umsk_i(:,:)
            vmsk_i(:,:,jk) = vmask(:,:,jk) * z_vmsk_i(:,:)
            fmsk_i(:,:,jk) = fmask(:,:,jk) * z_fmsk_i(:,:)
            
         END DO
      END IF

      IF ( ( jmode == 0 ) .OR. ( jmode == 1 ) .AND. ( .NOT. ll_alloctl) ) THEN
      ALLOCATE( ub_tl   (jpi,jpj,jpk)      , un_tl   (jpi,jpj,jpk)      , ua_tl(jpi,jpj,jpk)       ,     &
         &      vb_tl   (jpi,jpj,jpk)      , vn_tl   (jpi,jpj,jpk)      , va_tl(jpi,jpj,jpk)       ,     &
         &      wn_tl   (jpi,jpj,jpk)      ,                                                       &
         &      rotb_tl (jpi,jpj,jpk)      , rotn_tl (jpi,jpj,jpk)      ,                             &
         &      hdivb_tl(jpi,jpj,jpk)      , hdivn_tl(jpi,jpj,jpk)      ,                             &
         &      tsb_tl  (jpi,jpj,jpk,jpts) , tsn_tl  (jpi,jpj,jpk,jpts) , tsa_tl(jpi,jpj,jpk,jpts) ,     &
         &      rn2b_tl (jpi,jpj,jpk)      , rn2_tl  (jpi,jpj,jpk)                              , STAT=ierr(1) )
         !
      ALLOCATE(rhd_tl (jpi,jpj,jpk) ,                                         &
         &     rhop_tl(jpi,jpj,jpk) ,                                         &
         &     sshb_tl  (jpi,jpj)   , sshn_tl  (jpi,jpj) , ssha_tl  (jpi,jpj) ,     &
         &     sshu_b_tl(jpi,jpj)   , sshu_n_tl(jpi,jpj) , sshu_a_tl(jpi,jpj) ,     &
         &     sshv_b_tl(jpi,jpj)   , sshv_n_tl(jpi,jpj) , sshv_a_tl(jpi,jpj) ,     &
         &                         sshf_n_tl(jpi,jpj) ,                       &
         &     spgu_tl  (jpi,jpj)   , spgv_tl(jpi,jpj)   ,                       &
         &     gtsu_tl(jpi,jpj,jpts), gtsv_tl(jpi,jpj,jpts),                     &
         &     gru_tl(jpi,jpj)      , grv_tl(jpi,jpj)                      , STAT=ierr(2) )

#if defined key_zdfddm
         ALLOCATE( rrau_tl(jpi,jpj,jpk), STAT=ierr(5) )
#endif
         !
         ll_alloctl = .TRUE.
      END IF
      !
      IF ( ( jmode == 0 ) .OR. ( jmode == 2 ) .AND. ( .NOT. ll_allocad)  ) THEN
         ALLOCATE( ub_ad   (jpi,jpj,jpk)      , un_ad   (jpi,jpj,jpk)      , ua_ad(jpi,jpj,jpk)       , &
            &      vb_ad   (jpi,jpj,jpk)      , vn_ad   (jpi,jpj,jpk)      , va_ad(jpi,jpj,jpk)       , &
            &      wn_ad   (jpi,jpj,jpk)      ,                                                         &
            &      rotb_ad (jpi,jpj,jpk)      , rotn_ad (jpi,jpj,jpk)      ,                            &
            &      hdivb_ad(jpi,jpj,jpk)      , hdivn_ad(jpi,jpj,jpk)      ,                            &
            &      tsb_ad  (jpi,jpj,jpk,jpts) , tsn_ad  (jpi,jpj,jpk,jpts) , tsa_ad(jpi,jpj,jpk,jpts) , &
            &      rn2b_ad (jpi,jpj,jpk)      , rn2_ad  (jpi,jpj,jpk)                              , STAT=ierr(3) )
         !
         ALLOCATE(rhd_ad (jpi,jpj,jpk) ,                                           &
            &     rhop_ad(jpi,jpj,jpk) ,                                           &
            &     sshb_ad  (jpi,jpj)   , sshn_ad  (jpi,jpj) , ssha_ad  (jpi,jpj) , &
            &     sshu_b_ad(jpi,jpj)   , sshu_n_ad(jpi,jpj) , sshu_a_ad(jpi,jpj) , &
            &     sshv_b_ad(jpi,jpj)   , sshv_n_ad(jpi,jpj) , sshv_a_ad(jpi,jpj) , &
            &                         sshf_n_ad(jpi,jpj) ,                         &
            &     spgu_ad  (jpi,jpj)   , spgv_ad(jpi,jpj)   ,                      &
            &     gtsu_ad(jpi,jpj,jpts), gtsv_ad(jpi,jpj,jpts),                    &
            &     gru_ad(jpi,jpj)      , grv_ad(jpi,jpj)                    , STAT=ierr(4) )

#if defined key_zdfddm
         ALLOCATE( rrau_ad(jpi,jpj,jpk), STAT=ierr(5) )
#endif

         ll_allocad = .TRUE.
      END IF
      oce_alloc_tam = MAXVAL( ierr )
      IF( oce_alloc_tam /= 0 )   CALL ctl_warn('oce_alloc_tam: failed to allocate arrays')
      !
   END FUNCTION oce_alloc_tam
   !
   INTEGER FUNCTION oce_dealloc_tam( kmode )
      !!----------------------------------------------------------------------
      !!                   ***  FUNCTION oce_dealloc  ***
      !!----------------------------------------------------------------------
      INTEGER, OPTIONAL :: kmode
      INTEGER :: jmode
      INTEGER :: ierr(5)
      !!----------------------------------------------------------------------
      !
      IF ( PRESENT(kmode) ) THEN
         jmode = kmode
      ELSE
         jmode = 0
      END IF

      IF ( ( jmode == 0 ) .OR. ( jmode == 1 ) .AND. ( ll_alloctl ) ) THEN
         DEALLOCATE( ub_tl         , un_tl         , ua_tl       , &
            &        vb_tl         , vn_tl         , va_tl       , &
            &        wn_tl         ,                               &
            &        rotb_tl       , rotn_tl       ,               &
            &        hdivb_tl      , hdivn_tl      ,               &
            &        tsb_tl        , tsn_tl        , tsa_tl      , &
            &        rn2b_tl       , rn2_tl                      , STAT=ierr(1) )
         !
         DEALLOCATE( rhd_tl     ,                                  &
            &       rhop_tl     ,                                  &
            &       sshb_tl     , sshn_tl   , ssha_tl   ,          &
            &       sshu_b_tl   , sshu_n_tl , sshu_a_tl ,          &
            &       sshv_b_tl   , sshv_n_tl , sshv_a_tl ,          &
            &       sshf_n_tl   ,                                  &
            &       spgu_tl     , spgv_tl   ,                      &
            &       gtsu_tl     , gtsv_tl   ,                      &
            &       gru_tl      , grv_tl                         , STAT=ierr(2) )

#if defined key_zdfddm
         DEALLOCATE( rrau_tl, STAT=ierr(5) )
#endif
         !
         ll_alloctl = .FALSE.
      END IF
      !
      IF ( ( jmode == 0 ) .OR. ( jmode == 2 ) .and. ( ll_allocad ) ) THEN
         DEALLOCATE( ub_ad       , un_ad         , ua_ad       , &
            &        vb_ad       , vn_ad         , va_ad       , &
            &        wn_ad       ,                               &
            &        rotb_ad     , rotn_ad       ,               &
            &        hdivb_ad    , hdivn_ad      ,               &
            &        tsb_ad      , tsn_ad        , tsa_ad      , &
            &        rn2b_ad     , rn2_ad                      , STAT=ierr(3) )
         !
         DEALLOCATE( rhd_ad      ,                               &
            &        rhop_ad     ,                               &
            &        sshb_ad     , sshn_ad   , ssha_ad   ,       &
            &        sshu_b_ad   , sshu_n_ad , sshu_a_ad ,       &
            &        sshv_b_ad   , sshv_n_ad , sshv_a_ad ,       &
            &                      sshf_n_ad ,                   &
            &        spgu_ad     , spgv_ad   ,                   &
            &        gtsu_ad     , gtsv_ad,                      &
            &        gru_ad      , grv_ad                      , STAT=ierr(4) )

#if defined key_zdfddm
         DEALLOCATE( rrau_ad, STAT=ierr(5) )
#endif

         ll_allocad = .FALSE.
      END IF
      oce_dealloc_tam = MAXVAL( ierr )
      IF( oce_dealloc_tam /= 0 )   CALL ctl_warn('oce_dealloc_tam: failed to deallocate arrays')
      !
   END FUNCTION oce_dealloc_tam
   !
   SUBROUTINE oce_tam_init( kmode )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE oce_tam_init  ***
      !!
      !! ** Purpose : Allocate and initialize the tangent linear and
      !!              adjoint arrays
      !!
      !! ** Method  : kindic = 0  allocate/initialize both tl and ad variables
      !!              kindic = 1  allocate/initialize only tl variables
      !!              kindic = 2  allocate/initialize only ad variables
      !!
      !! ** Action  :
      !!
      !! References :
      !!
      !! History :
      !!        ! 2009-03 (A. Weaver) Initial version (based on oce_tam_init)
      !!        ! 2010-04 (A. Vidard) Nemo3.2 update
      !!        ! 2012-09 (A. Vidard) Nemo3.4 update
      !!-----------------------------------------------------------------------
      !! * Arguments
      INTEGER, INTENT(IN) :: kmode
      INTEGER :: ierr
      IF ( ( kmode == 0 ) .OR. ( kmode == 1 ) ) THEN
         IF ( .NOT. ll_alloctl ) ierr = oce_alloc_tam ( 1 )
         ub_tl(:,:,:) = 0._wp
         vb_tl(:,:,:) = 0._wp
         un_tl(:,:,:) = 0._wp
         vn_tl(:,:,:) = 0._wp
         ua_tl(:,:,:) = 0._wp
         va_tl(:,:,:) = 0._wp
         wn_tl(:,:,:) = 0._wp
         rotb_tl(:,:,:) = 0._wp
         rotn_tl(:,:,:) = 0._wp
         hdivb_tl(:,:,:) = 0._wp
         hdivn_tl(:,:,:) = 0._wp
         tsb_tl(:,:,:,:) = 0._wp
         tsn_tl(:,:,:,:) = 0._wp
         tsa_tl(:,:,:,:) = 0._wp
         rn2_tl(:,:,:) = 0._wp
         rhd_tl(:,:,:) = 0._wp
         rn2b_tl(:,:,:) = 0._wp
         rhop_tl(:,:,:) = 0._wp
         sshb_tl(:,:) = 0._wp
         sshn_tl(:,:) = 0._wp
         ssha_tl(:,:) = 0._wp
         sshu_b_tl(:,:) = 0._wp
         sshu_n_tl(:,:) = 0._wp
         sshu_a_tl(:,:) = 0._wp
         sshv_b_tl(:,:) = 0._wp
         sshv_n_tl(:,:) = 0._wp
         sshv_a_tl(:,:) = 0._wp
         sshf_n_tl(:,:) = 0._wp
         spgu_tl(:,:) = 0._wp
         spgv_tl(:,:) = 0._wp
         gtsu_tl(:,:,:) = 0._wp
         gtsv_tl(:,:,:) = 0._wp
         gru_tl(:,:) = 0._wp
         grv_tl(:,:) = 0._wp
#if defined key_zdfddm
         rrau_tl(:,:,:) = 0._wp
#endif
      END IF
      IF ( ( kmode == 0 ) .OR. ( kmode == 2 ) ) THEN
         IF ( .NOT. ll_allocad ) ierr = oce_alloc_tam ( 2 )
         ub_ad(:,:,:) = 0._wp
         vb_ad(:,:,:) = 0._wp
         un_ad(:,:,:) = 0._wp
         vn_ad(:,:,:) = 0._wp
         ua_ad(:,:,:) = 0._wp
         va_ad(:,:,:) = 0._wp
         wn_ad(:,:,:) = 0._wp
         rotb_ad(:,:,:) = 0._wp
         rotn_ad(:,:,:) = 0._wp
         hdivb_ad(:,:,:) = 0._wp
         hdivn_ad(:,:,:) = 0._wp
         tsb_ad(:,:,:,:) = 0._wp
         tsn_ad(:,:,:,:) = 0._wp
         tsa_ad(:,:,:,:) = 0._wp
         rn2_ad(:,:,:) = 0._wp
         rhd_ad(:,:,:) = 0._wp
         rn2b_ad(:,:,:) = 0._wp
         rhop_ad(:,:,:) = 0._wp
         sshb_ad(:,:) = 0._wp
         sshn_ad(:,:) = 0._wp
         ssha_ad(:,:) = 0._wp
         sshu_b_ad(:,:) = 0._wp
         sshu_n_ad(:,:) = 0._wp
         sshu_a_ad(:,:) = 0._wp
         sshv_b_ad(:,:) = 0._wp
         sshv_n_ad(:,:) = 0._wp
         sshv_a_ad(:,:) = 0._wp
         sshf_n_ad(:,:) = 0._wp
         spgu_ad(:,:) = 0._wp
         spgv_ad(:,:) = 0._wp
         gtsu_ad(:,:,:) = 0._wp
         gtsv_ad(:,:,:) = 0._wp
         gru_ad(:,:) = 0._wp
         grv_ad(:,:) = 0._wp
         !
#if defined key_zdfddm
         rrau_ad(:,:,:) = 0._Wp
#endif
      END IF

   END SUBROUTINE oce_tam_init

END MODULE oce_tam
