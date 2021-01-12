MODULE step_tam
#ifdef key_tam
   !!======================================================================
   !!                       ***  MODULE step_tam  ***
   !! Time-stepping    : manager of the adjoint ocean time stepping
   !!                    Tangent and Adjoint module
   !!======================================================================
   !! History of
   !! the direct:      !  91-03  ()  Original code
   !!                  !  91-11  (G. Madec)
   !!                  !  92-06  (M. Imbard)  add a first output record
   !!                  !  96-04  (G. Madec)  introduction of dynspg
   !!                  !  96-04  (M.A. Foujols)  introduction of passive tracer
   !!             8.0  !  97-06  (G. Madec)  new architecture of call
   !!             8.2  !  97-06  (G. Madec, M. Imbard, G. Roullet)  free surface
   !!             8.2  !  99-02  (G. Madec, N. Grima)  hpg implicit
   !!             8.2  !  00-07  (J-M Molines, M. Imbard)  Open Bondary Conditions
   !!             9.0  !  02-06  (G. Madec)  free form, suppress macro-tasking
   !!             " "  !  04-08  (C. Talandier) New trends organization
   !!             " "  !  05-01  (C. Ethe) Add the KPP closure scheme
   !!             " "  !  05-11  (V. Garnier) Surface pressure gradient organization
   !!             " "  !  05-11  (G. Madec)  Reorganisation of tra and dyn calls
   !!             " "  !  06-01  (L. Debreu, C. Mazauric)  Agrif implementation
   !!             " "  !  06-07  (S. Masson)  restart using iom
   !!    "        " "  !  07-04  (K. Mogensen, A. Weaver, M. Martin) Assimilation interface
   !! History of the TAM
   !!                  !  08-06  (A. Vidard and A. Weaver) Tangent and Adjoint version of 9.0
   !!                  !  08-11  (A. Vidard) Nemo v3 update
   !!                  !  06-09  (F. Vigilant)  Modified to split NEMOVAR / NEMOTAM
   !!                  !  07-12  (P.-A. Bouttier) Phasing with 3.4 version
   !!----------------------------------------------------------------------
   !!   stp_tam        : OPA system time-stepping (tangent linear)
   !!   stp_adj        : OPA system time-stepping (adjoint)
   !!----------------------------------------------------------------------

   USE step_oce_tam
   USE tamtrj, ONLY: nn_ittrjoffset
#if defined key_agrif
#error 'agrif not yet implemented in nemotam'
#endif

   IMPLICIT NONE
   PRIVATE

   PUBLIC stp_tan,      &
      &   stp_adj,      & ! called by simvar.F90
      &   stp_adj_tst

   !! * Substitutions
#  include "domzgr_substitute.h90"
#  include "zdfddm_substitute.h90"

CONTAINS
   SUBROUTINE stp_tan( kstp )
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE stp_tan  ***
      !!
      !! ** Purpose of the direct routine:
      !!              - Time stepping of OPA (momentum and active tracer eqs.)
      !!              - Time stepping of LIM (dynamic and thermodynamic eqs.)
      !!              - Tme stepping  of TRC (passive tracer eqs.)
      !!
      !! ** Method of the direct routine:
      !!              -1- Update forcings and data
      !!              -2- Update ocean physics
      !!              -3- Compute the t and s trends
      !!              -4- Update t and s
      !!              -5- Compute the momentum trends
      !!              -6- Update the horizontal velocity
      !!              -7- Compute the diagnostics variables (rd,N2, div,cur,w)
      !!              -8- Outputs and diagnostics
      !!----------------------------------------------------------------------
      !! * Arguments
      INTEGER, INTENT( in ) :: kstp   ! ocean time-step index

      !! * local declarations
      INTEGER ::   indic    ! error indicator if < 0
      !! ---------------------------------------------------------------------

      indic = 0                    ! reset to no error condition
      IF ( kstp == nit000 )    CALL tl_trj_wri(nit000-1)
      IF ( kstp /= nit000 )    CALL day_tam( kstp, 0 )             ! Calendar (day was already called at nit000 in day_init)

                               CALL iom_setkt( kstp )                          ! say to iom that we are at time step kstp
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Update data, open boundaries, surface boundary condition (including sea-ice)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

      ! update 3D temperature data ... not needed in tangent

      ! update 3D salinity data ... not needed in tangent (to be investigated, see sbc_ssr)
                             CALL sbc_tan ( kstp ) ! Sea boundary condition (including sea-ice)

      ! Output the initial state and forcings ... not needed in tangent

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      !  Ocean dynamics : ssh, wn, hdiv, rot                                 !
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

                             CALL ssh_wzv_tan( kstp )         ! after ssh & vertical velocity

      ! saving direct variables ua,va, ta, sa before entering in tracer

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Ocean physics update                (ua, va, ta, sa used as workspace)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                             CALL bn2_tan( tsb, tsb_tl, rn2b_tl )        ! now Brunt-Vaisala frequency
                             CALL bn2_tan( tsn, tsn_tl, rn2_tl )         ! now Brunt-Vaisala frequency

      !-----------------------------------------------------------------------
      !  VERTICAL PHYSICS
      !-----------------------------------------------------------------------
                             CALL zdf_bfr_tan ( kstp ) ! bottom friction...

      !                                                ! Vertical eddy viscosity and diffusivity coefficients
      ! Richardson number dependent Kz  ... not available
      ! TKE closure scheme for Kz ... not available
      ! KPP closure scheme for Kz ... not available
      ! Constant Kz (reset avt, avm[uv] to the background value)...
      ! increase diffusivity at rivers mouths... not needed in tangent
      ! enhanced vertical eddy diffusivity ... not needed in tangent with lk_zdfcst_tan
      ! double diffusive mixing ... not needed in tangent with lk_zdfcst_tan
      ! mixed layer depth... not needed in tangent with lk_zdfcst_tan
      !-----------------------------------------------------------------------
      !  LATERAL PHYSICS
      !-----------------------------------------------------------------------
      ! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !-----------------------------------------------------------------------
      ! before slope of the lateral mixing... not needed in tangent with lk_zdfcst_tan
#if defined key_traldf_c2d
      ! eddy induced velocity coefficient... not needed in tangent with lk_zdfcst_tan
#endif
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! diagnostics and outputs             (ua, va, ta, sa used as workspace)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      !      not needed in tangent
#if defined key_top
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Passive Tracer Model
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !-----------------------------------------------------------------------

      ! time-stepping... not needed in tangent for the time being

#endif

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Active tracers                              (ua, va used as workspace)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                             tsa_tl(:,:,:,:) = 0.0_wp            ! set tracer trends to zero

      ! apply tracer assimilation increment   ... not needed in tangent
                             CALL tra_sbc_tan( kstp )       ! surface boundary condition
      !IF( ln_traqsr      )   CALL tra_qsr_tan( kstp )       ! penetrative solar radiation qsr
      IF( ln_trabbc      )   CALL tra_bbc_tan( kstp )       ! bottom heat flux
      IF( lk_trabbl  )       CALL tra_bbl_tan( kstp )   ! diffusive bottom boundary layer scheme
      !! advective (and/or diffusive) bottom boundary layer scheme ... currently not available
      IF( ln_tradmp      )   CALL tra_dmp_tan( kstp )       ! internal damping trends
                             CALL tra_adv_tan( kstp )       ! horizontal & vertical advection
                             CALL tra_ldf_tan( kstp )       ! lateral mixing
                             CALL tra_zdf_tan( kstp )       ! vertical mixing

      IF( ln_dynhpg_imp  ) THEN                             ! semi-implicit hpg (time stepping then eos)
      ! update the new (t,s) fields by non
      ! penetrative convective adjustment ... not available
                             CALL tra_nxt_tan( kstp )                              ! tracer fields at next time step
                             CALL eos_tan( tsa, tsa_tl,          &                 ! Time-filtered in situ density used in dynhpg module
                                         & rhd_tl, rhop_tl         )
         IF( ln_zps    )     CALL zps_hde_tan( kstp, jpts, tsa, tsa_tl, rhd_tl,     &
            &                                  gtsu_tl, gru_tl, gtsv_tl,    &   ! Partial steps: time filtered hor. gradient
            &                                  grv_tl )                 ! of t, s, rd at the bottom ocean level
      ELSE                                              ! centered hpg (default case)
                             CALL eos_tan( tsn, tsn_tl, &                 ! now (swap=before) in situ density for dynhpg module
            &                              rhd_tl, rhop_tl )

         IF( ln_zps    )     CALL zps_hde_tan( kstp, jpts, tsn, tsn_tl, rhd_tl,     &
            &                                  gtsu_tl, gru_tl, gtsv_tl,    &   ! Partial steps: time filtered hor. gradient
            &                                  grv_tl )
                             CALL tra_nxt_tan( kstp )                              ! tracer fields at next time step
      ENDIF

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Dynamics                                    (ta, sa used as workspace)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                             ua_tl(:,:,:) = 0.0_wp               ! set dynamics trends to zero
                             va_tl(:,:,:) = 0.0_wp

      ! apply dynamics assimilation increment ... not needed in tangent
                             CALL dyn_adv_tan( kstp )       ! advection (vector or flux form)
                             CALL dyn_vor_tan( kstp )       ! vorticity term including Coriolis
                             CALL dyn_ldf_tan( kstp )       ! lateral mixing
                             CALL dyn_hpg_tan( kstp )           ! horizontal gradient of Hydrostatic pressure
                             CALL dyn_bfr_tan( kstp )           ! bottom friction
                             CALL dyn_zdf_tan( kstp )           ! vertical diffusion
                             CALL dyn_spg_tan( kstp, indic )    ! surface pressure gradient
                             CALL dyn_nxt_tan( kstp )           ! lateral velocity at next time step
                             CALL ssh_nxt_tan( kstp )           ! sea surface height at next time step

      ! vertical mesh at next time step ... not available
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Computation of diagnostic variables
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !-----------------------------------------------------------------------
      CALL tl_trj_wri( kstp )
      CALL trj_rea( kstp, 1) ! ... Read basic state trajectory at end of current step
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Control, and restarts
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !-----------------------------------------------------------------------
      !                                                     ! Time loop: control and print
!*B This fails with cgmod. To be revised      CALL stp_ctl_tan( kstp, indic, 0 )
      IF( indic < 0          )   CALL ctl_stop( 'step_tan: indic < 0' )
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! diagnostics and outputs
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !-----------------------------------------------------------------------
      IF ( nstop == 0 ) THEN                                ! Diagnostics
         ! not needed in tangent
      ENDIF
   END SUBROUTINE stp_tan

   SUBROUTINE stp_adj( kstp )
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE stp_adj  ***
      !!
      !! ** Purpose of the direct routine:
      !!              - Time stepping of OPA (momentum and active tracer eqs.)
      !!              - Time stepping of LIM (dynamic and thermodynamic eqs.)
      !!              - Tme stepping  of TRC (passive tracer eqs.)
      !!
      !! ** Method of the direct routine:
      !!              -1- Update forcings and data
      !!              -2- Update ocean physics
      !!              -3- Compute the t and s trends
      !!              -4- Update t and s
      !!              -5- Compute the momentum trends
      !!              -6- Update the horizontal velocity
      !!              -7- Compute the diagnostics variables (rd,N2, div,cur,w)
      !!              -8- Outputs and diagnostics
      !!----------------------------------------------------------------------
      !! * Arguments
      INTEGER, INTENT( in ) :: kstp   ! ocean time-step index
      !! * local declarations
      INTEGER ::   indic    ! error indicator if < 0
      !! ---------------------------------------------------------------------

      indic = 1                    ! reset to no error condition
                             CALL day_tam( kstp, 1 )            ! Calendar

      ! ... Read basic state trajectory at end of previous step
                             CALL trj_rea( ( kstp - 1 ), -1 )

!!! 20191004H - allow adjoint output writing
                             CALL tl_trj_wri( kstp, 1)
!!! /20191004H

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Dynamics                                    (ta, sa used as workspace)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ! apply dynamics assimilation increment ... not needed in adjoint
                             indic=0
      ! vertical mesh at next time step ... not available

                             CALL ssh_nxt_adj( kstp )         ! sea surface height at next time step
                             CALL dyn_nxt_adj( kstp )           ! lateral velocity at next time step
                             CALL dyn_spg_adj( kstp, indic )    ! surface pressure gradient
                             CALL dyn_zdf_adj( kstp )           ! vertical diffusion
                             CALL dyn_bfr_adj( kstp )           ! bottom friction
                             CALL dyn_hpg_adj( kstp )           ! horizontal gradient of Hydrostatic pressure
                             CALL dyn_ldf_adj( kstp )           ! lateral mixing
                             CALL dyn_vor_adj( kstp )           ! vorticity term including Coriolis
                             CALL dyn_adv_adj( kstp )           ! advection (vector or flux form)

                             ua_ad(:,:,:)  = 0.0_wp                ! set tracer trends to zero
                             va_ad(:,:,:)  = 0.0_wp

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Active tracers                              (ua, va used as workspace)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

      IF( ln_dynhpg_imp  ) THEN                             ! semi-implicit hpg
         IF( ln_zps    )     CALL zps_hde_adj( kstp, jpts, tsa, tsa_ad,      &   ! Partial steps: time filtered hor. gradient
            &                                  rhd_ad, gtsu_ad, gru_ad, gtsv_ad,         &   ! of t, s, rd at the bottom ocean level
            &                                  grv_ad )
                             CALL eos_adj( tsa, tsa_ad, rhd_ad, rhop_ad )                    ! Time-filtered in situ density used in dynhpg module
                             CALL tra_nxt_adj( kstp )                                        ! tracer fields at next time step
      ELSE                                                                                   ! centered hpg (default case)
                             CALL tra_nxt_adj( kstp )                                        ! tracer fields at next time step
         IF( ln_zps    )     CALL zps_hde_adj( kstp, jpts, tsn, tsn_ad,      &   ! Partial steps: time filtered hor. gradient
            &                                  rhd_ad, gtsu_ad, gru_ad, gtsv_ad,         &   ! of t, s, rd at the bottom ocean level
            &                                  grv_ad )
                             CALL eos_adj( tsn, tsn_ad, rhd_ad, rhop_ad )  ! now (swap=before) in situ density for dynhpg module
      ENDIF

      ! update the new (t,s) fields by non
      ! penetrative convective adjustment ... not available

                             CALL tra_zdf_adj( kstp )       ! vertical mixing
                             CALL tra_ldf_adj( kstp )       ! lateral mixing
                             CALL tra_adv_adj( kstp )      ! horizontal & vertical advection
      IF( ln_tradmp      )   CALL tra_dmp_adj( kstp )       ! internal damping trends
      !! advective (and/or diffusive) bottom boundary layer scheme ... currently not available
      IF( lk_trabbl  )       CALL tra_bbl_adj( kstp )   ! diffusive bottom boundary layer scheme
      IF( ln_trabbc      )   CALL tra_bbc_adj( kstp )       ! bottom heat flux

      !IF( ln_traqsr      )   CALL tra_qsr_adj( kstp )       ! penetrative solar radiation qsr

                             CALL tra_sbc_adj( kstp )       ! surface boundary condition

                             tsa_ad(:,:,:,:) = 0.0_wp            ! set tracer trends to zero

      ! apply tracer assimilation increment ... not needed in adjoint
#if defined key_top
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Passive Tracer Model
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !-----------------------------------------------------------------------

      ! time-stepping... not needed in adjoint for the time being

#endif
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Ocean physics update                (ua, va, ta, sa used as workspace)
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      !-----------------------------------------------------------------------
      !  LATERAL PHYSICS
      !-----------------------------------------------------------------------
#if defined key_traldf_c2d
      ! eddy induced velocity coefficient... not needed in tangent with lk_zdfcst_adj
#endif
      ! before slope of the lateral mixing... not needed in adjoint with lk_zdfcst_adj
      !-----------------------------------------------------------------------
      !  VERTICAL PHYSICS
      !-----------------------------------------------------------------------
                             CALL zdf_bfr_adj ( kstp )
      !! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !!-----------------------------------------------------------------------
      !! mixed layer depth... not needed in adjoint with lk_zdfcst_adj
      !! bottom friction... not needed in adjoint with lk_zdfcst_adj
      !! double diffusive mixing ... not needed in adjoint with lk_zdfcst_adj
      !! enhanced vertical eddy diffusivity ... not needed in adjoint with lk_zdfcst_adj
      !! ! increase diffusivity at rivers mouths... not needed in tangent
      !! lk_zdfcst_adj:  Constant Kz read from the reference trajectory
      !! Constant Kz (reset avt, avm[uv] to the background value)... not available
      !! KPP closure scheme for Kz ... not available
      !! TKE closure scheme for Kz ... not available
      !! Richardson number dependent Kz  ... not available
      !!                                                                 ! Vertical eddy viscosity and diffusivity coefficients
                             CALL bn2_adj( tsn, tsn_ad, rn2_ad )         ! now Brunt-Vaisala frequency
                             CALL bn2_adj( tsb, tsb_ad, rn2b_ad )        ! now Brunt-Vaisala frequency
      !!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      !!  Ocean dynamics : ssh, wn, hdiv, rot                                 !
      !!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
                             CALL ssh_wzv_adj( kstp )         ! after ssh & vertical velocity
      !!>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      !! Update data, open boundaries and Forcings
      !!<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      !! Output the initial state and forcings ... not needed in adjoint
                             CALL sbc_adj ( kstp ) ! Sea boundary condition (including sea-ice)
      ! update 3D salinity data ... not needed in tangent
      ! update 3D temperature data ... not needed in adjoint

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Control, and restarts
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !-----------------------------------------------------------------------
      !                                                     ! Time loop: control and print
                             CALL stp_ctl_adj( kstp, indic, 0 )
      IF( indic < 0      )   CALL ctl_stop( 'step_adj: indic < 0' )

      ! close input  ocean restart file ... not needed in adjoint
      ! write output ocean restart file... not needed in adjoint
      ! write open boundary restart file... not needed in adjoint
      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! diagnostics and outputs
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
      ! N.B. ua, va, ta, sa arrays are used as workspace in this section
      !-----------------------------------------------------------------------

      IF ( nstop == 0 ) THEN                                ! Diagnostics
         ! not needed in adjoint
      ENDIF

      !>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
      ! Coupled mode
      !<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

      ! coupled mode : field exchanges ... not available for the next 20 years
      !
      !
      IF( nn_timing == 1 .AND.  kstp == nit000  )   CALL timing_reset
      !
   END SUBROUTINE stp_adj

   SUBROUTINE stp_adj_tst( kumadt )
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
      !!        ! 09-03 (F. Vigilant)
      !!-----------------------------------------------------------------------
      !! * Modules used
      USE sbcssr_tam, ONLY: &
         & qrp_tl, qrp_ad,  &
         & erp_tl, erp_ad
      USE sbcfwb_tam, ONLY: &
         & a_fwb_tl,        & ! for before year.
         & a_fwb_ad  ! for before year.

      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt               ! Output unit

      !! * Local declarations
      INTEGER ::             &
         & ji,               &  ! dummy loop indices
         & jj,               &
         & jk,               &
         & istp
      INTEGER, DIMENSION(jpi,jpj) :: &
         & iseed_2d             ! 2D seed for the random number generator
      REAL(KIND=wp) ::       &
         & zsp1    ,         &  ! scalar product involving the tangent routine
         & zsp1_U  ,         &
         & zsp1_V  ,         &
         & zsp1_T  ,         &
         & zsp1_S  ,         &
         & zsp1_SSH,         &
         & zsp2    ,         &  ! scalar product involving the adjoint routine
         & zsp2_U  ,         &
         & zsp2_V  ,         &
         & zsp2_T  ,         &
         & zsp2_S  ,         &
         & zsp2_SSH
      REAL(KIND=wp), DIMENSION(:,:,:), ALLOCATABLE :: &
         & zun_tlin ,        & ! Tangent input
         & zvn_tlin ,        & ! Tangent input
         & ztn_tlin ,        & ! Tangent input
         & zsn_tlin ,        & ! Tangent input
         & zun_tlout,        & ! Tangent output
         & zvn_tlout,        & ! Tangent output
         & ztn_tlout,        & ! Tangent output
         & zsn_tlout,        & ! Tangent outpu
         & zun_adin ,        & ! Adjoint input
         & zvn_adin ,        & ! Adjoint input
         & ztn_adin ,        & ! Adjoint input
         & zsn_adin ,        & ! Adjoint input
         & zun_adout,        & ! Adjoint output
         & zvn_adout,        & ! Adjoint output
         & ztn_adout,        & ! Adjoint output
         & zsn_adout,        & ! Adjoint output
         & z3r                 ! 3D random field
      REAL(KIND=wp), DIMENSION(:,:), ALLOCATABLE :: &
         & zsshn_tlin  ,     & ! Tangent input
         & zsshn_tlout ,     & ! Tangent input
         & zsshn_adin  ,     & ! Adjoint output
         & zsshn_adout ,     & ! Adjoint output
         & z2r                 ! 2D random field

      CHARACTER(LEN=14) ::   &
         & cl_name

      INTEGER ::             &
         & jpert
      INTEGER, PARAMETER :: jpertmax = 6

      ! Allocate memory

      ALLOCATE( &
         & zun_tlin(   jpi,jpj,jpk), zvn_tlin(   jpi,jpj,jpk), ztn_tlin(  jpi,jpj,jpk), &
         & zsn_tlin(   jpi,jpj,jpk), zsshn_tlin( jpi,jpj    ), zun_tlout( jpi,jpj,jpk), &
         & zvn_tlout(  jpi,jpj,jpk), ztn_tlout(  jpi,jpj,jpk), zsn_tlout( jpi,jpj,jpk), &
         & zsshn_tlout(jpi,jpj    ), zun_adin(   jpi,jpj,jpk), zvn_adin(  jpi,jpj,jpk), &
         & ztn_adin(   jpi,jpj,jpk), zsn_adin(   jpi,jpj,jpk), zsshn_adin(jpi,jpj    ), &
         & zun_adout(  jpi,jpj,jpk), zvn_adout(  jpi,jpj,jpk), ztn_adout( jpi,jpj,jpk), &
         & zsn_adout(  jpi,jpj,jpk), zsshn_adout(jpi,jpj    ), z2r(       jpi,jpj    ), &
         & z3r(        jpi,jpj,jpk)                                                     &
         & )

      !==================================================================
      ! 1) dx = ( un_tl, vn_tl, tn_tl, sn_tl, sshn_tl ) and
      !    dy = ( ..... )
      !==================================================================


      DO jpert = jpertmax, 1, -1
      
         !--------------------------------------------------------------------
         ! Reset the tangent and adjoint variables
         !--------------------------------------------------------------------
         zun_tlin   (:,:,:) = 0.0_wp
         zvn_tlin   (:,:,:) = 0.0_wp
         ztn_tlin   (:,:,:) = 0.0_wp
         zsn_tlin   (:,:,:) = 0.0_wp
         zsshn_tlin (  :,:) = 0.0_wp
         zun_tlout  (:,:,:) = 0.0_wp
         zvn_tlout  (:,:,:) = 0.0_wp
         ztn_tlout  (:,:,:) = 0.0_wp
         zsn_tlout  (:,:,:) = 0.0_wp
         zsshn_tlout(  :,:) = 0.0_wp
         zun_adin   (:,:,:) = 0.0_wp
         zvn_adin   (:,:,:) = 0.0_wp
         ztn_adin   (:,:,:) = 0.0_wp
         zsn_adin   (:,:,:) = 0.0_wp
         zsshn_adin (  :,:) = 0.0_wp
         zun_adout  (:,:,:) = 0.0_wp
         zvn_adout  (:,:,:) = 0.0_wp
         ztn_adout  (:,:,:) = 0.0_wp
         zsn_adout  (:,:,:) = 0.0_wp
         zsshn_adout(  :,:) = 0.0_wp
         z3r        (:,:,:) = 0.0_wp
         z2r        (  :,:) = 0.0_wp

         !--------------------------------------------------------------------
         ! Initialize the tangent input with random noise: dx
         !--------------------------------------------------------------------

         IF ( (jpert == 1) .OR. (jpert == jpertmax) ) THEN
            CALL grid_random(  z3r, 'U', 0.0_wp, stdu )
            DO jk = 1, jpk
               DO jj = nldj, nlej
                  DO ji = nldi, nlei
                     zun_tlin(ji,jj,jk) = z3r(ji,jj,jk)
                  END DO
               END DO
            END DO
         ENDIF
         IF ( (jpert == 2) .OR. (jpert == jpertmax) ) THEN
            CALL grid_random(  z3r, 'V', 0.0_wp, stdv )
            DO jk = 1, jpk
               DO jj = nldj, nlej
                  DO ji = nldi, nlei
                     zvn_tlin(ji,jj,jk) = z3r(ji,jj,jk)
                  END DO
               END DO
            END DO
         ENDIF
         IF ( (jpert == 3) .OR. (jpert == jpertmax) ) THEN
            CALL grid_random(  z3r, 'T', 0.0_wp, stdt )
            DO jk = 1, jpk
               DO jj = nldj, nlej
                  DO ji = nldi, nlei
                     ztn_tlin(ji,jj,jk) = z3r(ji,jj,jk)
                  END DO
               END DO
            END DO
         ENDIF
         IF ( (jpert == 4) .OR. (jpert == jpertmax) ) THEN
            CALL grid_random(  z3r, 'T', 0.0_wp, stds )
            DO jk = 1, jpk
               DO jj = nldj, nlej
                  DO ji = nldi, nlei
                     zsn_tlin(ji,jj,jk) = z3r(ji,jj,jk)
                  END DO
               END DO
            END DO
         ENDIF
         IF ( (jpert == 5) .OR. (jpert == jpertmax) ) THEN
            CALL grid_random(  z2r, 'T', 0.0_wp, stdssh )
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zsshn_tlin(ji,jj) = z2r(ji,jj)
               END DO
            END DO
         ENDIF

         CALL     oce_tam_init( 1 )    ! allocate/initialize tl variables
         CALL sbc_oce_tam_init( 1 )
         CALL sol_oce_tam_init( 1 )
#if defined key_tradmp
         CALL trc_oce_tam_init( 1 )
         strdmp_tl = 0.0_wp
         ttrdmp_tl = 0.0_wp
#endif

            !CALL     oce_tam_init( 1 )    ! allocate/initialize tl variables
            !CALL sbc_oce_tam_init( 1 )
            !CALL sol_oce_tam_init( 1 )
            !CALL trc_oce_tam_init( 1 )

            qrp_tl = 0.0_wp
            erp_tl = 0.0_wp

            emp_tl(:,:) = 0.0_wp
            a_fwb_tl = 0.0_wp

            istp = nit000 - 1
            CALL  trj_rea( istp, 1 )
            !--------------------------------------------------------------------
            ! Initialize the tangent variables: dy^* = W dy
            !--------------------------------------------------------------------

            un_tl  (:,:,:) = zun_tlin   (:,:,:)
            vn_tl  (:,:,:) = zvn_tlin   (:,:,:)
            tsn_tl  (:,:,:,jp_tem) = ztn_tlin   (:,:,:)
            tsn_tl  (:,:,:,jp_sal) = zsn_tlin   (:,:,:)
            sshn_tl(  :,:) = zsshn_tlin (  :,:)

            !CALL     oce_tam_deallocate( 2 )    ! deallocate adj variables
            !CALL sbc_oce_tam_deallocate( 2 )
            !CALL sol_oce_tam_deallocate( 2 )

            !-----------------------------------------------------------------------
            !  Initialization of the dynamics and tracer fields for the tangent
            !-----------------------------------------------------------------------

            CALL istate_init_tan

            DO istp = nit000, nitend, 1
               CALL stp_tan( istp )
            END DO

            zun_tlout  ( :,:,:) = un_tl   (:,:,:)
            zvn_tlout  ( :,:,:) = vn_tl   (:,:,:)
            ztn_tlout  ( :,:,:) = tsn_tl   (:,:,:,jp_tem)
            zsn_tlout  ( :,:,:) = tsn_tl   (:,:,:,jp_sal)
            zsshn_tlout(   :,:) = sshn_tl (  :,:)

            !--------------------------------------------------------------------
            ! Initialize the adjoint variables: dy^* = W dy
            !--------------------------------------------------------------------

            DO jk = 1, jpk
               DO jj = nldj, nlej
                  DO ji = nldi, nlei
                     zun_adin(ji,jj,jk) = zun_tlout(ji,jj,jk) &
                          &               * e1u(ji,jj) * e2u(ji,jj) * fse3u(ji,jj,jk) &
                          &               * umask(ji,jj,jk) * wesp_u
                     zvn_adin(ji,jj,jk) = zvn_tlout(ji,jj,jk) &
                          &               * e1v(ji,jj) * e2v(ji,jj) * fse3v(ji,jj,jk) &
                          &               * vmask(ji,jj,jk) * wesp_u
                     ztn_adin(ji,jj,jk) = ztn_tlout(ji,jj,jk) &
                          &               * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,jk) &
                          &               * tmask(ji,jj,jk) * wesp_t(jk)
                     zsn_adin(ji,jj,jk) = zsn_tlout(ji,jj,jk) &
                          &               * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,jk) &
                          &               * tmask(ji,jj,jk) * wesp_s(jk)
                  END DO
               END DO
            END DO

            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zsshn_adin(ji,jj) = zsshn_tlout(ji,jj) &
                       &               * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,1) &
                       &               * tmask(ji,jj,1) * wesp_ssh
               END DO
            END DO

            !--------------------------------------------------------------------
            ! Compute the scalar product: ( L dx )^T W dy
            !--------------------------------------------------------------------

            zsp1_U    = DOT_PRODUCT( zun_tlout  , zun_adin    )
            zsp1_V    = DOT_PRODUCT( zvn_tlout  , zvn_adin    )
            zsp1_T    = DOT_PRODUCT( ztn_tlout  , ztn_adin    )
            zsp1_S    = DOT_PRODUCT( zsn_tlout  , zsn_adin    )
            zsp1_SSH  = DOT_PRODUCT( zsshn_tlout, zsshn_adin  )

            zsp1      = zsp1_U + zsp1_V + zsp1_T + zsp1_S + zsp1_SSH

            !--------------------------------------------------------------------
            ! Call the adjoint routine: dx^* = L^T dy^*
            !--------------------------------------------------------------------
            CALL     oce_tam_init( 2 )    ! allocate/initialize adj variables
            CALL sbc_oce_tam_init( 2 )
            CALL sol_oce_tam_init( 2 )
#if defined key_tradmp
            CALL trc_oce_tam_init( 2 )
            strdmp_ad = 0.0_wp
            ttrdmp_ad = 0.0_wp
#endif
            qrp_ad = 0.0_wp
            erp_ad = 0.0_wp
            emp_ad(:,:) = 0.0_wp

            a_fwb_ad = 0.0_wp

            un_ad  (:,:,:) = zun_adin   (:,:,:)
            vn_ad  (:,:,:) = zvn_adin   (:,:,:)
            tsn_ad  (:,:,:,jp_tem) = ztn_adin   (:,:,:)
            tsn_ad  (:,:,:,jp_sal) = zsn_adin   (:,:,:)
            sshn_ad(  :,:) = zsshn_adin (  :,:)

            !CALL     oce_tam_deallocate( 1 )    !deallocate tl variables
            !CALL sbc_oce_tam_deallocate( 1 )
            !CALL sol_oce_tam_deallocate( 1 )
            !CALL trc_oce_tam_deallocate( 1 )
            istp = nitend
            CALL  trj_rea( istp, -1 )

            DO istp = nitend, nit000, -1
               CALL stp_adj(istp)
            END DO

            CALL istate_init_adj

            zun_adout  ( :,:,:) = un_ad   (:,:,:)
            zvn_adout  ( :,:,:) = vn_ad   (:,:,:)
            ztn_adout  ( :,:,:) = tsn_ad   (:,:,:,jp_tem)
            zsn_adout  ( :,:,:) = tsn_ad   (:,:,:,jp_sal)
            zsshn_adout(   :,:) = sshn_ad (  :,:)

            zsp2_U    = DOT_PRODUCT( zun_tlin  , zun_adout )
            zsp2_V    = DOT_PRODUCT( zvn_tlin  , zvn_adout )
            zsp2_T    = DOT_PRODUCT( ztn_tlin  , ztn_adout )
            zsp2_S    = DOT_PRODUCT( zsn_tlin  , zsn_adout )
            zsp2_SSH  = DOT_PRODUCT( zsshn_tlin, zsshn_adout )

            zsp2      = zsp2_U + zsp2_V + zsp2_T + zsp2_S + zsp2_SSH

            !CALL     oce_tam_deallocate( 2 )    !deallocate adj variables
            !CALL sbc_oce_tam_deallocate( 2 )
            !CALL sol_oce_tam_deallocate( 2 )
            !CALL trc_oce_tam_deallocate( 2 )

            ! 14 char:'12345678901234'
            SELECT CASE (jpert)
            CASE(1)
               cl_name = 'step_adj     U'
            CASE(2)
               cl_name = 'step_adj     V'
            CASE(3)
               cl_name = 'step_adj     T'
            CASE(4)
               cl_name = 'step_adj     S'
            CASE(5)
               cl_name = 'step_adj   SSH'
            CASE(jpertmax)
               cl_name = 'step_adj      '
            END SELECT
            CALL prntst_adj( cl_name, kumadt, zsp1, zsp2 )
      END DO

      nitsor(:) = jp_it0adj  ! restore nitsor to avoid non reproducible results with or without the tests

      DEALLOCATE(                                 &
         & zun_tlin   , zvn_tlin   , ztn_tlin   , &
         & zsn_tlin   , zsshn_tlin , zun_tlout  , &
         & zvn_tlout  , ztn_tlout  , zsn_tlout  , &
         & zsshn_tlout, zun_adin   , zvn_adin   , &
         & ztn_adin   , zsn_adin   , zsshn_adin , &
         & zun_adout  , zvn_adout  , ztn_adout  , &
         & zsn_adout  , zsshn_adout, z2r        , &
         & z3r                                    &
         & )

   END SUBROUTINE stp_adj_tst

      !!----------------------------------------------------------------------

   !!======================================================================
#endif
END MODULE step_tam
