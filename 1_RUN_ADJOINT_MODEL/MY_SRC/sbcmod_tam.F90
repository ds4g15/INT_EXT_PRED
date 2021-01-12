!
MODULE sbcmod_tam
#ifdef key_tam
   !!======================================================================
   !!                       ***  MODULE  sbcmod_tam  ***
   !! Surface module :  provide to the ocean its surface boundary condition
   !!                       Tangent&Adjoint module
   !!======================================================================
   !! History of the direct module :
   !!            3.0   !  07-2006  (G. Madec)  Original code
   !!             -    !  08-2008  (S. Masson, E. .... ) coupled interface
   !! History of the T&A module :
   !!            3.0   !  11-2008  (A. Vidard) TAM of the 2006-07 version
   !!            3.2   !  04-2010  (A. Vidard) Nemo3.2 update
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   sbc_init       : read namsbc namelist
   !!   sbc            : surface ocean momentum, heat and freshwater boundary conditions
   !!----------------------------------------------------------------------
   USE par_kind
   USE phycst
   USE lbclnk
   USE lbclnk_tam
   !   USE ice_oce         ! sea-ice model : LIM
   USE dom_oce
   USE sbc_oce
   USE sbcmod
   USE sbc_oce_tam
   USE sbcssm_tam
   USE sbcana_tam
!*B      & sbc_ana_tan,       &
!*B      & sbc_ana_adj,       &
!*B      & sbc_ana_adj_tst
   USE sbcflx_tam
!   USE sbcblk_clio     ! surface boundary condition: bulk formulation : CLIO
!   USE sbcblk_core     ! surface boundary condition: bulk formulation : CORE
!   USE sbcice_if       ! surface boundary condition: ice-if sea-ice model
!   USE sbcice_lim      ! surface boundary condition: LIM 3.0 sea-ice model
!   USE sbcice_lim_2    ! surface boundary condition: LIM 2.0 sea-ice model
!   USE sbccpl          ! surface boundary condition: coupled florulation
   USE sbcssr_tam
   USE closea_tam
!   USE sbcrnf_tam          ! surface boundary condition: runoffs
   USE sbcrnf
   USE sbcfwb_tam
   USE in_out_manager

   IMPLICIT NONE
   PRIVATE

   PUBLIC   sbc_init_tam ! routine called by step_tam.F90
   PUBLIC   sbc_tan      ! routine called by step_tam.F90
   PUBLIC   sbc_adj      ! routine called by step_tam.F90
   PUBLIC   sbc_adj_tst  ! routine called by tst.F90

   INTEGER          ::   nn_ico_cpl  = 0         !: ice-ocean coupling indicator
   !                                             !  = 0   LIM-3 old case
   !                                             !  = 1   stresses computed using now ocean velocity
   !                                             !  = 2   combination of 0 and 1 cases

   INTEGER ::   nsbc   ! type of surface boundary condition (deduced from namsbc informations)
   INTEGER ::   nice   ! type of ice in the surface boundary condition (deduced from namsbc informations)
   LOGICAL, SAVE :: lfirst = .TRUE. ! initialisation flag
   !! * Substitutions
#  include "domzgr_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.0 , LOCEAN-IPSL (2008)
   !! $Id: sbcmod.F90 1172 2008-09-10 15:32:47Z ctlod $
   !! Software governed by the CeCILL licence (modipsl/doc/NEMO_CeCILL.txt)
   !!----------------------------------------------------------------------

CONTAINS

   SUBROUTINE sbc_init_tam
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE sbc_init_tam ***
      !!
      !! ** Purpose of the direct routine:
      !!           Initialisation of the ocean surface boundary computation
      !!
      !! ** Method  :   Read the namsbc namelist and set derived parameters
      !!
      !! ** Action  : - read namsbc parameters
      !!              - nsbc: type of sbc
      !!----------------------------------------------------------------------
      INTEGER ::   icpt      ! temporary integer
      !!
      !!----------------------------------------------------------------------
      IF (lfirst) THEN
         !CALL sbc_init
         IF( nn_ice == 0  )   fr_i_tl(:,:) = 0.e0       ! no ice in the domain, ice fraction is always zero
         lfirst = .FALSE.
      END IF
      !
   END SUBROUTINE sbc_init_tam


   SUBROUTINE sbc_tan( kt )
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE sbc_tan  ***
      !!
      !! ** Purpose of the direct routine:
      !!                provide at each time-step the ocean surface boundary
      !!                condition (momentum, heat and freshwater fluxes)
      !!
      !! ** Method  :   blah blah  to be written ?????????
      !!                CAUTION : never mask the surface stress field (tke sbc)
      !!
      !! ** Action  : - set the ocean surface boundary condition, i.e.
      !!                utau, vtau, qns, qsr, emp, emps, qrp, erp
      !!              - updte the ice fraction : fr_i
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt       ! ocean time step
      !!---------------------------------------------------------------------

      ! ocean to sbc mean sea surface variables (ss._m)
      ! ---------------------------------------
      CALL sbc_ssm_tan( kt )                         ! sea surface mean currents (at U- and V-points),
      !                                          ! temperature and salinity (at T-point) over nf_sbc time-step
      !                                          ! (i.e. sst_m, sss_m, ssu_m, ssv_m)

      ! sbc formulation
      ! ---------------

!!! 20191004O: (SAM) force flux surface-boundary condition, c.f. http://forge.ipsl.jussieu.fr/nemo/attachment/ticket/1738/sbcmod_tam.F90.diff
      nsbc = 2
!!! /20191004O
      SELECT CASE( nsbc )                        ! Compute ocean surface boundary condition
      !                                          ! (i.e. utau,vtau, qns, qsr, emp, emps)
      CASE(  0 )   ;   CALL sbc_gyre_tan    ( kt )      ! analytical formulation : GYRE configuration
         ! no! in default
      !CASE(  1 )   ;   CALL sbc_ana_tan     ( kt )      ! analytical formulation : uniform sbc
      CASE(  2 )   ;   CALL sbc_flx_tan     ( kt )      ! flux formulation
      !CASE(  3 )   ;   CALL sbc_blk_clio_tan( kt )      ! bulk formulation : CLIO for the ocean
      !CASE(  4 )   ;   CALL sbc_blk_core_tan( kt )      ! bulk formulation : CORE for the ocean
      !CASE(  5 )   ;   CALL sbc_cpl_tan     ( kt )      ! coupled formulation
      CASE( 6 )     ;  CALL sbc_sqb_tan     (kt)
      END SELECT

      ! Misc. Options
      ! -------------
      ! not available
!*B      SELECT CASE( nn_ice )                                     ! Update heat and freshwater fluxes over sea-ice areas
!*B      CASE(  1 )   ;       CALL sbc_ice_if_tan   ( kt )                   ! Ice-cover climatology ("Ice-if" model)
!*B         !
!*B      CASE(  2 )   ;       CALL sbc_ice_lim_2_tan( kt, nsbc )             ! LIM 2.0 ice model
!*B         !
!*B      CASE(  3 )   ;       CALL sbc_ice_lim_tan  ( kt, nsbc, nn_ico_cpl)  ! LIM 3.0 ice model
!*B      END SELECT

      ! add runoffs to fresh water fluxes... not needed in tangent

      IF( ln_ssr       )   CALL sbc_ssr_tan( kt )                   ! add SST/SSS damping term

      IF( nn_fwb  /= 0 )   CALL sbc_fwb_tan( kt, nn_fwb, nn_fsbc )  ! control the freshwater budget

      IF( nn_closea == 1 )   CALL sbc_clo_tan( kt )                   ! treatment of closed sea in the model domain
      !                                                         ! (update freshwater fluxes)
      !
!RBbug do not understand why see ticket 667
      CALL lbc_lnk( emp_tl, 'T', 1. )
      !
      !
   END SUBROUTINE sbc_tan

   !!======================================================================
   SUBROUTINE sbc_adj( kt )
      !!---------------------------------------------------------------------
      !!                    ***  ROUTINE sbc_adj  ***
      !!
      !! ** Purpose of the direct routine:
      !!                provide at each time-step the ocean surface boundary
      !!                condition (momentum, heat and freshwater fluxes)
      !!
      !! ** Method  :   blah blah  to be written ?????????
      !!                CAUTION : never mask the surface stress field (tke sbc)
      !!
      !! ** Action  : - set the ocean surface boundary condition, i.e.
      !!                utau, vtau, qns, qsr, emp, emps, qrp, erp
      !!              - updte the ice fraction : fr_i
      !!----------------------------------------------------------------------
      INTEGER, INTENT(in) ::   kt       ! ocean time step
      !!---------------------------------------------------------------------

      ! Misc. Options
      ! -------------
!RBbug do not understand why see ticket 667
      CALL lbc_lnk_adj( emp_ad, 'T', 1. )
      !
      IF( nn_closea == 1 )   CALL sbc_clo_adj( kt )                   ! treatment of closed sea in the model domain
      IF( nn_fwb  /= 0 )   CALL sbc_fwb_adj( kt, nn_fwb, nn_fsbc )  ! control the freshwater budget
      IF( ln_ssr       )   CALL sbc_ssr_adj( kt )                   ! add SST/SSS damping term
      SELECT CASE( nn_ice )                                     ! Update heat and freshwater fluxes over ice-covered areas
!      CASE(  1 )   ;       CALL sbc_ice_if_adj ( kt )                     ! Ice-cover climatology ("Ice-if" model)
         !
!      CASE(  2 )   ;       CALL sbc_ice_lim_2_adj( kt, nsbc )             ! LIM 2.0 ice model
         !
!      CASE(  3 )   ;       CALL sbc_ice_lim_adj  ( kt, nsbc, nn_ico_cpl)  ! LIM 3.0 ice model
      END SELECT
      ! sbc formulation
      ! ---------------


!!! 20191004O: (SAM) force flux surface-boundary condition, c.f. http://forge.ipsl.jussieu.fr/nemo/attachment/ticket/1738/sbcmod_tam.F90.diff
      nsbc = 2
!!! /20191004O
      SELECT CASE( nsbc )                        ! Compute ocean surface boundary condition
      !                                          ! (i.e. utau,vtau, qns, qsr, emp, emps)
      CASE(  0 )   ;   CALL sbc_gyre_adj    ( kt )      ! analytical formulation : GYRE configuration
!      CASE(  1 )   ;   CALL sbc_ana_adj     ( kt )      ! analytical formulation : uniform sbc
      CASE(  2 )   ;   CALL sbc_flx_adj     ( kt )      ! flux formulation
!      CASE(  3 )   ;   CALL sbc_blk_clio_adj( kt )      ! bulk formulation : CLIO for the ocean
!      CASE(  4 )   ;   CALL sbc_blk_core_adj( kt )      ! bulk formulation : CORE for the ocean
!      CASE(  5 )   ;   CALL sbc_cpl_adj     ( kt )      ! coupled formulation
       !CASE(  6 )  ;   CALL sbc_sqb_adj     ( kt )
      END SELECT
      ! ocean to sbc mean sea surface variables (ss._m)
      ! ---------------------------------------
      CALL sbc_ssm_adj( kt )                         ! sea surface mean currents (at U- and V-points),
      !                                          ! temperature and salinity (at T-point) over nf_sbc time-step
      !                                          ! (i.e. sst_m, sss_m, ssu_m, ssv_m)


   END SUBROUTINE sbc_adj
   SUBROUTINE sbc_adj_tst( kumadt )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE sbc_adj_tst ***
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
      !!-----------------------------------------------------------------------
      !! * Modules used

      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt             ! Output unit
      CALL sbc_fwb_adj_tst( kumadt )       ! control the freshwater budget
      CALL sbc_ssr_adj_tst( kumadt )       ! add SST/SSS damping term
!!      CALL sbc_rnf_adj_tst( kumadt )       ! add runoffs to fresh water fluxes
!!      CALL sbc_ice_if_adj_tst( kumadt )    ! Ice-cover climatology ("Ice-if" model)
!!      CALL sbc_ice_lim_2_adj_tst( kumadt ) ! LIM 2.0 ice model
!!      CALL sbc_ice_lim_adj_tst( kumadt )   ! LIM 3.0 ice model
#if defined key_gyre
      CALL sbc_gyre_adj_tst( kumadt )      ! analytical formulation : GYRE configuration
#endif
!!      CALL sbc_ana_adj_tst( kumadt )       ! analytical formulation : uniform sbc
      CALL sbc_flx_adj_tst( kumadt )        ! flux formulation
!      CALL sbc_blk_clio_adj_tst( kumadt )  ! bulk formulation : CLIO for the ocean
!      CALL sbc_blk_core_adj_tst( kumadt )  ! bulk formulation : CORE for the ocean
!      CALL sbc_cpl_adj_tst( kumadt )       ! coupled formulation
      CALL sbc_ssm_adj_tst( kumadt )       ! sea surface mean currents (at U- and V-points),
      IF( nn_closea == 1 ) CALL sbc_clo_adj_tst( kumadt )       ! closed seas,

   END SUBROUTINE sbc_adj_tst
   !!======================================================================
#endif
END MODULE sbcmod_tam
