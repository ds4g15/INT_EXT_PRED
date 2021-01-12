MODULE tamctl
   !!======================================================================
   !!                       ***  MODULE tamctl ***
   !! NEMOTAMR : variables controlling the run.
   !!======================================================================

   !!----------------------------------------------------------------------
   !! History :
   !!        !  ...    ( ... )        Original code: varctl.F90
   !!        !  09-06  (F. Vigilant)  Created to split NEMOVAR / NEMOTAM
   !!---------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !! * Modules used
   USE par_kind

   !! * Routine accessibility

   IMPLICIT NONE

   PUBLIC

   !! * Module variables


   !! namtst:  assimilation test parameters

   INTEGER   :: &
      & ln_swi_opatam      ! Switch for NEMOTAM adjoint tests
                           ! = 0 => Routine by routine adjoint test
                           ! = 1 => Step test
                           ! = 2 => Run TL model
!!! 20191119A - Added switch for passive ocean (i.e. TS cannot impact pressure gradients, but can be modified by mechanics)
   LOGICAL :: ln_tl_passive_ocean
!!!/20191119A

!!! 20191004L initialise TAM from a netCDF file
   CHARACTER(len=64) :: cn_tam_input
!!!/20191004L
!!! 20191004W - incorporate EIV into trajectory velocity fields
   LOGICAL :: ln_tl_eiv
!!! /20191004W

   LOGICAL   :: &
      & ln_tst_tlh        ! Switch for tangent linear hypothesis testing

   LOGICAL   :: &         ! TLH and tan testing settings
      & ln_tlhts,       & ! Switch for perturbing T and S
      & ln_tlhuv,       & ! Switch for perturbing U and V
      & ln_incdx,       & ! Switch for using an increment file for the perturbation
                          ! (rather than the difference between 2 restarts
      & ln_hnorm,       & ! Switch for normalizing the perturbation
      & ln_tlhssh         ! Switch for perturbing SSH

   INTEGER   :: &
      & nn_stage          ! Current stage of the test (deprecated ?)

   REAL(wp)  :: &
      & rn_hstdt,         & ! Upper bound of norm. for T
      & rn_hstds,         & ! Upper bound of norm. for S
      & rn_hstduv,        & ! Upper bound of norm. for U and V
      & rn_hstdssh          ! Upper bound of norm. for SSH

   CHARACTER(len=32) :: &
      & cn_tlhinc_in,     & ! Name the perturbation file
      & cn_tlhrst_in,     & ! Suffix of the perturbed restart file (input)
      & cn_tlhrst_out,    & ! Suffix of the perturbed restart file (output)
      & cn_tlhtrj_out       ! Suffix of the perturbed trajectory file (output)

   !! Units for tangent test

   INTEGER :: &
      & numtan, &       ! Output for tangent diagnostics
      & numtan_sc  	!Output for tangent diagnostics (scalar sampling)

END MODULE tamctl
