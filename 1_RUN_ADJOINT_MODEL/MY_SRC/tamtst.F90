MODULE tamtst
#if defined key_tam
   !!======================================================================
   !!                       ***  MODULE tst ***
   !! NEMOVAR : Testing routine
   !!======================================================================

   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !!----------------------------------------------------------------------
   !! * Modules used
   USE par_oce
   USE dom_oce
   USE trabbc
   USE traqsr
   USE trabbl
   USE in_out_manager      ! Input/output
   USE iom                 ! netCDF output/input
   USE paresp              ! Weights for energy-type scalar product
   USE tamctl              ! TAM control
   USE tradmp
   USE sbcmod_tam          ! Tangent/adjoint of surface BCs
   USE eosbn2_tam          ! Tangent/adjoint of eq. of state, Brunt-Vaisala
   USE trasbc_tam          ! Tangent/adjoint of surface BCs application
   USE traqsr_tam          ! Tangent/adjoint of penetrative solar radiation
   USE trabbc_tam          ! Tangent/adjoint of bottom heat flux
   USE trabbl_tam          ! Tangent/adjoint of bottom boundary layer
   USE tradmp_tam          ! Tangent/adjoint of internal damping trends
   USE traadv_tam          ! Tangent/adjoint of horizontal/vertical advection
   USE cla_tam             ! Tangent/adjoint of cross land advection
   USE traldf_tam          ! Tangent/adjoint of lateral mixing
   USE trazdf_tam          ! Tangent/adjoint of vertical diffusion
   USE tranxt_tam          ! Tangent/adjoint of tracers at next time step
   USE zpshde_tam          ! Tangent/adjoint of horiz. derivs. for partial steps
   USE divcur_tam          ! Tangent/adjoint of horiz. div. and rel. vorticity
   USE dynadv_tam          ! Tangent/adjoint of horizontal/vertical advection
   USE dynhpg_tam          ! Tangent/adjoint of horiz. pressure gradient
   USE dynkeg_tam          ! Tangent/adjoint of kinetic energy gradient
   USE dynldf_tam          ! Tangent/adjoint of lateral mixing
   USE dynnxt_tam          ! Tangent/adjoint of dynamics at next time step
   USE dynspg_tam          ! Tangent/adjoint of surface pressure gradient
   USE dynvor_tam          ! Tangent/adjoint of relative and planetary vorticity
   USE dynzdf_tam          ! Tangent/adjoint of vertical diffusion
   USE dynbfr_tam          ! Tangent/adjoint of bottom friction
   USE sshwzv_tam          ! Tangent/adjoint of vertical velocity
   USE step_tam            ! manager of the adjoint ocean time stepping
   USE trj_tam             ! reference trajectory
   USE istate_tam
   USE solsor_tam
   USE closea_tam
   USE zdfbfr_tam
#if defined key_mpp_mpi
   USE lbcnfd_tam
#endif

   IMPLICIT NONE

   !! * Routine accessibility
   PRIVATE

   PUBLIC &
      & tam_tst,  &        !: Scalar product test of the adjoint routines
      & tam_tst_init, &    !: Reading of the namelist
      & numadt             !: File unit number for adjoint test output

   !! * Module variables
   INTEGER :: &
      & numadt             !: File unit number for adjoint test output

CONTAINS

   SUBROUTINE tam_tst
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE tam_tst  ***
      !!
      !! ** Purpose : Apply various tests (linearization, adjoint)
      !!              on the NEMOTAM code.
      !!
      !! ** Method  :
      !!
      !! ** Action  :
      !!
      !! History :
      !!        ! 2007-11 (A. Weaver) original (adjoint tests)
      !!        ! 2009-08 (F. Vigilant) Add tangent tests
      !!-----------------------------------------------------------------------
      !! * Modules used

      !! * Arguments

      !! * Local declarations
      CHARACTER (LEN=128) :: file_out

      ! Open adjoint test output unit

      IF (lwp) THEN

         WRITE(file_out,FMT="('adjoint_test.output_',I4.4)") &
            &   narea-1
         CALL ctl_opn( numadt, file_out, 'UNKNOWN', 'FORMATTED',   &
            &         'SEQUENTIAL', 1, numadt, .FALSE., 1 )

         WRITE(numout,*) ' tstopt: Start testing adjoint operators ...'
         WRITE(numout,*) ' ------'

         WRITE(numout,*)
         WRITE(numout,990) file_out
990      FORMAT('          Output in file = ',A20)
         WRITE(numout,*)

         WRITE(numadt,*)
         WRITE(numadt,997)
         WRITE(numadt,998)
         WRITE(numadt,999)
997      FORMAT('  Routine (L)',2X,' ( L * dx )^T W dy ',2X, &
            &   '     dx^T L^T W dy    ',2X,'Rel.',2X,      &
            &   'Mach.',2X,'Status')
998      FORMAT('             ',2X,'                   ',2X, &
            &   '                      ',2X,'Err.',2x,      &
            &   'Eps. ',2X,'      ')
999      FORMAT('  -----------',2X,'-------------------',2X, &
            &   '----------------------',2X,'----',2X,      &
            &   '-----',2X,'------')
         CALL FLUSH(numout)
         CALL FLUSH(numadt)

      ENDIF

      ! Initialize energy weights
      CALL par_esp

      ! -----------------------------------------------------
      ! Test the adjoint of the components of M (NEMOTAM)
      ! -----------------------------------------------------

      IF ( ln_swi_opatam == 0 ) THEN
         !
         ! *** initialize the reference trajectory
         ! ------------
          CALL trj_rea( nit000 - 1, 1 )
         ! *** Tracers
         ! -----------
         ! *** Surface boundary conditions
         ! ------------
         CALL sbc_adj_tst( numadt )        ! surface boundary conditions
         CALL tra_adv_adj_tst( numadt )    ! Horizontal and vertical advection
         IF (lwp) WRITE(numadt,*)
         IF ( ln_traqsr      )  THEN
            CALL tra_qsr_adj_tst( numadt )    ! Penetrative solar radiation
            IF (lwp) WRITE(numadt,*)
         ENDIF
         CALL tra_ldf_adj_tst( numadt )    ! Lateral mixing
         IF (lwp) WRITE(numadt,*)
         CALL eos_adj_tst( numadt )        ! In situ density
         IF (lwp) WRITE(numadt,*)
         IF ( ln_tradmp      )  THEN
            CALL tra_dmp_adj_tst( numadt )    ! Internal damping trends
            IF (lwp) WRITE(numadt,*)
         ENDIF
# if defined key_trabbl   ||   defined key_esopa
         CALL tra_bbl_adj_tst( numadt )! Bottom boundary layer
         IF (lwp) WRITE(numadt,*)
# endif
         IF ( nn_cla == 1     )  THEN
            CALL cla_traadv_adj_tst( numadt )    ! Cross land advection (update hor. advection)
            IF (lwp) WRITE(numadt,*)
         ENDIF
         CALL tra_zdf_adj_tst( numadt )    ! Vertical mixing
         IF (lwp) WRITE(numadt,*)
         CALL tra_nxt_adj_tst( numadt )    ! Tracer fields at next time step
         IF (lwp) WRITE(numadt,*)
         CALL istate_init_adj_tst( numadt )
         IF (lwp) WRITE(numadt,*)
         CALL tra_sbc_adj_tst( numadt )    ! Surface boundary condition
         IF (lwp) WRITE(numadt,*)
         CALL bn2_adj_tst( numadt )        ! Brunt-Vaisala frequency
         IF (lwp) WRITE(numadt,*)
         !
         !-------------- TESTED IN istate_adj_tst ----------------!
         !CALL zps_hde_adj_tst( numadt )    ! Partial steps: horiz. grad. at bottom level
         !IF (lwp) WRITE(numadt,*)
         !--------------------------------------------------------!
         !
         ! *** Vertical physics
         ! --------------------
         CALL dyn_zdf_adj_tst( numadt )    ! Vertical diffusion
         IF (lwp) WRITE(numadt,*)
         CALL dyn_ldf_adj_tst( numadt )    ! Lateral mixing
         IF (lwp) WRITE(numadt,*)
         CALL dyn_adv_adj_tst( numadt )    ! Advection (vector or flux form)
         IF (lwp) WRITE(numadt,*)
         CALL dyn_hpg_adj_tst( numadt )    ! Horizontal pressure gradient
         IF (lwp) WRITE(numadt,*)
         CALL div_cur_adj_tst( numadt )    ! Horizontal divergence and relative vorticity
         IF (lwp) WRITE(numadt,*)
         CALL ssh_wzv_adj_tst( numadt )    ! Vertical velocity
         IF (lwp) WRITE(numadt,*)
         CALL ssh_nxt_adj_tst( numadt )    ! Sea surface  height time stepping
         IF (lwp) WRITE(numadt,*)
         IF ( nn_cla == 1     )  THEN
            CALL cla_div_adj_tst( numadt )    ! Cross land advection (update hor. divergence)
            IF (lwp) WRITE(numadt,*)
         ENDIF
# if defined key_mpp_mpi
         CALL lbc_nfd_adj_tst( numadt )
         IF (lwp) WRITE(numadt,*)
# endif
         CALL dyn_vor_adj_tst( numadt )    ! Vorticity term including Coriolis
         IF (lwp) WRITE(numadt,*)
         CALL dyn_spg_adj_tst( numadt )    ! Surface pressure gradient
         IF (lwp) WRITE(numadt,*)
         IF ( nn_cla == 1 ) THEN
            CALL cla_dynspg_adj_tst( numadt )    ! Surface pressure gradient
            IF (lwp) WRITE(numadt,*)
         END IF
         CALL dyn_nxt_adj_tst( numadt )    ! Lateral velocity at next time step
         IF (lwp) WRITE(numadt,*)
         CALL dyn_bfr_adj_tst( numadt )    ! Surface pressure gradient
         IF (lwp) WRITE(numadt,*)
         CALL zdf_bfr_adj_tst( numadt )    ! Surface pressure gradient
         IF (lwp) WRITE(numadt,*)
# if defined key_dynspg_flt
         ! *** Red-Black SOR solver
         ! ------------
         CALL sol_sor_adj_tst( numadt )
         IF (lwp) WRITE(numadt,*)
# endif
         CALL flush(numadt)
         IF (lwp) THEN
            WRITE(numout,*)
            WRITE(numout,*) ' tstopt: Finished testing standalone operators'
            WRITE(numout,*) ' ------'
            WRITE(numout,*)
         ENDIF

      ELSEIF (ln_swi_opatam == 1) THEN
         !
         ! *** Time-loop operator
         ! ----------------------
            CALL stp_adj_tst( numadt )        !Time-stepping
            IF (lwp) WRITE(numadt,*)
            CALL flush(numadt)
      ENDIF

      ! Close output file

      IF (lwp) CLOSE(numadt)

      IF (lwp) THEN
         WRITE(numout,*)
         WRITE(numout,*) ' tamtst: Finished testing operators'
         WRITE(numout,*) ' ------'
         WRITE(numout,*)
      ENDIF
      CALL flush(numout)
   END SUBROUTINE tam_tst
   SUBROUTINE tam_tst_init
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE tam_init  ***
      !!
      !! ** Purpose :   read tam related namelists and print the variables.
      !!
      !! ** input   : - namtst_tam namelist
      !!              - namtlh namelist
      !!----------------------------------------------------------------------
      NAMELIST/namtst_tam/ ln_swi_opatam, &
!!! 20191119A - added switch for passive ocean 
                     ln_tl_passive_ocean, &
!!!/20191119A
!!! 20191004W - incorporate EIV into trajectory velocity fields
                     ln_tl_eiv,           &
!!!20191004L initialise TAM from netCDF file
                      cn_tam_input
!!!/20191004L
!!!20191004L initialise TAM from nc file
      cn_tam_input = 'tl_passive_init.nc'
!!!/20191004L
!!! 20191119A - added switch for passive ocean 
      ln_tl_passive_ocean = .false.
!!!/20191119A
!!!20191004W - include EIV into trajectory velocity fields 
      ln_tl_eiv = .false.
!!! .20191004W

      ln_swi_opatam  = 0

      REWIND( numnam )              ! Namelist namrun : parameters of the run
      READ  ( numnam, namtst_tam )
      IF (lwp) THEN                 ! control print
         WRITE(numout,*)
         WRITE(numout,*) 'tam_tst  : Tangent and Adjoint testing'
         WRITE(numout,*) '~~~~~~~'
         WRITE(numout,*) '   Namelist namtst_tam'
         WRITE(numout,*) '      switch for tam testing             ln_swi_opatam  = ', ln_swi_opatam
!!!20191004L initialise TAM from nc file        
	 WRITE(numout,*) '      initial passive tracer distr.      cn_tam_input =', cn_tam_input
!!!/20191004L
!!! 20191004W - incorporate EIV into trajectory velocity
	 WRITE(numout,*) '      add EIV to trajectory velocity     ln_tl_eiv =', ln_tl_eiv
!!! /20191004W

      END IF
   END SUBROUTINE tam_tst_init
#endif
END MODULE tamtst
