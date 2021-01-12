!
MODULE nemogcm_tam
#      if defined key_tam
   !!======================================================================
   !!                       ***  MODULE nemogcm   ***
   !! Ocean system   : NEMO GCM (ocean dynamics, on-line tracers, biochemistry and sea-ice)
   !!======================================================================
   !! History :  OPA  ! 1990-10  (C. Levy, G. Madec)  Original code
   !!            7.0  ! 1991-11  (M. Imbard, C. Levy, G. Madec)
   !!            7.1  ! 1993-03  (M. Imbard, C. Levy, G. Madec, O. Marti, M. Guyon, A. Lazar,
   !!                             P. Delecluse, C. Perigaud, G. Caniaux, B. Colot, C. Maes) release 7.1
   !!             -   ! 1992-06  (L.Terray)  coupling implementation
   !!             -   ! 1993-11  (M.A. Filiberti) IGLOO sea-ice
   !!            8.0  ! 1996-03  (M. Imbard, C. Levy, G. Madec, O. Marti, M. Guyon, A. Lazar,
   !!                             P. Delecluse, L.Terray, M.A. Filiberti, J. Vialar, A.M. Treguier, M. Levy) release 8.0
   !!            8.1  ! 1997-06  (M. Imbard, G. Madec)
   !!            8.2  ! 1999-11  (M. Imbard, H. Goosse)  LIM sea-ice model
   !!                 ! 1999-12  (V. Thierry, A-M. Treguier, M. Imbard, M-A. Foujols)  OPEN-MP
   !!                 ! 2000-07  (J-M Molines, M. Imbard)  Open Boundary Conditions  (CLIPPER)
   !!   NEMO     1.0  ! 2002-08  (G. Madec)  F90: Free form and modules
   !!             -   ! 2004-06  (R. Redler, NEC CCRLE, Germany) add OASIS[3/4] coupled interfaces
   !!             -   ! 2004-08  (C. Talandier) New trends organization
   !!             -   ! 2005-06  (C. Ethe) Add the 1D configuration possibility
   !!             -   ! 2005-11  (V. Garnier) Surface pressure gradient organization
   !!             -   ! 2006-03  (L. Debreu, C. Mazauric)  Agrif implementation
   !!             -   ! 2006-04  (G. Madec, R. Benshila)  Step reorganization
   !!             -   ! 2007-07  (J. Chanut, A. Sellar) Unstructured open boundaries (BDY)
   !!            3.2  ! 2009-08  (S. Masson)  open/write in the listing file in mpp
   !!            3.3  ! 2010-05  (K. Mogensen, A. Weaver, M. Martin, D. Lea) Assimilation interface
   !!             -   ! 2010-10  (C. Ethe, G. Madec) reorganisation of initialisation phase
   !!            3.3.1! 2011-01  (A. R. Porter, STFC Daresbury) dynamical allocation
   !!            3.4  ! 2011-11  (C. Harris) decomposition changes for running with CICE
   !! History of TAM:
   !!            3.4  ! 2012-07  (P.-A. Bouttier) Phasing with 3.4 version
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   nemo_gcm       : solve ocean dynamics, tracer, biogeochemistry and/or sea-ice
   !!   nemo_init      : initialization of the NEMO system
   !!   nemo_ctl       : initialisation of the contol print
   !!   nemo_alloc     : dynamical allocation
   !!   nemo_partition : calculate MPP domain decomposition
   !!   factorise      : calculate the factors of the no. of MPI processes
   !!----------------------------------------------------------------------
   USE step_oce        ! module used in the ocean time stepping module
   USE sbc_oce         ! surface boundary condition: ocean
   USE cla             ! cross land advection               (tra_cla routine)
   USE domcfg          ! domain configuration               (dom_cfg routine)
   USE mppini          ! shared/distributed memory setting (mpp_init routine)
   USE domain          ! domain initialization             (dom_init routine)
   USE obcini          ! open boundary cond. initialization (obc_ini routine)
   USE bdyini          ! open boundary cond. initialization (bdy_init routine)
   USE bdydta          ! open boundary cond. initialization (bdy_dta_init routine)
   USE bdytides        ! open boundary cond. initialization (tide_init routine)
   USE istate          ! initial state setting          (istate_init routine)
   USE ldfdyn          ! lateral viscosity setting      (ldfdyn_init routine)
   USE ldftra          ! lateral diffusivity setting    (ldftra_init routine)
   USE zdfini          ! vertical physics setting          (zdf_init routine)
   USE phycst          ! physical constant                  (par_cst routine)
   USE trdmod          ! momentum/tracers trends       (trd_mod_init routine)
   USE diaptr          ! poleward transports           (dia_ptr_init routine)
   USE diadct          ! sections transports           (dia_dct_init routine)
   USE diaobs          ! Observation diagnostics       (dia_obs_init routine)
   USE step            ! NEMO time-stepping                 (stp     routine)
   USE tradmp
   USE trabbl
#if defined key_oasis3
   USE cpl_oasis3      ! OASIS3 coupling
#elif defined key_oasis4
   USE cpl_oasis4      ! OASIS4 coupling (not working)
#endif
   USE c1d             ! 1D configuration
   USE step_c1d        ! Time stepping loop for the 1D configuration
#if defined key_top
   USE trcini          ! passive tracer initialisation
#endif
   USE lib_mpp         ! distributed memory computing
#if defined key_iomput
   USE mod_ioclient
#endif
   USE nemogcm
   USE step_tam
   USE sbcssr_tam
   USE step_oce_tam
   USE zdf_oce_tam
   USE trabbl_tam
   USE tamtst
   USE tamctl
   USE lib_mpp_tam
   USE paresp
   !USE tamtrj
   USE trj_tam
!!! 20191004K - proper initialisation of TAM variables
   USE wrk_nemo ! Memory allocation subroutines
!!! /20191004
   IMPLICIT NONE
   PRIVATE

   PUBLIC   nemo_gcm_tam    ! called by model.F90
   PUBLIC   nemo_init_tam   ! needed by AGRIF

! 2016-06-20: added adjoint time stepping loop
#  include "domzgr_substitute.h90"

   CHARACTER(lc) ::   cform_aaa="( /, 'AAAAAAAA', / ) "     ! flag for output listing

   !!----------------------------------------------------------------------
   !! NEMO/OPA 4.0 , NEMO Consortium (2011)
   !! $Id$
   !! Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
   !!----------------------------------------------------------------------
CONTAINS

   SUBROUTINE nemo_gcm_tam
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE nemo_gcm_tam  ***
      !!
      !! ** Purpose :   NEMO solves the primitive equations on an orthogonal
      !!              curvilinear mesh on the sphere.
      !!
      !! ** Method  : - model general initialization
      !!              - launch the time-stepping (stp routine)
      !!              - finalize the run by closing files and communications
      !!
      !! References : Madec, Delecluse, Imbard, and Levy, 1997:  internal report, IPSL.
      !!              Madec, 2008, internal report, IPSL.
      !!----------------------------------------------------------------------

!!! 20191004K - proper initialisation of TAM variables
      USE sbcssr_tam, ONLY: qrp_tl, erp_tl, qrp_ad, erp_ad
!!! /20191004K

!!! 20191004J - addition of adjoint time-stepping loop
      USE sbcfwb_tam, ONLY: a_fwb_tl, a_fwb_ad
      INTEGER :: jk
!!! /20191004J

      INTEGER ::   istp       ! time step index

!!! 20191004L - ability to read perturbation/cost function from netCDF file
      REAL(KIND=wp), POINTER, DIMENSION(:,:,:) :: ztn_tlin ! temperature
      REAL(KIND=wp), POINTER, DIMENSION(:,:,:) :: zvn_tlin ! meridional velocity
      INTEGER:: ncid
!!! /20191004L

      !!----------------------------------------------------------------------
      !                            !-----------------------!
      !                            !==  Initialisations  ==!
      CALL nemo_init               !-----------------------!
      CALL nemo_init_tam         
      !
      ! check that all process are still there... If some process have an error,
      ! they will never enter in step and other processes will wait until the end of the cpu time!
      IF( lk_mpp )   CALL mpp_max( nstop )

      IF(lwp) WRITE(numout,cform_aaa)   ! Flag AAAAAAA

      !                            !-----------------------!
      !                            !==   time stepping   ==!
      !                            !-----------------------!
      IF (ln_swi_opatam == 2) THEN
!!! 20191004K - proper initialisation of TAM variables
         CALL     oce_tam_init(1)
         CALL sbc_oce_tam_init(1)
         CALL sol_oce_tam_init(1)
#if defined key_tradmp
         CALL trc_oce_tam_init(1)
         strdmp_tl = 0.0_wp
         ttrdmp_tl = 0.0_wp
#endif
         qrp_tl = 0.0_wp
         erp_tl = 0.0_wp   
         emp_tl(:,:) = 0.0_wp
         a_fwb_tl = 0.0_wp      
!!! 20191004L - ability to read perturbation/cost function from netCDF
         ! Variable allocation and initialisation
         CALL wrk_alloc(jpi,jpj,jpk,ztn_tlin) ! T
         ztn_tlin(:,:,:) = 0.0_wp
         CALL wrk_alloc(jpi,jpj,jpk,zvn_tlin) ! V
         zvn_tlin(:,:,:) = 0.0_wp
         
         CALL iom_open(cn_tam_input,ncid,kiolib = jpnf90)
         CALL iom_get(ncid,jpdom_autoglo,"t0_tl",ztn_tlin,0)
         CALL iom_get(ncid,jpdom_autoglo,"v0_tl",zvn_tlin,0) 

#if defined key_mpp_mpi
         CALL lbc_lnk(ztn_tlin(:,:,:), 'T',  1.0_wp)
         CALL lbc_lnk(zvn_tlin(:,:,:), 'V', -1.0_wp) 
#endif
!!! /20191004L
!!! /20191004K

!!!
         ! Initialisation of TL model
         istp = nit000 - 1
         CALL trj_rea( istp, 1)
         istp = nit000
!!! 20191004K - proper initialisation of TAM variables 
         CALL day_tam(nit000, 0)
         un_tl(:,:,:) = 0.0_wp
         sshn_tl(:,:) = 0.0_wp
         tsn_tl(:,:,:,jp_sal) = 0.0_wp
!!! 20191004L - ability to read TAM input from netCDF
         tsn_tl(:,:,:,jp_tem) = ztn_tlin(:,:,:)
         vn_tl(:,:,:) = zvn_tlin(:,:,:)         
         CALL iom_close(ncid)
!!! /20191004L
!!! /20191004K

         CALL istate_init_tan

         ! Time step loop
         DO istp = nit000, nitend, 1
            CALL stp_tan( istp )            
         END DO
!!! 20191004L - ability to read TAM input from netCDF
         ! Variable deallocation
         CALL wrk_dealloc(jpi, jpj, jpk, ztn_tlin)
         CALL wrk_dealloc(jpi, jpj, jpk, zvn_tlin)
!!! /20191004L
         IF (lwp) THEN
            WRITE(numout,*)
            WRITE(numout,*) ' Tangent-linear run complete'
            WRITE(numout,*) ' ---------------------'
            WRITE(numout,*)
         ENDIF
         CALL flush(numout)

!!! 20191004J - addition of adjoint time-stepping loop
      ELSEIF (ln_swi_opatam == 3) THEN

         CALL trj_rea(nit000-1, 1)
         !!! 20191004K - proper initialisation of TAM variables
         DO istp = nit000, nitend
            CALL day_tam(istp, 0)
         END DO
         !!! /20191004K
         CALL trj_rea(istp-1, -1)
         
         !!! 20191004K - proper initialisation of TAM variables
         call oce_tam_init(2)
         call sbc_oce_tam_init(2)
         call sol_oce_tam_init(2)
         call trc_oce_tam_init(2)
#if defined key_tradmp
         strdmp_ad       = 0.0_wp
         ttrdmp_ad       = 0.0_wp
#endif
         qrp_ad          = 0.0_wp
         erp_ad          = 0.0_wp
         emp_ad(:,:)     = 0.0_wp
         a_fwb_ad        = 0.0_wp
         !!! 20191004L - ability to read cost function from nc file
         CALL wrk_alloc(jpi,jpj,jpk,ztn_tlin)
         CALL wrk_alloc(jpi,jpj,jpk,zvn_tlin)  
         ztn_tlin(:,:,:) = 0.0_wp
         zvn_tlin(:,:,:) = 0.0_wp              
         CALL iom_open(cn_tam_input,ncid,kiolib = jpnf90)
         CALL iom_get(ncid,jpdom_autoglo,"t0_ad",ztn_tlin,0) 
         CALL iom_get(ncid,jpdom_autoglo,"v0_ad",zvn_tlin,0) 

#if defined key_mpp_mpi
         CALL lbc_lnk(ztn_tlin(:,:,:), 'T',  1.0_wp)
         CALL lbc_lnk(zvn_tlin(:,:,:), 'V', -1.0_wp) !!!2017-04-05
#endif
         !!! /20191004L
         tsn_ad(:,:,:,:) = 0.0_wp

         un_ad(:,:,:)    = 0.0_wp
         vn_ad(:,:,:)    = 0.0_wp
         sshn_ad(:,:)    = 0.0_wp

         tsb_ad(:,:,:,:) = 0.0_wp
         ub_ad(:,:,:)    = 0.0_wp
         vb_ad(:,:,:)    = 0.0_wp
         sshb_ad(:,:)    = 0.0_wp
         !!! /20191004K

         !!! 20191004L - application of cost function from file
         vn_ad(:,:,:)         = vn_ad(:,:,:)         + ( (vmsk_i)*zvn_tlin(:,:,:) )
         tsn_ad(:,:,:,jp_tem) = tsn_ad(:,:,:,jp_tem) + ( (tmsk_i)*ztn_tlin(:,:,:) )

         DO istp = nitend, nit000, -1
            CALL stp_adj(istp)
            
         END DO

         CALL istate_init_adj
         !!! 20191004H switch to allow adjoint output writing
         CALL tl_trj_wri(nit000-1, 1)
         !!! /20191004H
!!! /20191004J
      ELSE
         CALL tam_tst
      ENDIF
      !                            !------------------------!
      !                            !==  finalize the run  ==!
      !                            !------------------------!
      IF(lwp) WRITE(numout,cform_aaa)   ! Flag AAAAAAA
      !
      IF( nstop /= 0 .AND. lwp ) THEN   ! error print
         WRITE(numout,cform_err)
         WRITE(numout,*) nstop, ' error have been found'
      ENDIF
      !
      IF( nn_timing == 1 )   CALL timing_finalize
      !!
      CALL nemo_closefile
      IF( lk_mpp )   CALL mppstop       ! end mpp communications
      !
   END SUBROUTINE nemo_gcm_tam


   SUBROUTINE nemo_init_tam
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE nemo_init  ***
      !!
      !! ** Purpose :   initialization of the NEMO GCM
      !!----------------------------------------------------------------------
      INTEGER ::   ji            ! dummy loop indices
      INTEGER ::   ilocal_comm   ! local integer
      CHARACTER(len=80), DIMENSION(16) ::   cltxt
      !!
      NAMELIST/namctl/ ln_ctl  , nn_print, nn_ictls, nn_ictle,   &
         &             nn_isplt, nn_jsplt, nn_jctls, nn_jctle,   &
         &             nn_bench, nn_timing
      !!----------------------------------------------------------------------
      !
      IF( ln_rnf        )   CALL sbc_rnf_init
      !!!!!!!!!!!!! TAM initialisation !!!!!!!!!!!!!!!!!!!!!!!!!!!
      CALL nemo_alloc_tam
      CALL nemo_ctl_tam                          ! Control prints & Benchmark

                            CALL  istate_init_tan   ! ocean initial state (Dynamics and tracers)
                            CALL  istate_init_adj   ! ocean initial state (Dynamics and tracers)
      !                                     ! Ocean physics
                            CALL     sbc_init_tam   ! Forcings : surface module
                            CALL     sbc_ssr_ini_tam   ! Forcings : surface module
            !                                     ! Active tracers
                            CALL tra_qsr_init_tam   ! penetrative solar radiation qsr
      IF( lk_trabbl     )   CALL tra_bbl_init_tam   ! advective (and/or diffusive) bottom boundary layer scheme
      IF( ln_tradmp     )   CALL tra_dmp_init_tam   ! internal damping trends
                            CALL tra_adv_init_tam   ! horizontal & vertical advection
                            CALL tra_ldf_init_tam   ! lateral mixing
                            CALL tra_zdf_init_tam   ! vertical mixing and after tracer fields

      !                                     ! Dynamics
                            CALL dyn_adv_init_tam   ! advection (vector or flux form)
                            CALL dyn_vor_init_tam   ! vorticity term including Coriolis
                            CALL dyn_ldf_init_tam   ! lateral mixing
                            CALL dyn_hpg_init_tam   ! horizontal gradient of Hydrostatic pressure
                            CALL dyn_zdf_init_tam   ! vertical diffusion
                            CALL dyn_spg_init_tam   ! surface pressure gradient

      !                                     ! Misc. options
      IF( nn_cla == 1   )   CALL cla_init_tam       ! Cross Land Advection
                            CALL sbc_rnf_init_tam

      CALL tam_tst_init
      CALL tl_trj_ini
   END SUBROUTINE nemo_init_tam

   SUBROUTINE nemo_ctl_tam
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE nemo_ctl  ***
      !!
      !! ** Purpose :   control print setting
      !!
      !! ** Method  : - print namctl information and check some consistencies
      !!----------------------------------------------------------------------
      !
      IF(lwp) THEN                  ! control print
         WRITE(numout,*)
         WRITE(numout,*) 'nemo_ctl_tam: Control prints & Benchmark'
         WRITE(numout,*) '~~~~~~~ '
         WRITE(numout,*) '   Namelist namctl'
         WRITE(numout,*) '      run control (for debugging)     ln_ctl     = ', ln_ctl
         WRITE(numout,*) '      level of print                  nn_print   = ', nn_print
         WRITE(numout,*) '      Start i indice for SUM control  nn_ictls   = ', nn_ictls
         WRITE(numout,*) '      End i indice for SUM control    nn_ictle   = ', nn_ictle
         WRITE(numout,*) '      Start j indice for SUM control  nn_jctls   = ', nn_jctls
         WRITE(numout,*) '      End j indice for SUM control    nn_jctle   = ', nn_jctle
         WRITE(numout,*) '      number of proc. following i     nn_isplt   = ', nn_isplt
         WRITE(numout,*) '      number of proc. following j     nn_jsplt   = ', nn_jsplt
         WRITE(numout,*) '      benchmark parameter (0/1)       nn_bench   = ', nn_bench
      ENDIF
      !
      nprint    = nn_print          ! convert DOCTOR namelist names into OLD names
      nictls    = nn_ictls
      nictle    = nn_ictle
      njctls    = nn_jctls
      njctle    = nn_jctle
      isplt     = nn_isplt
      jsplt     = nn_jsplt
      nbench    = nn_bench
      !                             ! Parameter control
      !
      IF( ln_ctl ) THEN                 ! sub-domain area indices for the control prints
         IF( lk_mpp .AND. jpnij > 1 ) THEN
            isplt = jpni   ;   jsplt = jpnj   ;   ijsplt = jpni*jpnj   ! the domain is forced to the real split domain
         ELSE
            IF( isplt == 1 .AND. jsplt == 1  ) THEN
               CALL ctl_warn( ' - isplt & jsplt are equal to 1',   &
                  &           ' - the print control will be done over the whole domain' )
            ENDIF
            ijsplt = isplt * jsplt            ! total number of processors ijsplt
         ENDIF
         IF(lwp) WRITE(numout,*)'          - The total number of processors over which the'
         IF(lwp) WRITE(numout,*)'            print control will be done is ijsplt : ', ijsplt
         !
         !                              ! indices used for the SUM control
         IF( nictls+nictle+njctls+njctle == 0 )   THEN    ! print control done over the default area
            lsp_area = .FALSE.
         ELSE                                             ! print control done over a specific  area
            lsp_area = .TRUE.
            IF( nictls < 1 .OR. nictls > jpiglo )   THEN
               CALL ctl_warn( '          - nictls must be 1<=nictls>=jpiglo, it is forced to 1' )
               nictls = 1
            ENDIF
            IF( nictle < 1 .OR. nictle > jpiglo )   THEN
               CALL ctl_warn( '          - nictle must be 1<=nictle>=jpiglo, it is forced to jpiglo' )
               nictle = jpiglo
            ENDIF
            IF( njctls < 1 .OR. njctls > jpjglo )   THEN
               CALL ctl_warn( '          - njctls must be 1<=njctls>=jpjglo, it is forced to 1' )
               njctls = 1
            ENDIF
            IF( njctle < 1 .OR. njctle > jpjglo )   THEN
               CALL ctl_warn( '          - njctle must be 1<=njctle>=jpjglo, it is forced to jpjglo' )
               njctle = jpjglo
            ENDIF
         ENDIF
      ENDIF
      !
      IF( nbench == 1 ) THEN              ! Benchmark
         SELECT CASE ( cp_cfg )
         CASE ( 'gyre' )   ;   CALL ctl_warn( ' The Benchmark is activated ' )
         CASE DEFAULT      ;   CALL ctl_stop( ' The Benchmark is based on the GYRE configuration:',   &
            &                                 ' key_gyre must be used or set nbench = 0' )
         END SELECT
      ENDIF
      !
      IF( lk_c1d .AND. .NOT.lk_iomput )   CALL ctl_stop( 'nemo_ctl_tam: The 1D configuration must be used ',   &
         &                                               'with the IOM Input/Output manager. '         ,   &
         &                                               'Compile with key_iomput enabled' )
      !
   END SUBROUTINE nemo_ctl_tam

   SUBROUTINE nemo_alloc_tam
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE nemo_alloc  ***
      !!
      !! ** Purpose :   Allocate all the dynamic arrays of the OPA modules
      !!
      !! ** Method  :
      !!----------------------------------------------------------------------
      !
      INTEGER :: ierr
      !!----------------------------------------------------------------------
      !
      ierr =        oce_alloc_tam       ( 0 )          ! ocean
      ierr = ierr + zdf_oce_alloc_tam   (   )          ! ocean vertical physics
      !
      ierr = ierr + lib_mpp_alloc_adj   (numout)    ! mpp exchanges
      ierr = ierr + trc_oce_alloc_tam   ( 0 )          ! shared TRC / TRA arrays
      ierr = ierr + sbc_oce_alloc_tam   ( 0 )          ! shared TRC / TRA arrays
      ierr = ierr + sol_oce_alloc_tam   ( 0 )          ! shared TRC / TRA arrays
      !
      IF( lk_mpp    )   CALL mpp_sum( ierr )
      IF( ierr /= 0 )   CALL ctl_stop( 'STOP', 'nemo_alloc_tam : unable to allocate standard ocean arrays' )
      !
   END SUBROUTINE nemo_alloc_tam

   SUBROUTINE factorise( kfax, kmaxfax, knfax, kn, kerr )
      !!----------------------------------------------------------------------
      !!                     ***  ROUTINE factorise  ***
      !!
      !! ** Purpose :   return the prime factors of n.
      !!                knfax factors are returned in array kfax which is of
      !!                maximum dimension kmaxfax.
      !! ** Method  :
      !!----------------------------------------------------------------------
      INTEGER                    , INTENT(in   ) ::   kn, kmaxfax
      INTEGER                    , INTENT(  out) ::   kerr, knfax
      INTEGER, DIMENSION(kmaxfax), INTENT(  out) ::   kfax
      !
      INTEGER :: ifac, jl, inu
      INTEGER, PARAMETER :: ntest = 14
      INTEGER :: ilfax(ntest)

      ! lfax contains the set of allowed factors.
      data (ilfax(jl),jl=1,ntest) / 16384, 8192, 4096, 2048, 1024, 512, 256,  &
         &                            128,   64,   32,   16,    8,   4,   2  /
      !!----------------------------------------------------------------------

      ! Clear the error flag and initialise output vars
      kerr = 0
      kfax = 1
      knfax = 0

      ! Find the factors of n.
      IF( kn == 1 )   GOTO 20

      ! nu holds the unfactorised part of the number.
      ! knfax holds the number of factors found.
      ! l points to the allowed factor list.
      ! ifac holds the current factor.

      inu   = kn
      knfax = 0

      DO jl = ntest, 1, -1
         !
         ifac = ilfax(jl)
         IF( ifac > inu )   CYCLE

         ! Test whether the factor will divide.

         IF( MOD(inu,ifac) == 0 ) THEN
            !
            knfax = knfax + 1            ! Add the factor to the list
            IF( knfax > kmaxfax ) THEN
               kerr = 6
               write (*,*) 'FACTOR: insufficient space in factor array ', knfax
               return
            ENDIF
            kfax(knfax) = ifac
            ! Store the other factor that goes with this one
            knfax = knfax + 1
            kfax(knfax) = inu / ifac
            !WRITE (*,*) 'ARPDBG, factors ',knfax-1,' & ',knfax,' are ', kfax(knfax-1),' and ',kfax(knfax)
         ENDIF
         !
      END DO

   20 CONTINUE      ! Label 20 is the exit point from the factor search loop.
      !
   END SUBROUTINE factorise

#if defined key_mpp_mpi
   SUBROUTINE nemo_northcomms
      !!======================================================================
      !!                     ***  ROUTINE  nemo_northcomms  ***
      !! nemo_northcomms    :  Setup for north fold exchanges with explicit peer to peer messaging
      !!=====================================================================
      !!----------------------------------------------------------------------
      !!
      !! ** Purpose :   Initialization of the northern neighbours lists.
      !!----------------------------------------------------------------------
      !!    1.0  ! 2011-10  (A. C. Coward, NOCS & J. Donners, PRACE)
      !!----------------------------------------------------------------------

      INTEGER ::   ji, jj, jk, ij, jtyp    ! dummy loop indices
      INTEGER ::   ijpj                    ! number of rows involved in north-fold exchange
      INTEGER ::   northcomms_alloc        ! allocate return status
      REAL(wp), ALLOCATABLE, DIMENSION ( :,: ) ::   znnbrs     ! workspace
      LOGICAL,  ALLOCATABLE, DIMENSION ( : )   ::   lrankset   ! workspace

      IF(lwp) WRITE(numout,*)
      IF(lwp) WRITE(numout,*) 'nemo_northcomms : Initialization of the northern neighbours lists'
      IF(lwp) WRITE(numout,*) '~~~~~~~~~~'

      !!----------------------------------------------------------------------
      ALLOCATE( znnbrs(jpi,jpj), stat = northcomms_alloc )
      ALLOCATE( lrankset(jpnij), stat = northcomms_alloc )
      IF( northcomms_alloc /= 0 ) THEN
         WRITE(numout,cform_war)
         WRITE(numout,*) 'northcomms_alloc : failed to allocate arrays'
         CALL ctl_stop( 'STOP', 'nemo_northcomms : unable to allocate temporary arrays' )
      ENDIF
      nsndto = 0
      isendto = -1
      ijpj   = 4
      !
      ! This routine has been called because ln_nnogather has been set true ( nammpp )
      ! However, these first few exchanges have to use the mpi_allgather method to
      ! establish the neighbour lists to use in subsequent peer to peer exchanges.
      ! Consequently, set l_north_nogather to be false here and set it true only after
      ! the lists have been established.
      !
      l_north_nogather = .FALSE.
      !
      ! Exchange and store ranks on northern rows

      DO jtyp = 1,4

         lrankset = .FALSE.
         znnbrs = narea
         SELECT CASE (jtyp)
            CASE(1)
               CALL lbc_lnk( znnbrs, 'T', 1. )      ! Type 1: T,W-points
            CASE(2)
               CALL lbc_lnk( znnbrs, 'U', 1. )      ! Type 2: U-point
            CASE(3)
               CALL lbc_lnk( znnbrs, 'V', 1. )      ! Type 3: V-point
            CASE(4)
               CALL lbc_lnk( znnbrs, 'F', 1. )      ! Type 4: F-point
         END SELECT

         IF ( njmppt(narea) .EQ. MAXVAL( njmppt ) ) THEN
            DO jj = nlcj-ijpj+1, nlcj
               ij = jj - nlcj + ijpj
               DO ji = 1,jpi
                  IF ( INT(znnbrs(ji,jj)) .NE. 0 .AND. INT(znnbrs(ji,jj)) .NE. narea ) &
               &     lrankset(INT(znnbrs(ji,jj))) = .true.
               END DO
            END DO

            DO jj = 1,jpnij
               IF ( lrankset(jj) ) THEN
                  nsndto(jtyp) = nsndto(jtyp) + 1
                  IF ( nsndto(jtyp) .GT. jpmaxngh ) THEN
                     CALL ctl_stop( ' Too many neighbours in nemo_northcomms ', &
                  &                 ' jpmaxngh will need to be increased ')
                  ENDIF
                  isendto(nsndto(jtyp),jtyp) = jj-1   ! narea converted to MPI rank
               ENDIF
            END DO
         ENDIF

      END DO

      !
      ! Type 5: I-point
      !
      ! ICE point exchanges may involve some averaging. The neighbours list is
      ! built up using two exchanges to ensure that the whole stencil is covered.
      ! lrankset should not be reset between these 'J' and 'K' point exchanges

      jtyp = 5
      lrankset = .FALSE.
      znnbrs = narea
      CALL lbc_lnk( znnbrs, 'J', 1. ) ! first ice U-V point

      IF ( njmppt(narea) .EQ. MAXVAL( njmppt ) ) THEN
         DO jj = nlcj-ijpj+1, nlcj
            ij = jj - nlcj + ijpj
            DO ji = 1,jpi
               IF ( INT(znnbrs(ji,jj)) .NE. 0 .AND. INT(znnbrs(ji,jj)) .NE. narea ) &
            &     lrankset(INT(znnbrs(ji,jj))) = .true.
         END DO
        END DO
      ENDIF

      znnbrs = narea
      CALL lbc_lnk( znnbrs, 'K', 1. ) ! second ice U-V point

      IF ( njmppt(narea) .EQ. MAXVAL( njmppt )) THEN
         DO jj = nlcj-ijpj+1, nlcj
            ij = jj - nlcj + ijpj
            DO ji = 1,jpi
               IF ( INT(znnbrs(ji,jj)) .NE. 0 .AND.  INT(znnbrs(ji,jj)) .NE. narea ) &
            &       lrankset( INT(znnbrs(ji,jj))) = .true.
            END DO
         END DO

         DO jj = 1,jpnij
            IF ( lrankset(jj) ) THEN
               nsndto(jtyp) = nsndto(jtyp) + 1
               IF ( nsndto(jtyp) .GT. jpmaxngh ) THEN
                  CALL ctl_stop( ' Too many neighbours in nemo_northcomms ', &
               &                 ' jpmaxngh will need to be increased ')
               ENDIF
               isendto(nsndto(jtyp),jtyp) = jj-1   ! narea converted to MPI rank
            ENDIF
         END DO
         !
         ! For northern row areas, set l_north_nogather so that all subsequent exchanges
         ! can use peer to peer communications at the north fold
         !
         l_north_nogather = .TRUE.
         !
      ENDIF
      DEALLOCATE( znnbrs )
      DEALLOCATE( lrankset )

   END SUBROUTINE nemo_northcomms
#else
   SUBROUTINE nemo_northcomms      ! Dummy routine
      WRITE(*,*) 'nemo_northcomms: You should not have seen this print! error?'
   END SUBROUTINE nemo_northcomms
#endif
#else
CONTAINS
   SUBROUTINE nemo_gcm_tam
      WRITE(*,*) 'nemo_gcm_tam: You should not have seen this print! error?'
   END SUBROUTINE nemo_gcm_tam
#endif
   !!======================================================================
END MODULE nemogcm_tam
