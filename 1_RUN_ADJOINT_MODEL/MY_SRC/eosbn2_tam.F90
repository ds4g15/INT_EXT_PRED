MODULE eosbn2_tam
   !!==============================================================================
   !!                       ***  MODULE  eosbn2_tam  ***
   !! Ocean diagnostic variable : equation of state - in situ and potential density
   !!                                               - Brunt-Vaisala frequency
   !!                            Tangent and Adjoint module
   !!==============================================================================
   !! History :  OPA  ! 1989-03  (O. Marti)  Original code
   !!            6.0  ! 1994-07  (G. Madec, M. Imbard)  add bn2
   !!            6.0  ! 1994-08  (G. Madec)  Add Jackett & McDougall eos
   !!            7.0  ! 1996-01  (G. Madec)  statement function for e3
   !!            8.1  ! 1997-07  (G. Madec)  density instead of volumic mass
   !!             -   ! 1999-02  (G. Madec, N. Grima) semi-implicit pressure gradient
   !!            8.2  ! 2001-09  (M. Ben Jelloul)  bugfix on linear eos
   !!   NEMO     1.0  ! 2002-10  (G. Madec)  add eos_init
   !!             -   ! 2002-11  (G. Madec, A. Bozec)  partial step, eos_insitu_2d
   !!             -   ! 2003-08  (G. Madec)  F90, free form
   !!            3.0  ! 2006-08  (G. Madec)  add tfreez function
   !!            3.3  ! 2010-05  (C. Ethe, G. Madec)  merge TRC-TRA
   !!             -   ! 2010-10  (G. Nurser, G. Madec)  add eos_alpbet used in ldfslp
   !!   History of the TAM Module :
   !!             8.2 ! 2005-03  (F. Van den Berghe, A. Weaver, N. Daget)
   !!                                              - eostan.F
   !!             9.0 ! 2007-07  (K. Mogensen) Initial version based on eostan.F
   !!                 ! 2008-07 (A. Vidard) bug fix in computation of prd_tl if neos=1
   !!                 ! 2008-11  (A. Vidard) TAM of the 06-08 version
   !!  NEMO       3.2 ! 2010-04  (F. Vigilant) version 3.2
   !!             3.4 ! 2012-04  (P.-A. Bouttier) version 3.4
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!  Direct subroutines
   !!   eos            : generic interface of the equation of state
   !!   eos_insitu     : Compute the in situ density
   !!   eos_insitu_pot : Compute the insitu and surface referenced potential
   !!                    volumic mass
   !!   eos_insitu_2d  : Compute the in situ density for 2d fields
   !!   eos_bn2        : Compute the Brunt-Vaisala frequency
   !!   eos_alpbet     : calculates the in situ thermal/haline expansion ratio
   !!   tfreez         : Compute the surface freezing temperature
   !!   eos_init       : set eos parameters (namelist)
   !!----------------------------------------------------------------------
   !! * Modules used
#if defined key_zdfddm
   USE oce_tam
#endif
   USE dom_oce
   USE par_kind
   USE par_oce
   USE oce
   USE phycst
   USE in_out_manager
!!!20191119A option to switch off T/S pressure feedbacks in TAM
   USE tamctl 
!!! /20191119A
#if defined key_zdfddm
   USE zdfddm          ! vertical physics: double diffusion
#endif
   USE eosbn2
   USE gridrandom
   USE dotprodfld
   USE tstool_tam
   USE wrk_nemo
   USE timing
   USE lib_mpp
   IMPLICIT NONE
   PRIVATE
   !
   !! * Interface
    INTERFACE eos_tan
       MODULE PROCEDURE eos_insitu_tan, eos_insitu_pot_tan, eos_insitu_2d_tan, &
                        eos_alpbet_tan
    END INTERFACE
    INTERFACE eos_adj
       MODULE PROCEDURE eos_insitu_adj, eos_insitu_pot_adj, eos_insitu_2d_adj, &
                        eos_alpbet_adj
    END INTERFACE
   INTERFACE bn2_tan
      MODULE PROCEDURE eos_bn2_tan
   END INTERFACE
   INTERFACE bn2_adj
      MODULE PROCEDURE eos_bn2_adj
   END INTERFACE
   !
   !! * Routine accessibility
   PUBLIC eos_tan    ! called by step.F90, inidtr.F90, tranpc.F90 and intgrd.F90
   PUBLIC bn2_tan    ! called by step.F90
   PUBLIC eos_adj    ! called by step.F90, inidtr.F90, tranpc.F90 and intgrd.F90
   PUBLIC bn2_adj    ! called by step.F90
#if defined key_tam
   PUBLIC eos_adj_tst
   PUBLIC bn2_adj_tst
#endif
   !! * Substitutions
#  include "domzgr_substitute.h90"
#  include "vectopt_loop_substitute.h90"
   !!----------------------------------------------------------------------
   !! NEMO/OPA 3.3 , NEMO Consortium (2010)
   !! $Id$
   !! Software governed by the CeCILL licence     (NEMOGCM/NEMO_CeCILL.txt)
   !!----------------------------------------------------------------------
CONTAINS
   SUBROUTINE eos_insitu_tan( pts, pts_tl, prd_tl )
      !!-----------------------------------------------------------------------
      !!
      !!        ***  ROUTINE eos_insitu_tan : TL OF ROUTINE   eos_insitu ***
      !!
      !! ** Purpose of direct routine   : Compute the in situ density
      !!       (ratio rho/rau0) from potential temperature and salinity
      !!       using an equation of state defined through the namelist
      !!       parameter nn_eos.
      !!
      !! ** Method of direct routine    : 3 cases:
      !!      nn_eos = 0 : Jackett and McDougall (1994) equation of state.
      !!         the in situ density is computed directly as a function of
      !!         potential temperature relative to the surface (the opa t
      !!         variable), salt and pressure (assuming no pressure variation
      !!         along geopotential surfaces, i.e. the pressure p in decibars
      !!         is approximated by the depth in meters.
      !!              prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0
      !!         with pressure                      p        decibars
      !!              potential temperature         t        deg celsius
      !!              salinity                      s        psu
      !!              reference volumic mass        rau0     kg/m**3
      !!              in situ volumic mass          rho      kg/m**3
      !!              in situ density anomalie      prd      no units
      !!         Check value: rho = 1060.93298 kg/m**3 for p=10000 dbar,
      !!          t = 40 deg celcius, s=40 psu
      !!      nn_eos = 1 : linear equation of state function of temperature only
      !!              prd(t) = 0.0285 - rn_alpha * t
      !!      nn_eos = 2 : linear equation of state function of temperature and
      !!               salinity
      !!              prd(t,s) = rn_beta * s - rn_alpha * tn - 1.
      !!      Note that no boundary condition problem occurs in this routine
      !!      as (ptem,psal) are defined over the whole domain.
      !!
      !! ** Comments on Adjoint Routine :
      !!      Care has been taken to avoid division by zero when computing
      !!      the inverse of the square root of salinity at masked salinity
      !!      points.
      !!
      !! * Arguments
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in   ) ::   pts, &   ! 1 : potential temperature  [Celcius]
      !                                                         ! 2 : salinity               [psu]
      &                                                pts_tl   ! 1 : TL of potential temperature [Celsius]
                                                                ! 2 : TL of salinity [psu]
      REAL(wp), DIMENSION(:,:,:), INTENT( out ) ::   &
         & prd_tl                   ! TL of potential density (surface referenced)
      !! * Local declarations
      INTEGER ::  ji, jj, jk      ! dummy loop indices
      REAL(wp) ::   &             ! temporary scalars
         zt, zs, zh, zsr, zr1, zr2, zr3, zr4, zrhop, ze, zbw,   &
         zb, zd, zc, zaw, za, zb1, za1, zkw, zk0,               &
         zttl, zstl, zhtl, zsrtl, zr1tl, zr2tl, zr3tl,          &
         zr4tl, zrhoptl, zetl, zbwtl,                           &
         zbtl, zdtl, zctl, zawtl, zatl, zb1tl, za1tl,           &
         zkwtl, zk0tl, zpes, zrdc1, zrdc2, zeps,                &
         zmask, zrau0r
      REAL(wp), POINTER, DIMENSION(:,:,:) :: zws
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 ) CALL timing_start('eos_tan')
      !
      CALL wrk_alloc( jpi, jpj, jpk, zws )
      !
      SELECT CASE ( nn_eos )

      CASE ( 0 )               !== Jackett and McDougall (1994) formulation ==!
!!!20191119A
IF ( ln_tl_passive_ocean == .true. ) THEN
prd_tl(:,:,:)=0.0_wp
ELSE
!!!/20191119A
         zrau0r = 1._wp / rau0
#ifdef key_sp
         zeps = 1.e-7_wp
#else
         zeps = 1.e-14_wp
#endif
!CDIR NOVERRCHK
         zws(:,:,:) = SQRT( ABS( pts(:,:,:, jp_sal) ) )
         !
         DO jk = 1, jpkm1
            DO jj = 1, jpj
               DO ji = 1, jpi
                  zt = pts(ji,jj,jk,jp_tem)
                  zs = pts(ji,jj,jk,jp_sal)
                  zh = fsdept(ji,jj,jk)        ! depth
                  zsr= zws(ji,jj,jk)           ! square root salinity
                  ! compute volumic mass pure water at atm pressure
                  zr1= ( ( ( ( 6.536332e-9_wp*zt-1.120083e-6_wp )*zt+1.001685e-4_wp)*zt   &
                     -9.095290e-3_wp )*zt+6.793952e-2_wp )*zt+999.842594_wp
                  ! seawater volumic mass atm pressure
                  zr2= ( ( ( 5.3875e-9_wp*zt-8.2467e-7_wp ) *zt+7.6438e-5_wp ) *zt   &
                     -4.0899e-3_wp ) *zt+0.824493_wp
                  zr3= ( -1.6546e-6_wp*zt+1.0227e-4_wp ) *zt-5.72466e-3_wp
                  zr4= 4.8314e-4_wp
                  !
                  ! potential volumic mass (reference to the surface)
                  zrhop= ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1
                  !
                  ! add the compression terms
                  ze = ( -3.508914e-8_wp*zt-1.248266e-8_wp ) *zt-2.595994e-6_wp
                  zbw= (  1.296821e-6_wp*zt-5.782165e-9_wp ) *zt+1.045941e-4_wp
                  zb = zbw + ze * zs

                  zd = -2.042967e-2_wp
                  zc =   (-7.267926e-5_wp*zt+2.598241e-3_wp ) *zt+0.1571896_wp
                  zaw= ( ( 5.939910e-6_wp*zt+2.512549e-3_wp ) *zt-0.1028859_wp ) *zt - 4.721788_wp
                  za = ( zd*zsr + zc ) *zs + zaw

                  zb1=   (-0.1909078_wp*zt+7.390729_wp ) *zt-55.87545_wp
                  za1= ( ( 2.326469e-3_wp*zt+1.553190_wp)*zt-65.00517_wp ) *zt+1044.077_wp
                  zkw= ( ( (-1.361629e-4_wp*zt-1.852732e-2_wp ) *zt-30.41638_wp ) *zt + 2098.925_wp ) *zt+190925.6_wp
                  zk0= ( zb1*zsr + za1 )*zs + zkw

                  ! Tangent linear part

                  zttl = pts_tl(ji,jj,jk, jp_tem)
                  zstl = pts_tl(ji,jj,jk, jp_sal)

                  zsrtl= ( 1.0 / MAX( 2.*zsr, zeps ) ) &
                     &   * tmask(ji,jj,jk)                * zstl

                  zr1tl= ( ( ( (  5.*6.536332e-9_wp   * zt &
                     &           -4.*1.120083e-6_wp ) * zt &
                     &           +3.*1.001685e-4_wp ) * zt &
                     &           -2.*9.095290e-3_wp ) * zt &
                     &           +   6.793952e-2_wp ) * zttl

                  zr2tl= ( ( (    4.*5.3875e-9_wp   * zt &
                     &           -3.*8.2467e-7_wp ) * zt &
                     &           +2.*7.6438e-5_wp ) * zt &
                     &           -   4.0899e-3_wp ) * zttl

                  zr3tl= (       -2.*1.6546e-6_wp   * zt &
                     &           +   1.0227e-4_wp ) * zttl

                  zrhoptl=                                  zr1tl &
                     &     + zs                           * zr2tl &
                     &     + zsr * zs                     * zr3tl &
                     &     + zr3 * zs                     * zsrtl &
                     &     + (  2. * zr4 * zs + zr2               &
                     &        + zr3 * zsr                         )     * zstl

                  zetl = (       -2.*3.508914e-8_wp   * zt &
                     &           -   1.248266e-8_wp        ) * zttl

                  zbwtl= (        2.*1.296821e-6_wp   * zt &
                     &           -   5.782165e-9_wp        ) * zttl

                  zbtl =                                    zbwtl &
                     &    + zs                            * zetl  &
                  &       + ze                            * zstl

                  zctl = (       -2.*7.267926e-5_wp   * zt &
                     &           +   2.598241e-3_wp        ) * zttl

                  zawtl= ( (      3.*5.939910e-6_wp   * zt &
                     &           +2.*2.512549e-3_wp ) * zt &
                     &           -   0.1028859_wp          ) * zttl

                  zatl =                                    zawtl &
                  &      + zd * zs                        * zsrtl &
                  &      + zs                             * zctl  &
                  &      + ( zd * zsr + zc )              * zstl

                  zb1tl= (       -2.*0.1909078_wp     * zt &
                     &           +   7.390729_wp           ) * zttl

                  za1tl= ( (      3.*2.326469e-3_wp   * zt &
                     &           +2.*1.553190_wp    ) * zt &
                     &           -   65.00517_wp           ) * zttl

                  zkwtl= ( ( (   -4.*1.361629e-4_wp   * zt &
                     &           -3.*1.852732e-2_wp ) * zt &
                     &           -2.*30.41638_wp    ) * zt &
                     &           +   2098.925_wp           ) * zttl

                  zk0tl=                                    zkwtl &
                     &  + zb1 * zs                        * zsrtl &
                     &  + zs  * zsr                       * zb1tl &
                     &  + zs                              * za1tl &
                     &  + ( zb1 * zsr + za1 )             * zstl

                  ! Masked in situ density anomaly

                  zrdc1 = 1.0 / ( zk0 - zh * ( za - zh * zb ) )
                  zrdc2 = 1.0 / ( 1.0 - zh * zrdc1 )

                  prd_tl(ji,jj,jk) = tmask(ji,jj,jk) * zrdc2 * &
                     &               (                              zrhoptl &
                     &                 - zrdc2 * zh * zrdc1**2 * zrhop      &
                     &                   * (                        zk0tl   &
                     &                       - zh * (               zatl    &
                     &                                - zh        * zbtl ) ) )&
                     &               * zrau0r
               END DO
            END DO
         END DO
!!!20191119A
      END IF
!!!/20191119A
         !
      CASE ( 1 )               !== Linear formulation function of temperature only ==!
         DO jk = 1, jpkm1
            prd_tl(:,:,jk) = ( - rn_alpha * pts_tl(:,:,jk,jp_tem) ) * tmask(:,:,jk)
         END DO
         !
      CASE ( 2 )               !== Linear formulation function of temperature and salinity ==!

         !                                                ! ===============
         DO jk = 1, jpkm1
            prd_tl(:,:,jk) = ( rn_beta  * pts_tl(:,:,jk,jp_sal) - rn_alpha * pts_tl(:,:,jk,jp_tem ) ) * tmask(:,:,jk)
         END DO
         !
      END SELECT
      !
      CALL wrk_dealloc( jpi, jpj, jpk, zws )
      !
      IF( nn_timing == 1 ) CALL timing_stop('eos_tan')
      !
   END SUBROUTINE eos_insitu_tan

   SUBROUTINE eos_insitu_pot_tan( pts, pts_tl, prd_tl, prhop_tl)
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE eos_insitu_pot_tan  ***
      !!
      !! ** Purpose or the direct routine:
      !!      Compute the in situ density (ratio rho/rau0) and the
      !!      potential volumic mass (Kg/m3) from potential temperature and
      !!      salinity fields using an equation of state defined through the
      !!     namelist parameter nn_eos.
      !!
      !! ** Method  :
      !!      nn_eos = 0 : Jackett and McDougall (1994) equation of state.
      !!         the in situ density is computed directly as a function of
      !!         potential temperature relative to the surface (the opa t
      !!         variable), salt and pressure (assuming no pressure variation
      !!         along geopotential surfaces, i.e. the pressure p in decibars
      !!         is approximated by the depth in meters.
      !!              prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0
      !!              rhop(t,s)  = rho(t,s,0)
      !!         with pressure                      p        decibars
      !!              potential temperature         t        deg celsius
      !!              salinity                      s        psu
      !!              reference volumic mass        rau0     kg/m**3
      !!              in situ volumic mass          rho      kg/m**3
      !!              in situ density anomalie      prd      no units
      !!
      !!         Check value: rho = 1060.93298 kg/m**3 for p=10000 dbar,
      !!          t = 40 deg celcius, s=40 psu
      !!
      !!      nn_eos = 1 : linear equation of state function of temperature only
      !!              prd(t) = ( rho(t) - rau0 ) / rau0 = 0.028 - rn_alpha * t
      !!              rhop(t,s)  = rho(t,s)
      !!
      !!      nn_eos = 2 : linear equation of state function of temperature and
      !!               salinity
      !!              prd(t,s) = ( rho(t,s) - rau0 ) / rau0
      !!                       = rn_beta * s - rn_alpha * tn - 1.
      !!              rhop(t,s)  = rho(t,s)
      !!      Note that no boundary condition problem occurs in this routine
      !!      as (tn,sn) or (ta,sa) are defined over the whole domain.
      !!
      !! ** Action  : - prd  , the in situ density (no units)
      !!              - prhop, the potential volumic mass (Kg/m3)
      !!
      !! References :
      !!      Jackett, D.R., and T.J. McDougall. J. Atmos. Ocean. Tech., 1994
      !!      Brown, J. A. and K. A. Campana. Mon. Weather Rev., 1978
      !!
      !!----------------------------------------------------------------------
      !! * Arguments
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::   pts, &  ! 1 : potential temperature  [Celcius]
      !                                                                 ! 2 : salinity               [psu]
      &                                                         pts_tl  ! 1 : TL of potential temperature  [Celcius]
      !                                                                 ! 2 : TL of salinity               [psu]
      REAL(wp), DIMENSION(jpi,jpj,jpk     ), INTENT(  out) ::   prd_tl  ! TL of in_situ density [-]
      REAL(wp), DIMENSION(jpi,jpj,jpk     ), INTENT(  out) ::   prhop_tl  ! TL of potential density (surface referenced)
      !! * Local declarations
      INTEGER ::  ji, jj, jk      ! dummy loop indices
      REAL(wp) ::   &             ! temporary scalars
         zt, zs, zh, zsr, zr1, zr2, zr3, zr4, zrhop, ze, zbw,   &
         zb, zd, zc, zaw, za, zb1, za1, zkw, zk0,               &
         zttl, zstl, zhtl, zsrtl, zr1tl, zr2tl, zr3tl,          &
         zr4tl, zrhoptl, zetl, zbwtl,                           &
         zbtl, zdtl, zctl, zawtl, zatl, zb1tl, za1tl,           &
         zkwtl, zk0tl, zpes, zrdc1, zrdc2, zeps,                &
         zmask, zrau0r
      REAL(wp), POINTER, DIMENSION(:,:,:) :: zws
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 ) CALL timing_start('eos-p_tan')
      !
      CALL wrk_alloc( jpi, jpj, jpk, zws )
      !
      SELECT CASE ( nn_eos )
      !
      CASE ( 0 )               !== Jackett and McDougall (1994) formulation ==!
!!!20191119A
IF (ln_tl_passive_ocean == .true.) THEN
prhop_tl(:,:,:)=0.0_wp
prd_tl(:,:,:)=0.0_wp
ELSE
!!!/20191119A
         zrau0r = 1.e0 / rau0
#ifdef key_sp
         zeps = 1.e-7
#else
         zeps = 1.e-14
#endif
!CDIR NOVERRCHK
         zws(:,:,:) = SQRT( ABS( pts(:,:,:, jp_sal) ) )
         !
         DO jk = 1, jpkm1
            DO jj = 1, jpj
               DO ji = 1, jpi
                  zt = pts(ji,jj,jk, jp_tem)
                  zs = pts(ji,jj,jk, jp_sal)
                  zh = fsdept(ji,jj,jk)        ! depth
                  zsr = zws(ji,jj,jk)          ! square root salinity
                  ! compute volumic mass pure water at atm pressure
                  zr1 = ( ( ( ( 6.536332e-9_wp   * zt - 1.120083e-6_wp ) * zt   &
                     &        + 1.001685e-4_wp ) * zt - 9.095290e-3_wp ) * zt   &
                     &        + 6.793952e-2_wp ) * zt + 999.842594_wp
                  ! seawater volumic mass atm pressure
                  zr2 = ( ( ( 5.3875e-9_wp   * zt - 8.2467e-7_wp ) * zt         &
                     &      + 7.6438e-5_wp ) * zt - 4.0899e-3_wp ) * zt         &
                     &      + 0.824493_wp
                  zr3 = ( -1.6546e-6_wp * zt + 1.0227e-4_wp ) * zt - 5.72466e-3_wp
                  zr4 = 4.8314e-4_wp

                  ! potential volumic mass (reference to the surface)
                  zrhop= ( zr4 * zs + zr3 * zsr + zr2 ) * zs + zr1

                  ! add the compression terms
                  ze  = ( -3.508914e-8_wp * zt - 1.248266e-8_wp ) * zt - 2.595994e-6_wp
                  zbw = (  1.296821e-6_wp * zt - 5.782165e-9_wp ) * zt + 1.045941e-4_wp
                  zb  = zbw + ze * zs

                  zd = -2.042967e-2_wp
                  zc =   (-7.267926e-5_wp * zt + 2.598241e-3_wp ) * zt + 0.1571896_wp
                  zaw= ( ( 5.939910e-6_wp * zt + 2.512549e-3_wp ) * zt - 0.1028859_wp ) * zt - 4.721788_wp
                  za = ( zd * zsr + zc ) * zs + zaw

                  zb1 =   (-0.1909078_wp   * zt + 7.390729_wp ) * zt - 55.87545_wp
                  za1 = ( ( 2.326469e-3_wp * zt + 1.553190_wp ) * zt - 65.00517_wp  &
                     &  ) * zt + 1044.077_wp
                  zkw = ( ( (-1.361629e-4_wp * zt - 1.852732e-2_wp ) * zt - 30.41638_wp &
                     &    ) * zt + 2098.925_wp ) * zt + 190925.6_wp
                  zk0 = ( zb1 * zsr + za1 ) * zs + zkw


                  zrdc1 = 1.0 / ( zk0 - zh * ( za - zh * zb ) )
                  zrdc2 = 1.0 / ( 1.0 - zh * zrdc1 )

                  ! Tangent linear part

                  zttl = pts_tl(ji,jj,jk, jp_tem)
                  zstl = pts_tl(ji,jj,jk, jp_sal)

                  zsrtl= ( 1.0 / MAX( 2.*zsr, zeps ) ) &
                     &   * tmask(ji,jj,jk)                * zstl

                  zr1tl= ( ( ( (  5.*6.536332e-9_wp   * zt &
                     &           -4.*1.120083e-6_wp ) * zt &
                     &           +3.*1.001685e-4_wp ) * zt &
                     &           -2.*9.095290e-3_wp ) * zt &
                     &           +   6.793952e-2_wp        ) * zttl

                  zr2tl= ( ( (    4.*5.3875e-9_wp   * zt &
                     &           -3.*8.2467e-7_wp ) * zt &
                     &           +2.*7.6438e-5_wp ) * zt &
                     &           -   4.0899e-3_wp        ) * zttl

                  zr3tl= (       -2.*1.6546e-6_wp   * zt &
                     &           +   1.0227e-4_wp        ) * zttl

                  zrhoptl=                                  zr1tl &
                     &     + zs                           * zr2tl &
                     &     + zsr * zs                     * zr3tl &
                     &     + zr3 * zs                     * zsrtl &
                     &     + (  2. * zr4 * zs + zr2               &
                     &        + zr3 * zsr           )     * zstl

                  prhop_tl(ji,jj,jk) = zrhoptl * tmask(ji,jj,jk)

                  zetl = (       -2.*3.508914e-8_wp   * zt &
                     &           -   1.248266e-8_wp        ) * zttl

                  zbwtl= (        2.*1.296821e-6_wp   * zt &
                     &           -   5.782165e-9_wp        ) * zttl

                  zbtl =                                    zbwtl &
                     &    + zs                            * zetl  &
                  &       + ze                            * zstl

                  zctl = (       -2.*7.267926e-5_wp   * zt &
                     &           +   2.598241e-3_wp        ) * zttl

                  zawtl= ( (      3.*5.939910e-6_wp   * zt &
                     &           +2.*2.512549e-3_wp ) * zt &
                     &           -   0.1028859_wp          ) * zttl

                  zatl =                                    zawtl &
                  &      + zd * zs                        * zsrtl &
                  &      + zs                             * zctl  &
                  &      + ( zd * zsr + zc )              * zstl

                  zb1tl= (       -2.*0.1909078_wp     * zt &
                     &           +   7.390729_wp           ) * zttl

                  za1tl= ( (      3.*2.326469e-3_wp   * zt &
                     &           +2.*1.553190_wp    ) * zt &
                     &           -   65.00517_wp           ) * zttl

                  zkwtl= ( ( (   -4.*1.361629e-4_wp   * zt &
                     &           -3.*1.852732e-2_wp ) * zt &
                     &           -2.*30.41638_wp    ) * zt &
                     &           +   2098.925_wp           ) * zttl

                  zk0tl=                                    zkwtl &
                     &  + zb1 * zs                        * zsrtl &
                     &  + zs  * zsr                       * zb1tl &
                     &  + zs                              * za1tl &
                     &  + ( zb1 * zsr + za1 )             * zstl

                  ! Masked in situ density anomaly

                  prd_tl(ji,jj,jk) = tmask(ji,jj,jk) * zrdc2 * &
                     &               (                              zrhoptl &
                     &                 - zrdc2 * zh * zrdc1**2 * zrhop      &
                     &                   * (                        zk0tl   &
                     &                       - zh * (               zatl    &
                     &                                - zh        * zbtl ) ) )&
                     &               * zrau0r
               END DO
            END DO
         END DO
         !
!!!20191119A
END IF
!!!/20191119A
      CASE ( 1 )                !==  Linear formulation = F( temperature )  ==!
         DO jk = 1, jpkm1
            prd_tl  (:,:,jk) = ( - rn_alpha * pts_tl(:,:,jk, jp_tem) ) * tmask(:,:,jk)
            prhop_tl(:,:,jk) = ( rau0 * prd_tl(:,:,jk)        ) * tmask(:,:,jk)
         END DO
         !
      CASE ( 2 )                !==  Linear formulation = F( temperature , salinity )  ==!
         DO jk = 1, jpkm1
            prd_tl(:,:,jk)   = ( rn_beta  * pts_tl(:,:,jk, jp_sal) - rn_alpha * pts_tl(:,:,jk, jp_tem) ) * tmask(:,:,jk)
            prhop_tl(:,:,jk) = ( rau0 * prd_tl(:,:,jk) )       * tmask(:,:,jk)
         END DO
         !
      END SELECT
      !
      CALL wrk_dealloc( jpi, jpj, jpk, zws )
      !
      IF( nn_timing == 1 ) CALL timing_stop('eos-p_tan')
      !
   END SUBROUTINE eos_insitu_pot_tan
   SUBROUTINE eos_insitu_2d_tan( pts, pdep, pts_tl, prd_tl )
      !!-----------------------------------------------------------------------
      !!
      !!        ***  ROUTINE eos_insitu_2d_tan : TL OF ROUTINE eos_insitu_2d ***
      !!
      !! ** Purpose of direct routine   : Compute the in situ density
      !!       (ratio rho/rau0) from potential temperature and salinity
      !!       using an equation of state defined through the namelist
      !!       parameter nn_eos. * 2D field case
      !!
      !! ** Method of direct routine    : 3 cases:
      !!      nn_eos = 0 : Jackett and McDougall (1994) equation of state.
      !!         the in situ density is computed directly as a function of
      !!         potential temperature relative to the surface (the opa t
      !!         variable), salt and pressure (assuming no pressure variation
      !!         along geopotential surfaces, i.e. the pressure p in decibars
      !!         is approximated by the depth in meters.
      !!              prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0
      !!         with pressure                      p        decibars
      !!              potential temperature         t        deg celsius
      !!              salinity                      s        psu
      !!              reference volumic mass        rau0     kg/m**3
      !!              in situ volumic mass          rho      kg/m**3
      !!              in situ density anomalie      prd      no units
      !!         Check value: rho = 1060.93298 kg/m**3 for p=10000 dbar,
      !!          t = 40 deg celcius, s=40 psu
      !!      nn_eos = 1 : linear equation of state function of temperature only
      !!              prd(t) = 0.0285 - ralpha * t
      !!      nn_eos = 2 : linear equation of state function of temperature and
      !!               salinity
      !!              prd(t,s) = rn_beta * s - rn_alpha * tn - 1.
      !!      Note that no boundary condition problem occurs in this routine
      !!      as (ptem,psal) are defined over the whole domain.
      !!
      !! ** Comments on Adjoint Routine :
      !!      Care has been taken to avoid division by zero when computing
      !!      the inverse of the square root of salinity at masked salinity
      !!      points.
      !!
      !! ** Action  :
      !!
      !! References :
      !!
      !! History :
      !!    8.2 ! 05-03 ((F. Van den Berghe, A. Weaver, N. Daget)  - eostan.F
      !!    9.0 ! 07-07 (K. Mogensen) Initial version based on eostan.F
      !!        ! 08-07 (A. Vidard) bug fix in computation of prd_tl if neos=1
      !!-----------------------------------------------------------------------
      !! * Modules used
      !! * Arguments
      REAL(wp), DIMENSION(jpi,jpj,jpts), INTENT(in   ) ::   pts, &  ! 1 : potential temperature  [Celcius]
      !                                                             ! 2 : salinity               [psu]
      &                                                     pts_tl  ! 1 : TL of potential temperature  [Celcius]
      !                                                             ! 2 : TL of salinity               [psu]
      REAL(wp), DIMENSION(jpi,jpj ), INTENT(  out) ::   prd_tl    ! TL of in_situ density [-]
      REAL(wp), DIMENSION(jpi,jpj)     , INTENT(in   ) ::       pdep  ! depth                  [m]
      !
      !! * Local declarations
      INTEGER ::  ji, jj, jk      ! dummy loop indices
      REAL(wp) ::   &             ! temporary scalars
         zt, zs, zh, zsr, zr1, zr2, zr3, zr4, zrhop, ze, zbw,   &
         zb, zd, zc, zaw, za, zb1, za1, zkw, zk0,               &
         zttl, zstl, zhtl, zsrtl, zr1tl, zr2tl, zr3tl,          &
         zr4tl, zrhoptl, zetl, zbwtl,                           &
         zbtl, zdtl, zctl, zawtl, zatl, zb1tl, za1tl,           &
         zkwtl, zk0tl, zpes, zrdc1, zrdc2, zeps,                &
         zmask
      REAL(wp), POINTER, DIMENSION(:,:) :: zws
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 ) CALL timing_start('eos2d_tan')
      !
      CALL wrk_alloc( jpi, jpj, zws )
      !
      SELECT CASE ( nn_eos )
      !
      CASE ( 0 )               !== Jackett and McDougall (1994) formulation ==!
      !
!!!20191119A
IF (ln_tl_passive_ocean == .true.) THEN
prd_tl(:,:)=0.0_wp
ELSE
!!!/20191119A
#ifdef key_sp
         zeps = 1.e-7
#else
         zeps = 1.e-14
#endif
!CDIR NOVERRCHK
         DO jj = 1, jpjm1
!CDIR NOVERRCHK
            DO ji = 1, fs_jpim1   ! vector opt.
               zws(ji,jj) = SQRT( ABS( pts(ji,jj, jp_sal) ) )
            END DO
         END DO
         DO jj = 1, jpjm1
            DO ji = 1, fs_jpim1   ! vector opt.

               zmask = tmask(ji,jj,1)      ! land/sea bottom mask = surf. mask

               zt = pts (ji,jj, jp_tem)            ! interpolated T
               zs = pts (ji,jj, jp_sal)            ! interpolated S
               zsr= zws(ji,jj)            ! square root of interpolated S
               zh = pdep(ji,jj)            ! depth at the partial step level
                  ! compute volumic mass pure water at atm pressure
                  zr1= ( ( ( ( 6.536332e-9_wp*zt-1.120083e-6_wp )*zt+1.001685e-4_wp)*zt   &
                     & -9.095290e-3_wp )*zt+6.793952e-2_wp )*zt+999.842594_wp
                  ! seawater volumic mass atm pressure
                  zr2= ( ( ( 5.3875e-9_wp*zt-8.2467e-7_wp ) *zt+7.6438e-5_wp ) *zt   &
                     & -4.0899e-3_wp ) *zt+0.824493_wp
                  zr3= ( -1.6546e-6_wp*zt+1.0227e-4_wp ) *zt-5.72466e-3_wp
                  zr4= 4.8314e-4_wp

                  ! potential volumic mass (reference to the surface)
                  zrhop= ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1

                  ! add the compression terms
                  ze = ( -3.508914e-8_wp*zt-1.248266e-8_wp ) *zt-2.595994e-6_wp
                  zbw= (  1.296821e-6*zt-5.782165e-9_wp ) *zt+1.045941e-4_wp
                  zb = zbw + ze * zs

                  zd = -2.042967e-2_wp
                  zc =   (-7.267926e-5_wp*zt+2.598241e-3_wp ) *zt+0.1571896_wp
                  zaw= ( ( 5.939910e-6_wp*zt+2.512549e-3_wp ) *zt-0.1028859_wp ) *zt - 4.721788_wp
                  za = ( zd*zsr + zc ) *zs + zaw

                  zb1=   (-0.1909078_wp*zt+7.390729_wp ) *zt-55.87545_wp
                  za1= ( ( 2.326469e-3_wp*zt+1.553190_wp)*zt-65.00517_wp ) *zt+1044.077_wp
                  zkw= ( ( (-1.361629e-4_wp*zt-1.852732e-2_wp ) *zt-30.41638_wp ) *zt + 2098.925_wp ) *zt+190925.6_wp
                  zk0= ( zb1*zsr + za1 )*zs + zkw

                  ! Tangent linear part

                  zttl = pts_tl(ji,jj, jp_tem)
                  zstl = pts_tl(ji,jj, jp_sal)

                  zsrtl= ( 1.0 / MAX( 2.*zsr, zeps ) ) &
                     &   * tmask(ji,jj,1)                 * zstl

                  zr1tl= ( ( ( (  5.*6.536332e-9_wp   * zt &
                     &           -4.*1.120083e-6_wp ) * zt &
                     &           +3.*1.001685e-4_wp ) * zt &
                     &           -2.*9.095290e-3_wp ) * zt &
                     &           +   6.793952e-2_wp        ) * zttl

                  zr2tl= ( ( (    4.*5.3875e-9_wp   * zt &
                     &           -3.*8.2467e-7_wp ) * zt &
                     &           +2.*7.6438e-5_wp ) * zt &
                     &           -   4.0899e-3_wp        ) * zttl

                  zr3tl= (       -2.*1.6546e-6_wp   * zt &
                     &           +   1.0227e-4_wp        ) * zttl

                  zrhoptl=                                  zr1tl &
                     &     + zs                           * zr2tl &
                     &     + zsr * zs                     * zr3tl &
                     &     + zr3 * zs                     * zsrtl &
                     &     + (  2. * zr4 * zs + zr2              &
                     &        + zr3 * zsr           )     * zstl

                  zetl = (       -2.*3.508914e-8_wp   * zt &
                     &           -   1.248266e-8_wp        ) * zttl

                  zbwtl= (        2.*1.296821e-6_wp   * zt &
                     &           -   5.782165e-9_wp        ) * zttl

                  zbtl =                                    zbwtl &
                     &    + zs                            * zetl  &
                  &       + ze                            * zstl

                  zctl = (       -2.*7.267926e-5_wp   * zt &
                     &           +   2.598241e-3_wp        ) * zttl

                  zawtl= ( (      3.*5.939910e-6_wp   * zt &
                     &           +2.*2.512549e-3_wp ) * zt &
                     &           -   0.1028859_wp          ) * zttl

                  zatl =                                    zawtl &
                  &      + zd * zs                        * zsrtl &
                  &      + zs                             * zctl  &
                  &      + ( zd * zsr + zc )              * zstl

                  zb1tl= (       -2.*0.1909078_wp     * zt &
                     &           +   7.390729_wp           ) * zttl

                  za1tl= ( (      3.*2.326469e-3_wp   * zt &
                     &           +2.*1.553190_wp    ) * zt &
                     &           -   65.00517_wp           ) * zttl

                  zkwtl= ( ( (   -4.*1.361629e-4_wp   * zt &
                     &           -3.*1.852732e-2_wp ) * zt &
                     &           -2.*30.41638_wp    ) * zt &
                     &           +   2098.925_wp           ) * zttl

                  zk0tl=                                    zkwtl &
                     &  + zb1 * zs                        * zsrtl &
                     &  + zs  * zsr                       * zb1tl &
                     &  + zs                              * za1tl &
                     &  + ( zb1 * zsr + za1 )             * zstl

                  ! Masked in situ density anomaly

                  zrdc1 = 1.0 / ( zk0 - zh * ( za - zh * zb ) )
                  zrdc2 = 1.0 / ( 1.0 - zh * zrdc1 )

                  prd_tl(ji,jj) = tmask(ji,jj,1)  * zrdc2 *                &
                     &            (                              zrhoptl   &
                     &              - zrdc2 * zh * zrdc1**2 * zrhop        &
                     &                * (                        zk0tl     &
                     &                    - zh * (               zatl      &
                     &                             - zh        * zbtl ) ) )&
                     &               / rau0


            END DO
         END DO
         !
!!!20191119A
END IF
!!!/20191119A
      CASE ( 1 )                !==  Linear formulation = F( temperature )  ==!
         DO jj = 1, jpjm1
            DO ji = 1, fs_jpim1   ! vector opt.
               prd_tl(ji,jj) = ( - rn_alpha * pts_tl(ji,jj,jp_tem) )     * tmask(ji,jj,1)
            END DO
         END DO
         !
      CASE ( 2 )                !==  Linear formulation = F( temperature , salinity )  ==!
         DO jj = 1, jpjm1
            DO ji = 1, fs_jpim1   ! vector opt.
               prd_tl (ji,jj) = ( rn_beta * pts_tl(ji,jj, jp_sal) - rn_alpha * pts_tl(ji,jj, jp_tem) ) * tmask(ji,jj,1)
            END DO
         END DO
         !
      END SELECT
      CALL wrk_dealloc( jpi, jpj, zws )
      !
      IF( nn_timing == 1 ) CALL timing_stop('eos2d_tan')
   END SUBROUTINE eos_insitu_2d_tan

   SUBROUTINE eos_insitu_adj(pts, pts_ad, prd_ad)
      !!-----------------------------------------------------------------------
      !!
      !!        ***  ROUTINE eos_insitu_tan : Adjoint OF ROUTINE eos_insitu ***
      !!
      !! ** Purpose of direct routine   : Compute the in situ density
      !!       (ratio rho/rau0) from potential temperature and salinity
      !!       using an equation of state defined through the namelist
      !!       parameter nneos.
      !!
      !! ** Method of direct routine    : 3 cases:
      !!      nn_eos = 0 : Jackett and McDougall (1994) equation of state.
      !!         the in situ density is computed directly as a function of
      !!         potential temperature relative to the surface (the opa t
      !!         variable), salt and pressure (assuming no pressure variation
      !!         along geopotential surfaces, i.e. the pressure p in decibars
      !!         is approximated by the depth in meters.
      !!              prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0
      !!         with pressure                      p        decibars
      !!              potential temperature         t        deg celsius
      !!              salinity                      s        psu
      !!              reference volumic mass        rau0     kg/m**3
      !!              in situ volumic mass          rho      kg/m**3
      !!              in situ density anomalie      prd      no units
      !!         Check value: rho = 1060.93298 kg/m**3 for p=10000 dbar,
      !!          t = 40 deg celcius, s=40 psu
      !!      nn_eos = 1 : linear equation of state function of temperature only
      !!              prd(t) = 0.0285 - rn_alpha * t
      !!      nn_eos = 2 : linear equation of state function of temperature and
      !!               salinity
      !!              prd(t,s) = rn_beta * s - rn_alpha * tn - 1.
      !!      Note that no boundary condition problem occurs in this routine
      !!      as (ptem,psal) are defined over the whole domain.
      !!
      !! ** Comments on Adjoint Routine :
      !!      Care has been taken to avoid division by zero when computing
      !!      the inverse of the square root of salinity at masked salinity
      !!      points.
      !!
      !! ** Action  :
      !!
      !! References :
      !!
      !! History :
      !!    8.2 ! 05-03 ((F. Van den Berghe, A. Weaver, N. Daget)  - eostan.F
      !!    9.0 ! 08-08 (A. Vidard) 9.0 version
      !!-----------------------------------------------------------------------
      !! * Modules used
      !! * Arguments
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in   ) ::   pts      ! 1 : potential temperature  [Celcius]
      !                                                         ! 2 : salinity               [psu]
      REAL(wp), DIMENSION(:,:,:,:), INTENT(inout) ::  pts_ad    ! 1 : TL of potential temperature [Celsius]
                                                                ! 2 : TL of salinity [psu]
      REAL(wp), DIMENSION(:,:,:), INTENT( inout ) ::   &
         & prd_ad                   ! TL of potential density (surface referenced)

      !! * Local declarations
      INTEGER ::  ji, jj, jk      ! dummy loop indices
      REAL(wp) ::   &             ! temporary scalars
         zt, zs, zh, zsr, zr1, zr2, zr3, zr4, zrhop, ze, zbw,   &
         zb, zd, zc, zaw, za, zb1, za1, zkw, zk0,               &
         ztad, zsad, zhad, zsrad, zr1ad, zr2ad, zr3ad,          &
         zr4ad, zrhopad, zead, zbwad,                           &
         zbad, zdad, zcad, zawad, zaad, zb1ad, za1ad,           &
         zkwad, zk0ad, zpes, zrdc1, zrdc2, zeps,                &
         zmask, zrau0r
      REAL(wp), POINTER, DIMENSION(:,:,:) :: zws
      !!----------------------------------------------------------------------
      IF( nn_timing == 1 ) CALL timing_start('eos_adj')
      !
      CALL wrk_alloc( jpi, jpj, jpk, zws )
      !
      ! initialization of adjoint variables
      ztad    = 0.0_wp
      zsad    = 0.0_wp
      zhad    = 0.0_wp
      zsrad   = 0.0_wp
      zr1ad   = 0.0_wp
      zr2ad   = 0.0_wp
      zr3ad   = 0.0_wp
      zr4ad   = 0.0_wp
      zrhopad = 0.0_wp
      zead    = 0.0_wp
      zbwad   = 0.0_wp
      zbad    = 0.0_wp
      zdad    = 0.0_wp
      zcad    = 0.0_wp
      zawad   = 0.0_wp
      zaad    = 0.0_wp
      zb1ad   = 0.0_wp
      za1ad   = 0.0_wp
      zkwad   = 0.0_wp
      zk0ad   = 0.0_wp
      SELECT CASE ( nn_eos )

      CASE ( 0 )               !== Jackett and McDougall (1994) formulation ==!
!!!20191119A
IF (ln_tl_passive_ocean==.true.) THEN
prd_ad(:,:,:)=0.0_wp
ELSE
!!!/20191119A
         zrau0r = 1.e0 / rau0
#ifdef key_sp
         zeps = 1.e-7
#else
         zeps = 1.e-14
#endif
!CDIR NOVERRCHK
         zws(:,:,:) = SQRT( ABS( pts(:,:,:, jp_sal) ) )
         DO jk = jpkm1, 1, -1
            DO jj = jpj, 1, -1
               DO ji = jpi, 1, -1
                  zt = pts(ji,jj,jk, jp_tem)
                  zs = pts(ji,jj,jk, jp_sal)
                  zh = fsdept(ji,jj,jk)        ! depth
                  zsr= zws(ji,jj,jk)           ! square root salinity
                  ! compute volumic mass pure water at atm pressure
                  zr1= ( ( ( ( 6.536332e-9_wp*zt-1.120083e-6_wp )*zt+1.001685e-4_wp)*zt   &
                     -9.095290e-3_wp )*zt+6.793952e-2_wp )*zt+999.842594_wp
                  ! seawater volumic mass atm pressure
                  zr2= ( ( ( 5.3875e-9_wp*zt-8.2467e-7_wp ) *zt+7.6438e-5_wp ) *zt   &
                     -4.0899e-3_wp ) *zt+0.824493_wp
                  zr3= ( -1.6546e-6_wp*zt+1.0227e-4_wp ) *zt-5.72466e-3_wp
                  zr4= 4.8314e-4_wp

                  ! potential volumic mass (reference to the surface)
                  zrhop= ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1

                  ! add the compression terms
                  ze = ( -3.508914e-8_wp*zt-1.248266e-8_wp ) *zt-2.595994e-6_wp
                  zbw= (  1.296821e-6_wp*zt-5.782165e-9_wp ) *zt+1.045941e-4_wp
                  zb = zbw + ze * zs

                  zd = -2.042967e-2_wp
                  zc =   (-7.267926e-5_wp*zt+2.598241e-3_wp ) *zt+0.1571896_wp
                  zaw= ( ( 5.939910e-6_wp*zt+2.512549e-3_wp ) *zt-0.1028859_wp ) *zt - 4.721788_wp
                  za = ( zd*zsr + zc ) *zs + zaw

                  zb1=   (-0.1909078_wp*zt+7.390729_wp ) *zt-55.87545_wp
                  za1= ( ( 2.326469e-3_wp*zt+1.553190_wp)*zt-65.00517_wp ) *zt+1044.077_wp
                  zkw= ( ( (-1.361629e-4_wp*zt-1.852732e-2_wp ) *zt-30.41638_wp ) *zt + 2098.925_wp ) *zt+190925.6_wp
                  zk0= ( zb1*zsr + za1 )*zs + zkw

                  zrdc1 = 1.0 / ( zk0 - zh * ( za - zh * zb ) )
                  zrdc2 = 1.0 / ( 1.0 - zh * zrdc1 )
                  ! ============
                  ! Adjoint part
                  ! ============

                  ! Masked in situ density anomaly

                  zrhopad = zrhopad + prd_ad(ji,jj,jk) * tmask(ji,jj,jk)    &
                     &                                 * zrdc2 * zrau0r
                  zk0ad   = zk0ad   - prd_ad(ji,jj,jk) * tmask(ji,jj,jk)    &
                     &                                 * zrdc2 * zrdc2 * zh &
                     &                                 * zrdc1**2 * zrhop   &
                     &                                 * zrau0r
                  zaad    = zaad    + prd_ad(ji,jj,jk) * tmask(ji,jj,jk)    &
                     &                                 * zrdc2 * zrdc2 * zh &
                     &                                 * zrdc1**2 * zrhop   &
                     &                                 * zh * zrau0r
                  zbad    = zbad    - prd_ad(ji,jj,jk) * tmask(ji,jj,jk)    &
                     &                                 * zrdc2 * zrdc2 * zh &
                     &                                 * zrdc1**2 * zrhop   &
                     &                                 * zh * zh * zrau0r
                  prd_ad(ji,jj,jk) = 0.0_wp

                  zkwad = zkwad + zk0ad
                  zsrad = zsrad + zk0ad * zb1 * zs
                  zb1ad = zb1ad + zk0ad * zs  * zsr
                  za1ad = za1ad + zk0ad * zs
                  zsad  = zsad  + zk0ad * ( zb1 * zsr + za1 )
                  zk0ad = 0.0_wp

                  ztad  = ztad + zkwad * ( ( (-4.*1.361629e-4_wp   * zt &
                     &                        -3.*1.852732e-2_wp ) * zt &
                     &                        -2.*30.41638_wp    ) * zt &
                     &                        +   2098.925_wp           )
                  zkwad = 0.0_wp

                  ztad  = ztad + za1ad * ( ( 3.*2.326469e-3_wp   * zt &
                     &                      +2.*1.553190_wp    ) * zt &
                     &                      -   65.00517_wp           )
                  za1ad = 0.0_wp

                  ztad  = ztad + zb1ad * (-2.*0.1909078_wp     * zt &
                     &                    +   7.390729_wp           )
                  zb1ad = 0.0_wp

                  zawad = zawad + zaad
                  zsrad = zsrad + zaad *   zd * zs
                  zcad  = zcad  + zaad *   zs
                  zsad  = zsad  + zaad * ( zd * zsr + zc )
                  zaad  = 0.0_wp

                  ztad  = ztad + zawad * ( ( 3.*5.939910e-6_wp   * zt &
                     &                      +2.*2.512549e-3_wp ) * zt &
                     &                      -   0.1028859_wp          )
                  zawad = 0.0_wp

                  ztad  = ztad + zcad * (-2.*7.267926e-5_wp   * zt &
                     &                   +   2.598241e-3_wp        )
                  zcad  = 0.0_wp

                  zbwad = zbwad + zbad
                  zead  = zead  + zbad * zs
                  zsad  = zsad  + zbad * ze
                  zbad  = 0.0_wp

                  ztad  = ztad + zbwad *  ( 2.*1.296821e-6_wp   * zt &
                     &                     -   5.782165e-9_wp        )
                  zbwad = 0.0_wp

                  ztad  = ztad + zead * (-2.*3.508914e-8_wp   * zt &
                     &                   -   1.248266e-8_wp        )
                  zead =  0.0_wp

                  zr1ad   = zr1ad + zrhopad
                  zr2ad   = zr2ad + zrhopad * zs
                  zr3ad   = zr3ad + zrhopad * zsr * zs
                  zsrad   = zsrad + zrhopad * zr3 * zs
                  zsad    = zsad  + zrhopad * ( 2. * zr4 * zs + zr2  &
                     &                        + zr3 * zsr           )
                  zrhopad = 0.0_wp

                  ztad  = ztad + zr3ad * (-2.*1.6546e-6_wp   * zt &
                     &                    +   1.0227e-4_wp        )
                  zr3ad = 0.0_wp

                  ztad  = ztad + zr2ad * ( ( ( 4.*5.3875e-9_wp   * zt &
                     &                        -3.*8.2467e-7_wp ) * zt &
                     &                        +2.*7.6438e-5_wp ) * zt &
                     &                        -   4.0899e-3_wp        )
                  zr2ad = 0.0_wp

                  ztad  = ztad + zr1ad * ( ( ( ( 5.*6.536332e-9_wp   * zt &
                     &                          -4.*1.120083e-6_wp ) * zt &
                     &                          +3.*1.001685e-4_wp ) * zt &
                     &                          -2.*9.095290e-3_wp ) * zt &
                     &                          +   6.793952e-2_wp        )
                  zr1ad = 0.0_wp

                  zsad  = zsad + zsrad * ( 1.0 / MAX( 2.*zsr, zeps ) ) &
                     &                 * tmask(ji,jj,jk)
                  zsrad = 0.0_wp

                  pts_ad(ji,jj,jk, jp_sal) = pts_ad(ji,jj,jk, jp_sal) + zsad
                  pts_ad(ji,jj,jk,jp_tem) = pts_ad(ji,jj,jk, jp_tem) + ztad
                  ztad = 0.0_wp
                  zsad = 0.0_wp
               END DO
            END DO
         END DO
         !
!!!20191119A
END IF
!!!/20191119A
      CASE ( 1 )               !== Linear formulation function of temperature only ==!
         DO jk = jpkm1, 1, -1
            pts_ad(:,:,jk,jp_tem) = pts_ad(:,:,jk,jp_tem) - rn_alpha * prd_ad(:,:,jk) * tmask(:,:,jk)
            prd_ad(:,:,jk) = 0.0_wp
         END DO
         !
      CASE ( 2 )               !== Linear formulation function of temperature and salinity ==!
         DO jk = jpkm1, 1, -1
            pts_ad(:,:,jk,jp_tem) = pts_ad(:,:,jk,jp_tem) - rn_alpha * prd_ad(:,:,jk) * tmask(:,:,jk)
            pts_ad(:,:,jk,jp_sal) = pts_ad(:,:,jk,jp_sal) + rn_beta * prd_ad( :,:,jk) * tmask(:,:,jk)
            prd_ad( :,:,jk) = 0.0_wp
         END DO
         !
      END SELECT
      !
      CALL wrk_dealloc( jpi, jpj, jpk, zws )
      !
      IF( nn_timing == 1 ) CALL timing_stop('eos_adj')
      !


   END SUBROUTINE eos_insitu_adj

   SUBROUTINE eos_insitu_pot_adj ( pts, pts_ad, prd_ad, prhop_ad )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE eos_insitu_pot_adj  ***
      !!
      !! ** Purpose or the direct routine:
      !!      Compute the in situ density (ratio rho/rau0) and the
      !!      potential volumic mass (Kg/m3) from potential temperature and
      !!      salinity fields using an equation of state defined through the
      !!     namelist parameter nn_eos.
      !!
      !! ** Method  :
      !!      nn_eos = 0 : Jackett and McDougall (1994) equation of state.
      !!         the in situ density is computed directly as a function of
      !!         potential temperature relative to the surface (the opa t
      !!         variable), salt and pressure (assuming no pressure variation
      !!         along geopotential surfaces, i.e. the pressure p in decibars
      !!         is approximated by the depth in meters.
      !!              prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0
      !!              rhop(t,s)  = rho(t,s,0)
      !!         with pressure                      p        decibars
      !!              potential temperature         t        deg celsius
      !!              salinity                      s        psu
      !!              reference volumic mass        rau0     kg/m**3
      !!              in situ volumic mass          rho      kg/m**3
      !!              in situ density anomalie      prd      no units
      !!
      !!         Check value: rho = 1060.93298 kg/m**3 for p=10000 dbar,
      !!          t = 40 deg celcius, s=40 psu
      !!
      !!      neos = 1 : linear equation of state function of temperature only
      !!              prd(t) = ( rho(t) - rau0 ) / rau0 = 0.028 - ralpha * t
      !!              rhop(t,s)  = rho(t,s)
      !!
      !!      nn_eos = 2 : linear equation of state function of temperature and
      !!               salinity
      !!              prd(t,s) = ( rho(t,s) - rau0 ) / rau0
      !!                       = rn_beta * s - rn_alpha * tn - 1.
      !!              rhop(t,s)  = rho(t,s)
      !!      Note that no boundary condition problem occurs in this routine
      !!      as (tn,sn) or (ta,sa) are defined over the whole domain.
      !!
      !! ** Action  : - prd  , the in situ density (no units)
      !!              - prhop, the potential volumic mass (Kg/m3)
      !!
      !! References :
      !!      Jackett, D.R., and T.J. McDougall. J. Atmos. Ocean. Tech., 1994
      !!      Brown, J. A. and K. A. Campana. Mon. Weather Rev., 1978
      !!
      !! History of the adjoint routine:
      !!   9.0  !  08-06  (A. Vidard) Initial version
      !!----------------------------------------------------------------------
      !! * Arguments

      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::   pts ! 1 : potential temperature/salinity [Celcius/psu]
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(inout) ::   pts_ad ! 1 : potential temperature/salinity [Celcius/psu]
      REAL(wp), DIMENSION(jpi,jpj,jpk     ), INTENT(inout) ::   prd_ad      ! TL of in_situ density [-]
      REAL(wp), DIMENSION(jpi,jpj,jpk     ), INTENT(inout) ::   prhop_ad    ! TL of potential density (surface referenced)
      !! * Local declarations
      INTEGER ::  ji, jj, jk      ! dummy loop indices
      REAL(wp) ::   &             ! temporary scalars
         zt, zs, zh, zsr, zr1, zr2, zr3, zr4, zrhop, ze, zbw,   &
         zb, zd, zc, zaw, za, zb1, za1, zkw, zk0,               &
         ztad, zsad, zhad, zsrad, zr1ad, zr2ad, zr3ad,          &
         zr4ad, zrhopad, zead, zbwad,                           &
         zbad, zdad, zcad, zawad, zaad, zb1ad, za1ad,           &
         zkwad, zk0ad, zpes, zrdc1, zrdc2, zeps,                &
         zmask, zrau0r
      REAL(wp), POINTER, DIMENSION(:,:,:) :: zws
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 ) CALL timing_start('eos-p_adj')
      !
      CALL wrk_alloc( jpi, jpj, jpk, zws )
      !
      ! initialization of adjoint variables
      ztad = 0.0_wp
      zsad = 0.0_wp
      zhad = 0.0_wp
      zsrad = 0.0_wp
      zr1ad = 0.0_wp
      zr2ad = 0.0_wp
      zr3ad = 0.0_wp
      zr4ad = 0.0_wp
      zrhopad = 0.0_wp
      zead = 0.0_wp
      zbwad = 0.0_wp
      zbad = 0.0_wp
      zdad = 0.0_wp
      zcad = 0.0_wp
      zawad = 0.0_wp
      zaad = 0.0_wp
      zb1ad  = 0.0_wp
      za1ad = 0.0_wp
      zkwad = 0.0_wp
      zk0ad = 0.0_wp

      SELECT CASE ( nn_eos )

      CASE ( 0 )               !== Jackett and McDougall (1994) formulation ==!
!!!20191119A
IF (ln_tl_passive_ocean==.true.) THEN
prd_ad(:,:,:)=0.0_wp
prhop_ad(:,:,:)=0.0_wp
ELSE
!!!/20191119A

         zrau0r = 1.e0 / rau0
#ifdef key_sp
         zeps = 1.e-7
#else
         zeps = 1.e-14
#endif
!CDIR NOVERRCHK
         zws(:,:,:) = SQRT( ABS( pts(:,:,:,jp_sal) ) )
         !
         DO jk =  jpkm1, 1, -1
            DO jj = jpj, 1, -1
               DO ji = jpi, 1, -1
                  ! direct recomputing
                  zt = pts(ji,jj,jk,jp_tem)
                  zs = pts(ji,jj,jk,jp_sal)
                  zh = fsdept(ji,jj,jk)        ! depth
                  zsr = zws(ji,jj,jk)          ! square root salinity
                  ! compute volumic mass pure water at atm pressure
                  zr1 = ( ( ( ( 6.536332e-9_wp   * zt - 1.120083e-6_wp ) * zt &
                     &        + 1.001685e-4_wp ) * zt - 9.095290e-3_wp ) * zt &
                     &        + 6.793952e-2_wp ) * zt + 999.842594_wp
                  ! seawater volumic mass atm pressure
                  zr2 = ( ( ( 5.3875e-9_wp   * zt - 8.2467e-7_wp ) * zt       &
                     &      + 7.6438e-5_wp ) * zt - 4.0899e-3_wp ) * zt + 0.824493_wp
                  zr3 = ( -1.6546e-6_wp * zt + 1.0227e-4_wp ) * zt - 5.72466e-3_wp
                  zr4 = 4.8314e-4_wp
                  ! potential volumic mass (reference to the surface)
                  zrhop = ( zr4 * zs + zr3*zsr + zr2 ) * zs + zr1
                  ! add the compression terms
                  ze  = ( -3.508914e-8_wp * zt - 1.248266e-8_wp ) * zt - 2.595994e-6_wp
                  zbw = (  1.296821e-6_wp * zt - 5.782165e-9_wp ) * zt + 1.045941e-4_wp
                  zb  = zbw + ze * zs

                  zd = -2.042967e-2_wp
                  zc =   (-7.267926e-5_wp * zt + 2.598241e-3_wp ) * zt + 0.1571896_wp
                  zaw= ( ( 5.939910e-6_wp * zt + 2.512549e-3_wp ) * zt - 0.1028859_wp &
                     &   ) * zt - 4.721788
                  za = ( zd * zsr + zc ) * zs + zaw

                  zb1=   (-0.1909078_wp   * zt + 7.390729_wp ) * zt - 55.87545_wp
                  za1= ( ( 2.326469e-3_wp * zt + 1.553190_wp ) * zt - 65.00517_wp &
                     &   ) * zt + 1044.077_wp
                  zkw= ( ( (-1.361629e-4_wp * zt - 1.852732e-2_wp ) * zt - 30.41638_wp &
                     &    ) * zt + 2098.925_wp ) * zt + 190925.6_wp
                  zk0= ( zb1 * zsr + za1 ) * zs + zkw


                  zrdc1 = 1.0 / ( zk0 - zh * ( za - zh * zb ) )
                  zrdc2 = 1.0 / ( 1.0 - zh * zrdc1 )

                  ! ============
                  ! Adjoint part
                  ! ============

                  ! Masked in situ density anomaly

                  zrhopad = zrhopad + prd_ad(ji,jj,jk) * tmask(ji,jj,jk)    &
                     &                                 * zrdc2 * zrau0r
                  zk0ad   = zk0ad   - prd_ad(ji,jj,jk) * tmask(ji,jj,jk)    &
                     &                                 * zrdc2 * zrdc2 * zh &
                     &                                 * zrdc1**2 * zrhop   &
                     &                                 * zrau0r
                  zaad    = zaad    + prd_ad(ji,jj,jk) * tmask(ji,jj,jk)    &
                     &                                 * zrdc2 * zrdc2 * zh &
                     &                                 * zrdc1**2 * zrhop   &
                     &                                 * zh * zrau0r
                  zbad    = zbad    - prd_ad(ji,jj,jk) * tmask(ji,jj,jk)    &
                     &                                 * zrdc2 * zrdc2 * zh &
                     &                                 * zrdc1**2 * zrhop   &
                     &                                 * zh * zh * zrau0r
                  prd_ad(ji,jj,jk) = 0.0_wp

                  zkwad = zkwad + zk0ad
                  zsrad = zsrad + zk0ad * zb1 * zs
                  zb1ad = zb1ad + zk0ad * zs  * zsr
                  za1ad = za1ad + zk0ad * zs
                  zsad  = zsad  + zk0ad * ( zb1 * zsr + za1 )
                  zk0ad = 0.0_wp

                  ztad  = ztad + zkwad * ( ( (-4.*1.361629e-4_wp   * zt &
                     &                        -3.*1.852732e-2_wp ) * zt &
                     &                        -2.*30.41638_wp    ) * zt &
                     &                        +   2098.925_wp           )
                  zkwad = 0.0_wp

                  ztad  = ztad + za1ad * ( ( 3.*2.326469e-3_wp   * zt &
                     &                      +2.*1.553190_wp    ) * zt &
                     &                      -   65.00517_wp           )
                  za1ad = 0.0_wp

                  ztad  = ztad + zb1ad * (-2.*0.1909078_wp     * zt &
                     &                    +   7.390729_wp           )
                  zb1ad = 0.0_wp

                  zawad = zawad + zaad
                  zsrad = zsrad + zaad *   zd * zs
                  zcad  = zcad  + zaad *   zs
                  zsad  = zsad  + zaad * ( zd * zsr + zc )
                  zaad  = 0.0_wp

                  ztad  = ztad + zawad * ( ( 3.*5.939910e-6_wp   * zt &
                     &                      +2.*2.512549e-3_wp ) * zt &
                     &                      -   0.1028859_wp          )
                  zawad = 0.0_wp

                  ztad  = ztad + zcad * (-2.*7.267926e-5_wp   * zt &
                     &                   +   2.598241e-3_wp        )
                  zcad  = 0.0_wp


                  zsad  = zsad  + zbad * ze
                  zead  = zead  + zbad * zs
                  zbwad = zbwad + zbad
                  zbad  = 0.0_wp

                  ztad  = ztad + zbwad *  ( 2.*1.296821e-6_wp   * zt &
                     &                     -   5.782165e-9_wp        )
                  zbwad = 0.0_wp

                  ztad  = ztad + zead * (-2.*3.508914e-8_wp   * zt &
                     &                   -   1.248266e-8_wp        )
                  zead =  0.0_wp

                  zrhopad = zrhopad + prhop_ad(ji,jj,jk) * tmask(ji,jj,jk)
                  prhop_ad(ji,jj,jk) = 0.0_wp

                  zr1ad   = zr1ad + zrhopad
                  zr2ad   = zr2ad + zrhopad * zs
                  zr3ad   = zr3ad + zrhopad * zsr * zs
                  zsrad   = zsrad + zrhopad * zr3 * zs
                  zsad    = zsad  + zrhopad * ( 2. * zr4 * zs + zr2  &
                     &                        + zr3 * zsr           )
                  zrhopad = 0.0_wp

                  ztad  = ztad + zr3ad * (-2.*1.6546e-6_wp   * zt &
                     &                    +   1.0227e-4_wp        )
                  zr3ad = 0.0_wp

                  ztad  = ztad + zr2ad * ( ( ( 4.*5.3875e-9_wp   * zt &
                     &                        -3.*8.2467e-7_wp ) * zt &
                     &                        +2.*7.6438e-5_wp ) * zt &
                     &                        -   4.0899e-3_wp        )
                  zr2ad = 0.0_wp

                  ztad  = ztad + zr1ad * ( ( ( ( 5.*6.536332e-9_wp   * zt &
                     &                          -4.*1.120083e-6_wp ) * zt &
                     &                          +3.*1.001685e-4_wp ) * zt &
                     &                          -2.*9.095290e-3_wp ) * zt &
                     &                          +   6.793952e-2_wp        )
                  zr1ad = 0.0_wp

                  zsad  = zsad + zsrad * ( 1.0 / MAX( 2.*zsr, zeps ) ) &
                     &                 * tmask(ji,jj,jk)
                  zsrad = 0.0_wp

                  pts_ad(ji,jj,jk,jp_sal) = pts_ad(ji,jj,jk,jp_sal) + zsad
                  pts_ad(ji,jj,jk,jp_tem) = pts_ad(ji,jj,jk,jp_tem) + ztad
                  ztad = 0.0_wp
                  zsad = 0.0_wp
               END DO
            END DO
         END DO
!!!20191119A
END IF
!!!/20191119A
         !
      CASE ( 1 )               !==  Linear formulation = F( temperature )  ==!
         DO jk = jpkm1, 1, -1
            prd_ad(:,:,jk) = prd_ad(:,:,jk) + rau0 * prhop_ad(:,:,jk) * tmask(:,:,jk)
            prhop_ad(:,:,jk) = 0.0_wp
            pts_ad(:,:,jk,jp_tem) = pts_ad(:,:,jk,jp_tem) -  rn_alpha * prd_ad(:,:,jk) * tmask(:,:,jk)
            prd_ad(:,:,jk) = 0.0_wp
         END DO
         !
      CASE ( 2 )               !==  Linear formulation = F( temperature , salinity )  ==!
         DO jk = jpkm1, 1, -1
            prd_ad(  :,:,jk) = prd_ad(:,:,jk) + rau0 * prhop_ad(:,:,jk) * tmask(:,:,jk)
            prhop_ad(:,:,jk) = 0.0_wp
            pts_ad( :,:,jk,jp_tem) = pts_ad(:,:,jk,jp_tem) - rn_alpha * prd_ad(:,:,jk) * tmask(:,:,jk)
            pts_ad( :,:,jk,jp_sal) = pts_ad(:,:,jk,jp_sal) + rn_beta   * prd_ad(:,:,jk) * tmask(:,:,jk)
            prd_ad(  :,:,jk) = 0.0_wp
         END DO
         !
      END SELECT
      CALL wrk_dealloc( jpi, jpj, jpk, zws )
      !
      IF( nn_timing == 1 ) CALL timing_stop('eos-p_adj')
      !

   END SUBROUTINE eos_insitu_pot_adj

   SUBROUTINE eos_insitu_2d_adj( pts, pdep, pts_ad, prd_ad )
      !!-----------------------------------------------------------------------
      !!
      !!        ***  ROUTINE eos_insitu_2d_adj : adj OF ROUTINE eos_insitu_2d ***
      !!
      !! ** Purpose of direct routine   : Compute the in situ density
      !!       (ratio rho/rau0) from potential temperature and salinity
      !!       using an equation of state defined through the namelist
      !!       parameter nn_eos. * 2D field case
      !!
      !! ** Method of direct routine    : 3 cases:
      !!      nn_eos = 0 : Jackett and McDougall (1994) equation of state.
      !!         the in situ density is computed directly as a function of
      !!         potential temperature relative to the surface (the opa t
      !!         variable), salt and pressure (assuming no pressure variation
      !!         along geopotential surfaces, i.e. the pressure p in decibars
      !!         is approximated by the depth in meters.
      !!              prd(t,s,p) = ( rho(t,s,p) - rau0 ) / rau0
      !!         with pressure                      p        decibars
      !!              potential temperature         t        deg celsius
      !!              salinity                      s        psu
      !!              reference volumic mass        rau0     kg/m**3
      !!              in situ volumic mass          rho      kg/m**3
      !!              in situ density anomalie      prd      no units
      !!         Check value: rho = 1060.93298 kg/m**3 for p=10000 dbar,
      !!          t = 40 deg celcius, s=40 psu
      !!      nn_eos = 1 : linear equation of state function of temperature only
      !!              prd(t) = 0.0285 - rn_alpha * t
      !!      nn_eos = 2 : linear equation of state function of temperature and
      !!               salinity
      !!              prd(t,s) = rn_beta * s - rn_alpha * tn - 1.
      !!      Note that no boundary condition problem occurs in this routine
      !!      as (ptem,psal) are defined over the whole domain.
      !!
      !! ** Comments on Adjoint Routine :
      !!      Care has been taken to avoid division by zero when computing
      !!      the inverse of the square root of salinity at masked salinity
      !!      points.
      !!
      !! ** Action  :
      !!
      !! References :
      !!
      !! History :
      !!    8.2 ! 05-03 ((F. Van den Berghe, A. Weaver, N. Daget)  - eosadj.F
      !!    9.0 ! 08-07 (A. Vidard) first version based on eosadj
      !!-----------------------------------------------------------------------
      !! * Modules used
      !! * Arguments
      REAL(wp), DIMENSION(jpi,jpj,jpts), INTENT(in   ) ::   pts    ! 1 : potential temperature  [Celcius]
      !                                                            ! 2 : salinity               [psu]
       REAL(wp), DIMENSION(jpi,jpj,jpts), INTENT(inout) ::  pts_ad ! 1 : TL of potential temperature  [Celcius]
      !                                                            ! 2 : TL of salinity               [psu]
      REAL(wp), DIMENSION(jpi,jpj ), INTENT(  inout) ::    prd_ad  ! TL of in_situ density [-]
      REAL(wp), DIMENSION(jpi,jpj)     , INTENT(in   ) ::    pdep  ! depth                  [m]
      !

      INTEGER ::  ji, jj          ! dummy loop indices
      REAL(wp) ::   &             ! temporary scalars
         zt, zs, zh, zsr, zr1, zr2, zr3, zr4, zrhop, ze, zbw,   &
         zb, zd, zc, zaw, za, zb1, za1, zkw, zk0,               &
         ztad, zsad, zhad, zsrad, zr1ad, zr2ad, zr3ad,          &
         zr4ad, zrhopad, zead, zbwad,                           &
         zbad, zdad, zcad, zawad, zaad, zb1ad, za1ad,           &
         zkwad, zk0ad, zpes, zrdc1, zrdc2, zeps,                &
         zmask
      REAL(wp), POINTER, DIMENSION(:,:) :: zws
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 ) CALL timing_start('eos2d_adj')
      !
      CALL wrk_alloc( jpi, jpj, zws )
      !
      ! initialization of adjoint variables
      ztad    = 0.0_wp
      zsad    = 0.0_wp
      zhad    = 0.0_wp
      zsrad   = 0.0_wp
      zr1ad   = 0.0_wp
      zr2ad   = 0.0_wp
      zr3ad   = 0.0_wp
      zr4ad   = 0.0_wp
      zrhopad = 0.0_wp
      zead    = 0.0_wp
      zbwad   = 0.0_wp
      zbad    = 0.0_wp
      zdad    = 0.0_wp
      zcad    = 0.0_wp
      zawad   = 0.0_wp
      zaad    = 0.0_wp
      zb1ad   = 0.0_wp
      za1ad   = 0.0_wp
      zkwad   = 0.0_wp
      zk0ad   = 0.0_wp
      SELECT CASE ( nn_eos )
      CASE ( 0 )               !== Jackett and McDougall (1994) formulation ==!
!!!20191119A
IF (ln_tl_passive_ocean==.true.) THEN
prd_ad(:,:)=0.0_wp
ELSE
!!!/20191119A

#ifdef key_sp
         zeps = 1.e-7
#else
         zeps = 1.e-14
#endif
!CDIR NOVERRCHK
         DO jj = jpjm1, 1, -1
!CDIR NOVERRCHK
            DO ji = fs_jpim1, 1, -1   ! vector opt.
               zws(ji,jj) = SQRT( ABS( pts(ji,jj,jp_sal) ) )
            END DO
         END DO
         !
         DO jj = jpjm1, 1, -1
            DO ji = fs_jpim1, 1, -1   ! vector opt.
               zmask = tmask(ji,jj,1)      ! land/sea bottom mask = surf. mask
               zt = pts (ji,jj,jp_tem)           ! interpolated T
               zs = pts (ji,jj,jp_sal)           ! interpolated S
               zsr= zws(ji,jj)             ! square root of interpolated S
               zh = pdep(ji,jj)            ! depth at the partial step level
                  ! compute volumic mass pure water at atm pressure
                  zr1= ( ( ( ( 6.536332e-9_wp*zt-1.120083e-6_wp )*zt+1.001685e-4_wp)*zt   &
                     &  -9.095290e-3_wp )*zt+6.793952e-2_wp )*zt+999.842594_wp
                  ! seawater volumic mass atm pressure
                  zr2= ( ( ( 5.3875e-9_wp*zt-8.2467e-7_wp ) *zt+7.6438e-5_wp ) *zt   &
                     &  -4.0899e-3_wp ) *zt+0.824493_wp
                  zr3= ( -1.6546e-6_wp*zt+1.0227e-4_wp ) *zt-5.72466e-3_wp
                  zr4= 4.8314e-4_wp

                  ! potential volumic mass (reference to the surface)
                  zrhop= ( zr4*zs + zr3*zsr + zr2 ) *zs + zr1

                  ! add the compression terms
                  ze = ( -3.508914e-8_wp*zt-1.248266e-8_wp ) *zt-2.595994e-6_wp
                  zbw= (  1.296821e-6_wp*zt-5.782165e-9_wp ) *zt+1.045941e-4_wp
                  zb = zbw + ze * zs

                  zd = -2.042967e-2_wp
                  zc =   (-7.267926e-5_wp*zt+2.598241e-3_wp ) *zt+0.1571896_wp
                  zaw= ( ( 5.939910e-6_wp*zt+2.512549e-3_wp ) *zt-0.1028859_wp ) *zt - 4.721788_wp
                  za = ( zd*zsr + zc ) *zs + zaw

                  zb1=   (-0.1909078_wp*zt+7.390729_wp ) *zt-55.87545_wp
                  za1= ( ( 2.326469e-3_wp*zt+1.553190_wp)*zt-65.00517_wp ) *zt+1044.077_wp
                  zkw= ( ( (-1.361629e-4_wp*zt-1.852732e-2_wp ) *zt-30.41638_wp ) *zt + 2098.925_wp ) *zt+190925.6_wp
                  zk0= ( zb1*zsr + za1 )*zs + zkw

                  zrdc1 = 1.0 / ( zk0 - zh * ( za - zh * zb ) )
                  zrdc2 = 1.0 / ( 1.0 - zh * zrdc1 )
                  ! ============
                  ! Adjoint part
                  ! ============

                  ! Masked in situ density anomaly

                  zrhopad = zrhopad + prd_ad(ji,jj) * tmask(ji,jj,1)     &
                     &                              * zrdc2 / rau0
                  zk0ad   = zk0ad   - prd_ad(ji,jj) * tmask(ji,jj,1)     &
                     &                              * zrdc2 * zrdc2 * zh &
                     &                              * zrdc1**2 * zrhop   &
                     &                              / rau0
                  zaad    = zaad    + prd_ad(ji,jj) * tmask(ji,jj,1)     &
                     &                              * zrdc2 * zrdc2 * zh &
                     &                              * zrdc1**2 * zrhop   &
                     &                              * zh / rau0
                  zbad    = zbad    - prd_ad(ji,jj) * tmask(ji,jj,1)     &
                     &                              * zrdc2 * zrdc2 * zh &
                     &                              * zrdc1**2 * zrhop   &
                     &                              * zh * zh / rau0
                  prd_ad(ji,jj) = 0.0_wp

                  zkwad = zkwad + zk0ad
                  zsrad = zsrad + zk0ad * zb1 * zs
                  zb1ad = zb1ad + zk0ad * zs  * zsr
                  za1ad = za1ad + zk0ad * zs
                  zsad  = zsad  + zk0ad * ( zb1 * zsr + za1 )
                  zk0ad = 0.0_wp

                  ztad  = ztad + zkwad * ( ( (-4.*1.361629e-4_wp   * zt &
                     &                        -3.*1.852732e-2_wp ) * zt &
                     &                        -2.*30.41638_wp    ) * zt &
                     &                        +   2098.925_wp           )
                  zkwad = 0.0_wp

                  ztad  = ztad + za1ad * ( ( 3.*2.326469e-3_wp   * zt &
                     &                      +2.*1.553190_wp    ) * zt &
                     &                      -   65.00517_wp           )
                  za1ad = 0.0_wp

                  ztad  = ztad + zb1ad * (-2.*0.1909078_wp     * zt &
                     &                    +   7.390729_wp           )
                  zb1ad = 0.0_wp

                  zawad = zawad + zaad
                  zsrad = zsrad + zaad *   zd * zs
                  zcad  = zcad  + zaad *   zs
                  zsad  = zsad  + zaad * ( zd * zsr + zc )
                  zaad  = 0.0_wp

                  ztad  = ztad + zawad * ( ( 3.*5.939910e-6_wp   * zt &
                     &                      +2.*2.512549e-3_wp ) * zt &
                     &                      -   0.1028859_wp          )
                  zawad = 0.0_wp

                  ztad  = ztad + zcad * (-2.*7.267926e-5_wp   * zt &
                     &                   +   2.598241e-3_wp        )
                  zcad  = 0.0_wp

                  zbwad = zbwad + zbad
                  zead  = zead  + zbad * zs
                  zsad  = zsad  + zbad * ze
                  zbad  = 0.0_wp

                  ztad  = ztad + zbwad *  ( 2.*1.296821e-6_wp   * zt &
                     &                     -   5.782165e-9_wp        )
                  zbwad = 0.0_wp

                  ztad  = ztad + zead * (-2.*3.508914e-8_wp   * zt &
                     &                   -   1.248266e-8_wp        )
                  zead =  0.0_wp

                  zr1ad   = zr1ad + zrhopad
                  zr2ad   = zr2ad + zrhopad * zs
                  zr3ad   = zr3ad + zrhopad * zsr * zs
                  zsrad   = zsrad + zrhopad * zr3 * zs
                  zsad    = zsad  + zrhopad * ( 2. * zr4 * zs + zr2  &
                     &                        + zr3 * zsr           )
                  zrhopad = 0.0_wp

                  ztad  = ztad + zr3ad * (-2.*1.6546e-6_wp   * zt &
                     &                    +   1.0227e-4_wp        )
                  zr3ad = 0.0_wp

                  ztad  = ztad + zr2ad * ( ( ( 4.*5.3875e-9_wp   * zt &
                     &                        -3.*8.2467e-7_wp ) * zt &
                     &                        +2.*7.6438e-5_wp ) * zt &
                     &                        -   4.0899e-3_wp        )
                  zr2ad = 0.0_wp

                  ztad  = ztad + zr1ad * ( ( ( ( 5.*6.536332e-9_wp   * zt &
                     &                          -4.*1.120083e-6_wp ) * zt &
                     &                          +3.*1.001685e-4_wp ) * zt &
                     &                          -2.*9.095290e-3_wp ) * zt &
                     &                          +   6.793952e-2_wp        )
                  zr1ad = 0.0_wp

                  zsad  = zsad + zsrad * ( 1.0 / MAX( 2.*zsr, zeps ) ) &
                     &                 * tmask(ji,jj, 1)
                  zsrad = 0.0_wp

                  pts_ad(ji,jj,jp_sal) = pts_ad(ji,jj,jp_sal) + zsad
                  pts_ad(ji,jj,jp_tem) = pts_ad(ji,jj,jp_tem) + ztad
                  ztad = 0.0_wp
                  zsad = 0.0_wp
            END DO
         END DO
!!!20191119A
END IF
!!!/20191119A
         !
      CASE ( 1 )                !==  Linear formulation = F( temperature )  ==!
         DO jj = jpjm1, 1, -1
            DO ji = fs_jpim1, 1, -1   ! vector opt.
               pts_ad(ji,jj,jp_tem) = pts_ad(ji,jj,jp_tem) - prd_ad(ji,jj) * rn_alpha * tmask(ji,jj,1)
               prd_ad(ji,jj) = 0.0_wp
            END DO
         END DO
         !
      CASE ( 2 )               !==  Linear formulation = F( temperature , salinity )  ==!
         DO jj = jpjm1, 1, -1
            DO ji = fs_jpim1, 1, -1   ! vector opt.
               pts_ad(ji,jj,jp_tem) = pts_ad(ji,jj,jp_tem) - prd_ad(ji,jj) * rn_alpha * tmask(ji,jj,1)
               pts_ad(ji,jj,jp_sal) = pts_ad(ji,jj,jp_sal) + prd_ad(ji,jj) * rn_beta  * tmask(ji,jj,1)
               prd_ad (ji,jj) = 0.0_wp
            END DO
         END DO
         !
      END SELECT
      !
      CALL wrk_dealloc( jpi, jpj, zws )
      !
      IF( nn_timing == 1 ) CALL timing_stop('eos2d_adj')
      !

   END SUBROUTINE eos_insitu_2d_adj

   SUBROUTINE eos_bn2_tan ( pts, pts_tl, pn2_tl )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE eos_bn2_tan  ***
      !!
      !! ** Purpose of the direct routine:   Compute the local
      !!      Brunt-Vaisala frequency at the time-step of the input arguments
      !!
      !! ** Method of the direct routine:
      !!       * nn_eos = 0  : UNESCO sea water properties
      !!         The brunt-vaisala frequency is computed using the polynomial
      !!      polynomial expression of McDougall (1987):
      !!            N^2 = grav * beta * ( alpha/beta*dk[ t ] - dk[ s ] )/e3w
      !!      If lk_zdfddm=T, the heat/salt buoyancy flux ratio Rrau is
      !!      computed and used in zdfddm module :
      !!              Rrau = alpha/beta * ( dk[ t ] / dk[ s ] )
      !!       * nn_eos = 1  : linear equation of state (temperature only)
      !!            N^2 = grav * rn_alpha * dk[ t ]/e3w
      !!       * nn_eos = 2  : linear equation of state (temperature & salinity)
      !!            N^2 = grav * (rn_alpha * dk[ t ] - rn_beta * dk[ s ] ) / e3w
      !!      The use of potential density to compute N^2 introduces e r r o r
      !!      in the sign of N^2 at great depths. We recommand the use of
      !!      nn_eos = 0, except for academical studies.
      !!        Macro-tasked on horizontal slab (jk-loop)
      !!      N.B. N^2 is set to zero at the first level (JK=1) in inidtr
      !!      and is never used at this level.
      !!
      !! ** Action  : - pn2 : the brunt-vaisala frequency
      !!
      !! References :
      !!      McDougall, T. J., J. Phys. Oceanogr., 17, 1950-1964, 1987.
      !!
      !! History:
      !!        !  08-07  (A. Vidard) First version
      !!----------------------------------------------------------------------
      !! * Arguments
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::   pts, &   ! 1 : potential temperature  [Celcius]
      !                                                                  ! 2 : salinity               [psu]
      &                                                         pts_tl   ! 1 : TL of potential temperature [Celsius]
                                                                         ! 2 : TL of salinity [psu]
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT( out ) ::   &
         & pn2_tl                   ! TL of potential density (surface referenced)
      !! * Local declarations
      INTEGER  ::   ji, jj, jk   ! dummy loop indices
      REAL(wp) ::             &
         zgde3w, zt, zs, zh,  &  ! temporary scalars
         zalbet, zbeta           !    "         "
      REAL(wp) ::             &
         zttl, zstl,          &  ! temporary scalars
         zalbettl, zbetatl       !    "         "
#if defined key_zdfddm
      REAL(wp) ::   zds, zdstl   ! temporary scalars
#endif

      ! pn2_tl : interior points only (2=< jk =< jpkm1 )
      ! --------------------------
      SELECT CASE ( nn_eos )

      CASE ( 0 )               !== Jackett and McDougall (1994) formulation ==!
!!! 20191119A 
IF ( ln_tl_passive_ocean == .true.) THEN
pn2_tl(:,:,:)=0.0_wp
ELSE
!!! /20191119A
         DO jk = 2, jpkm1
            DO jj = 1, jpj
               DO ji = 1, jpi
                  zgde3w = grav / fse3w(ji,jj,jk)
                  zt = 0.5 * ( pts(ji,jj,jk,jp_tem) + pts(ji,jj,jk-1,jp_tem) )          ! potential temperature at w-point
                  zs = 0.5 * ( pts(ji,jj,jk,jp_sal) + pts(ji,jj,jk-1,jp_sal) ) - 35.0   ! salinity anomaly (s-35) at w-point
                  zh = fsdepw(ji,jj,jk)                                     ! depth in meters  at w-point

                  zalbet = ( ( ( - 0.255019e-07_wp * zt + 0.298357e-05_wp ) * zt   &   ! ratio alpha/beta
                     &                               - 0.203814e-03_wp ) * zt   &
                     &                               + 0.170907e-01_wp ) * zt   &
                     &   + 0.665157e-01_wp                                      &
                     &   +     ( - 0.678662e-05_wp * zs                         &
                     &           - 0.846960e-04_wp * zt + 0.378110e-02_wp ) * zs   &
                     &   +   ( ( - 0.302285e-13_wp * zh                         &
                     &           - 0.251520e-11_wp * zs                         &
                     &           + 0.512857e-12_wp * zt * zt           ) * zh   &
                     &           - 0.164759e-06_wp * zs                         &
                     &        +(   0.791325e-08_wp * zt - 0.933746e-06_wp ) * zt   &
                     &                               + 0.380374e-04_wp ) * zh

                  zbeta  = ( ( -0.415613e-09_wp * zt + 0.555579e-07_wp ) * zt      &   ! beta
                     &                            - 0.301985e-05_wp ) * zt      &
                     &   + 0.785567e-03_wp                                      &
                     &   + (     0.515032e-08_wp * zs                           &
                     &         + 0.788212e-08_wp * zt - 0.356603e-06_wp ) * zs     &
                     &   +(  (   0.121551e-17_wp * zh                           &
                     &         - 0.602281e-15_wp * zs                           &
                     &         - 0.175379e-14_wp * zt + 0.176621e-12_wp ) * zh     &
                     &                             + 0.408195e-10_wp   * zs     &
                     &     + ( - 0.213127e-11_wp * zt + 0.192867e-09_wp ) * zt     &
                     &                             - 0.121555e-07_wp ) * zh

                  !! tangent part
                  zttl = 0.5 * ( pts_tl(ji,jj,jk,jp_tem) + pts_tl(ji,jj,jk-1,jp_tem) )  ! potential temperature at w-point
                  zstl = 0.5 * ( pts_tl(ji,jj,jk,jp_sal) + pts_tl(ji,jj,jk-1,jp_sal) )  ! salinity anomaly at w-point
                  zalbettl = (    (   ( -4.*0.255019e-07_wp   * zt                    & ! ratio alpha/beta
                     &              +3.*0.298357e-05_wp ) * zt                        &
                     &              -2.*0.203814e-03_wp ) * zt                        &
                     &      +           0.170907e-01_wp                               &
                     &      -           0.846960e-04_wp   * zs                        &
                     &      - (         0.933746e-06_wp                               &
                     &          - (  2.*0.791325e-08_wp                               &
                     &              +2.*0.512857e-12_wp   * zh ) * zt ) * zh ) * zttl &
                     & + (  -        2.*0.678662e-05_wp   * zs                        &
                     &      -           0.846960e-04_wp   * zt                        &
                     &      +           0.378110e-02_wp                               &
                     &      + (     -   0.164759e-06_wp                               &
                     &              -   0.251520e-11_wp   * zh ) * zh         ) * zstl

                  zbetatl = (    (     -3.*0.415613e-09_wp   * zt               &
                     &              +2.*0.555579e-07_wp ) * zt                  &
                     &      -           0.301985e-05_wp                         &
                     &      +           0.788212e-08_wp   * zs                  &
                     &      + (     -2.*0.213127e-11_wp   * zt                  &
                     &              -   0.175379e-14_wp   * zh                  &
                     &              +   0.192867e-09_wp         ) * zh ) * zttl &
                     & + (           2.*0.515032e-08_wp   * zs                  &
                     &      +           0.788212e-08_wp   * zt                  &
                     &      -           0.356603e-06_wp                         &
                     &      + (     -   0.602281e-15_wp   * zh                  &
                     &              +   0.408195e-10_wp         ) * zh ) * zstl

                     pn2_tl(ji,jj,jk) = zgde3w * tmask(ji,jj,jk) * (                                   &
                  &       zbeta   * (  zalbet                                        &
                  &                  * ( pts_tl(ji,jj,jk-1,jp_tem) - pts_tl(ji,jj,jk,jp_tem) )   &
                  &                 +  zalbettl                                      &
                  &                 * ( pts  (ji,jj,jk-1,jp_tem) - pts  (ji,jj,jk,jp_tem) )      &
                  &                 -  ( pts_tl(ji,jj,jk-1,jp_sal) - pts_tl(ji,jj,jk,jp_sal) ) ) &
                  &     + zbetatl * (  zalbet                                        &
                  &                  * ( pts  (ji,jj,jk-1,jp_tem) - pts  (ji,jj,jk,jp_tem) )     &
                  &                 -  ( pts  (ji,jj,jk-1,jp_sal) - pts  (ji,jj,jk,jp_sal) ) ) )
#if defined key_zdfddm
                     zds = ( pts(ji,jj,jk-1,jp_sal) - pts(ji,jj,jk,jp_sal) )
                     zdstl = ( pts_tl(ji,jj,jk-1,jp_sal) - pts_tl(ji,jj,jk,jp_sal) )
                     IF ( ABS( zds) <= 1.e-20 ) THEN
                        zds = 1.e-20
                        rrau_tl(ji,jj,jk) = zalbettl * &
                             &    ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) / zds       &
                             &              + zalbet *   &
                             &  ( ( pts_tl(ji,jj,jk-1,jp_tem) - pts_tl(ji,jj,jk,jp_tem) ) / zds )
                     ELSE
                     rrau_tl(ji,jj,jk) = zalbettl * &
                        &    ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) / zds       &
                        &              + zalbet *   &
                        &  ( ( pts_tl(ji,jj,jk-1,jp_tem) - pts_tl(ji,jj,jk,jp_tem) ) / zds &
                        &  - ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) * zdstl/ zds**2 )
                  ENDIF
#endif
               END DO
            END DO
         END DO
         !
!!!20191119A
END IF
!!!/20191119A
      CASE ( 1 )                !==  Linear formulation = F( temperature )  ==!
         DO jk = 2, jpkm1
            pn2_tl(:,:,jk) = rn_alpha * ( pts_tl(:,:,jk-1,jp_tem) - pts_tl(:,:,jk,jp_tem) ) * grav / fse3w(:,:,jk) * tmask(:,:,jk)
         END DO
         !
      CASE ( 2 )               !==  Linear formulation = F( temperature , salinity )  ==!
         DO jk = 2, jpkm1
            pn2_tl(:,:,jk) = ( rn_alpha * ( pts_tl(:,:,jk-1,jp_tem) - pts_tl(:,:,jk,jp_tem) )    &
                             & - rn_beta  * ( pts_tl(:,:,jk-1,jp_sal) - pts_tl(:,:,jk,jp_sal) )  ) &
                             & * grav / fse3w(:,:,jk) * tmask(:,:,jk)
         END DO
#if defined key_zdfddm
         DO jk = 2, jpkm1
            DO jj = 1, jpj
               DO ji = 1, jpi
                  zds = ( pts(ji,jj,jk-1,jp_sal) - pts(ji,jj,jk,jp_sal) )
                  zdstl = pts_tl(ji,jj,jk-1,jp_sal) - pts_tl(ji,jj,jk,jp_sal)
                  IF ( ABS( zds) <= 1.e-20 ) THEN
                     zds = 1.e-20
                     rrau_tl(ji,jj,jk) = ralpbet * &
                     & ( ( pts_tl(ji,jj,jk-1,jp_tem) - pts_tl(ji,jj,jk,jp_tem) ) / zds )
                  ELSE
                     rrau_tl(ji,jj,jk) = ralpbet * &
                     & ( ( pts_tl(ji,jj,jk-1,jp_tem) - pts_tl(ji,jj,jk,jp_tem) ) / zds &
                     & - ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) * zdstl / zds**2 )
                  ENDIF
                  rrau(ji,jj,jk) = ralpbet * ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) / zds
               END DO
            END DO
         END DO
#endif
      END SELECT
   END SUBROUTINE eos_bn2_tan

   SUBROUTINE eos_bn2_adj ( pts, pts_ad, pn2_ad )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE eos_bn2_adj  ***
      !!
      !! ** Purpose of the direct routine:   Compute the local
      !!      Brunt-Vaisala frequency at the time-step of the input arguments
      !!
      !! ** Method of the direct routine:
      !!       * nn_eos = 0  : UNESCO sea water properties
      !!         The brunt-vaisala frequency is computed using the polynomial
      !!      polynomial expression of McDougall (1987):
      !!            N^2 = grav * beta * ( alpha/beta*dk[ t ] - dk[ s ] )/e3w
      !!      If lk_zdfddm=T, the heat/salt buoyancy flux ratio Rrau is
      !!      computed and used in zdfddm module :
      !!              Rrau = alpha/beta * ( dk[ t ] / dk[ s ] )
      !!       * nn_eos = 1  : linear equation of state (temperature only)
      !!            N^2 = grav * rn_alpha * dk[ t ]/e3w
      !!       * nn_eos = 2  : linear equation of state (temperature & salinity)
      !!            N^2 = grav * (rn_alpha * dk[ t ] - rn_beta * dk[ s ] ) / e3w
      !!      The use of potential density to compute N^2 introduces e r r o r
      !!      in the sign of N^2 at great depths. We recommand the use of
      !!      nn_eos = 0, except for academical studies.
      !!        Macro-tasked on horizontal slab (jk-loop)
      !!      N.B. N^2 is set to zero at the first level (JK=1) in inidtr
      !!      and is never used at this level.
      !!
      !! ** Action  : - pn2 : the brunt-vaisala frequency
      !!
      !! References :
      !!      McDougall, T. J., J. Phys. Oceanogr., 17, 1950-1964, 1987.
      !!
      !! History:
      !!        !  08-07  (A. Vidard) First version
      !!----------------------------------------------------------------------
      !! * Arguments
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::  pts       ! 1 : potential temperature  [Celcius]
      !                                                                  ! 2 : salinity               [psu]
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(inout   ) ::  pts_ad    ! 1 : Adjoint of potential temperature [Celsius]
                                                                         ! 2 : Adjoint of salinity [psu]
      REAL(wp), DIMENSION(jpi,jpj,jpk), INTENT( inout ) ::   &
         & pn2_ad                   ! Adjoint of potential density (surface referenced)
      !
      !! * Local declarations
      INTEGER  ::   ji, jj, jk   ! dummy loop indices
      REAL(wp) ::             &
         zgde3w, zt, zs, zh,  &  ! temporary scalars
         zalbet, zbeta           !    "         "
      REAL(wp) ::             &
         ztad, zsad,          &  ! temporary scalars
         zalbetad, zbetaad       !    "         "
#if defined key_zdfddm
      REAL(wp) ::   zds, zdsad          ! temporary scalars
#endif
      ! pn2_tl : interior points only (2=< jk =< jpkm1 )
      ! --------------------------

      zalbetad = 0.0_wp
      zbetaad  = 0.0_wp
      ztad     = 0.0_wp
      zsad     = 0.0_wp
#if defined key_zdfddm
      zdsad    = 0.0_wp
#endif

      SELECT CASE ( nn_eos )

      CASE ( 0 )               !== Jackett and McDougall (1994) formulation ==!
         DO jk = jpkm1, 2, -1
            DO jj = jpj, 1, -1
               DO ji = jpi, 1, -1
                  zgde3w = grav / fse3w(ji,jj,jk)
                  zt = 0.5 * ( pts(ji,jj,jk,jp_tem) + pts(ji,jj,jk-1,jp_tem) )          ! potential temperature at w-point
                  zs = 0.5 * ( pts(ji,jj,jk,jp_sal) + pts(ji,jj,jk-1,jp_sal) ) - 35.0   ! salinity anomaly (s-35) at w-point
                  zh = fsdepw(ji,jj,jk)                                     ! depth in meters  at w-point

                  zalbet = ( ( ( - 0.255019e-07_wp * zt + 0.298357e-05_wp ) * zt   &   ! ratio alpha/beta
                     &                               - 0.203814e-03_wp ) * zt   &
                     &                               + 0.170907e-01_wp ) * zt   &
                     &   + 0.665157e-01_wp                                      &
                     &   +     ( - 0.678662e-05_wp * zs                         &
                     &           - 0.846960e-04_wp * zt + 0.378110e-02_wp ) * zs   &
                     &   +   ( ( - 0.302285e-13_wp * zh                         &
                     &           - 0.251520e-11_wp * zs                         &
                     &           + 0.512857e-12_wp * zt * zt           ) * zh   &
                     &           - 0.164759e-06_wp * zs                         &
                     &        +(   0.791325e-08_wp * zt - 0.933746e-06_wp ) * zt   &
                     &                                  + 0.380374e-04_wp ) * zh

                  zbeta  = ( ( -0.415613e-09_wp * zt + 0.555579e-07_wp ) * zt      &   ! beta
                     &                               - 0.301985e-05_wp ) * zt      &
                     &   + 0.785567e-03_wp                                      &
                     &   + (     0.515032e-08_wp * zs                           &
                     &         + 0.788212e-08_wp * zt - 0.356603e-06_wp ) * zs     &
                     &   +(  (   0.121551e-17_wp * zh                           &
                     &         - 0.602281e-15_wp * zs                           &
                     &         - 0.175379e-14_wp * zt + 0.176621e-12_wp ) * zh     &
                     &                                + 0.408195e-10_wp   * zs     &
                     &     + ( - 0.213127e-11_wp * zt + 0.192867e-09_wp ) * zt     &
                     &                                - 0.121555e-07_wp ) * zh

#if defined key_zdfddm

                  zds = ( pts(ji,jj,jk-1,jp_sal) - pts(ji,jj,jk,jp_sal) )
                  IF ( ABS( zds) <= 1.e-20 ) THEN
                     zds = 1.e-20
                     zdsad = 0.0_wp
                  ELSE
                     zds = ( pts(ji,jj,jk-1,jp_sal) - pts(ji,jj,jk,jp_sal) )
                     zdsad = rrau_ad(ji,jj,jk) * zalbet *( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) / zds**2
                     zds = ( pts(ji,jj,jk-1,jp_sal) - pts(ji,jj,jk,jp_sal) )
                  ENDIF
                  pts_ad(ji,jj,jk-1,jp_tem) =  pts_ad(ji,jj,jk-1,jp_tem) +  rrau_ad(ji,jj,jk) * zalbet / zds
                  pts_ad(ji,jj,jk,jp_tem  ) =  pts_ad(ji,jj,jk,jp_tem  ) -  rrau_ad(ji,jj,jk) * zalbet / zds
                  zalbetad = zalbetad + rrau_ad(ji,jj,jk) * ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) / zds
                  rrau_ad(ji,jj,jk) = 0._wp
                  pts_ad(ji,jj,jk-1,jp_sal) = pts_ad(ji,jj,jk-1,jp_sal) + zdsad
                  pts_ad(ji,jj,jk,jp_sal  ) = pts_ad(ji,jj,jk,jp_sal  ) - zdsad
                  zdsad = 0._wp
#endif
                     pts_ad(ji,jj,jk-1,jp_tem) = pts_ad(ji,jj,jk-1,jp_tem) + zalbet*zbeta*zgde3w*tmask(ji,jj,jk)*pn2_ad(ji,jj,jk)
                     pts_ad(ji,jj,jk,jp_tem  ) = pts_ad(ji,jj,jk,jp_tem  ) - zalbet*zbeta*zgde3w*tmask(ji,jj,jk)*pn2_ad(ji,jj,jk)
                     zalbetad = zalbetad + zbeta*zgde3w*tmask(ji,jj,jk)   &
                              &   *( pts  (ji,jj,jk-1,jp_tem) - pts  (ji,jj,jk,jp_tem) ) * pn2_ad(ji,jj,jk)
                     pts_ad(ji,jj,jk-1,jp_sal) = pts_ad(ji,jj,jk-1,jp_sal) - zbeta*tmask(ji,jj,jk)*zgde3w*pn2_ad(ji,jj,jk)
                     pts_ad(ji,jj,jk,jp_sal  ) = pts_ad(ji,jj,jk,jp_sal  ) + zbeta*tmask(ji,jj,jk)*zgde3w*pn2_ad(ji,jj,jk)
                     zbetaad = zbetaad &
                        & + zgde3w *tmask(ji,jj,jk)* (  zalbet * ( pts  (ji,jj,jk-1,jp_tem) - pts  (ji,jj,jk,jp_tem) )  &
                        & -  ( pts  (ji,jj,jk-1,jp_sal) - pts  (ji,jj,jk,jp_sal) ) )*pn2_ad(ji,jj,jk)

                     pn2_ad(ji,jj,jk)  = 0.0_wp

                     ztad = ztad + (    (     -3.*0.415613e-09_wp   * zt           &
                        &              +2.*0.555579e-07_wp ) * zt                  &
                        &      -           0.301985e-05_wp                         &
                        &      +           0.788212e-08_wp   * zs                  &
                        &      + (     -2.*0.213127e-11_wp   * zt                  &
                        &              -   0.175379e-14_wp   * zh                  &
                        &              +   0.192867e-09_wp         ) * zh ) *zbetaad

                     zsad = zsad + (           2.*0.515032e-08_wp   * zs           &
                        &      +           0.788212e-08_wp   * zt                  &
                        &      -           0.356603e-06_wp                         &
                        &      + (     -   0.602281e-15_wp   * zh                  &
                        &              +   0.408195e-10_wp         ) * zh ) * zbetaad

                     zbetaad = 0.0_wp

                     ztad = ztad + (    (   ( -4.*0.255019e-07_wp   * zt           &! ratio alpha/beta
                        &              +3.*0.298357e-05_wp ) * zt                  &
                        &              -2.*0.203814e-03_wp ) * zt                  &
                        &      +           0.170907e-01_wp                         &
                        &      -           0.846960e-04_wp   * zs                  &
                        &      - (         0.933746e-06_wp                         &
                        &          - (  2.*0.791325e-08_wp                         &
                        &              +2.*0.512857e-12_wp   * zh ) * zt ) * zh    &
                        &                                                 ) *zalbetad

                     zsad = zsad + (  -        2.*0.678662e-05_wp   * zs           &
                        &      -           0.846960e-04_wp   * zt                  &
                        &      +           0.378110e-02_wp                         &
                        &      + (     -   0.164759e-06_wp                         &
                        &              -   0.251520e-11_wp   * zh ) * zh           &
                        &                                                 ) *zalbetad

                     zalbetad = 0.0_wp


                     pts_ad(ji,jj,jk,jp_sal) = pts_ad(ji,jj,jk,jp_sal) + 0.5 * zsad
                     pts_ad(ji,jj,jk-1,jp_sal) = pts_ad(ji,jj,jk-1,jp_sal) + 0.5 * zsad
                     zsad = 0.0_wp

                     pts_ad(ji,jj,jk,jp_tem) = pts_ad(ji,jj,jk,jp_tem) + 0.5 * ztad
                     pts_ad(ji,jj,jk-1,jp_tem) = pts_ad(ji,jj,jk-1,jp_tem) + 0.5 * ztad
                     ztad = 0.0_wp

               END DO
            END DO
         END DO
         !
      CASE ( 1 )               !==  Linear formulation = F( temperature )  ==!
         DO jk = jpkm1, 2, -1
            pts_ad(:,:,jk-1,jp_tem) = pts_ad(:,:,jk-1,jp_tem)  + rn_alpha * pn2_ad(:,:,jk) &
                                      & * grav / fse3w(:,:,jk) * tmask(:,:,jk)
            pts_ad(:,:,jk,jp_tem  ) = pts_ad(:,:,jk,jp_tem  )  - rn_alpha * pn2_ad(:,:,jk) &
                                      & * grav / fse3w(:,:,jk) * tmask(:,:,jk)
            pn2_ad(:,:,jk) = 0.0_wp
         END DO
         !
      CASE ( 2 )               !==  Linear formulation = F( temperature , salinity )  ==!
#if defined key_zdfddm
         DO jk = jpkm1, 2, -1
            DO jj = jpj, 1, -1
               DO ji = jpi, 1, -1
                  zds = ( pts(ji,jj,jk-1,jp_sal) - pts(ji,jj,jk,jp_sal) )
                  IF ( ABS( zds) <= 1.e-20 ) THEN
                     zds = 1.e-20
                     zdsad = 0.0_wp
                  ELSE
                     zdsad = zdsad - rrau_ad(ji,jj,jk) * ralpbet &
                     & * ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) / zds**2
                  ENDIF
                  rrau(ji,jj,jk) = ralpbet * ( pts(ji,jj,jk-1,jp_tem) - pts(ji,jj,jk,jp_tem) ) / zds
                  pts_ad(ji,jj,jk-1,jp_tem) = pts_ad(ji,jj,jk-1,jp_tem)   &
                     & + rrau_ad(ji,jj,jk) * ralpbet / zds
                  pts_ad(ji,jj,jk,jp_tem  ) = pts_ad(ji,jj,jk,jp_tem  )   &
                     & - rrau_ad(ji,jj,jk) * ralpbet / zds
                  rrau_ad(ji,jj,jk)  = 0._wp

                  pts_ad(ji,jj,jk-1,jp_sal) = pts_ad(ji,jj,jk-1,jp_sal) + zdsad
                  pts_ad(ji,jj,jk,jp_sal  ) = pts_ad(ji,jj,jk,jp_sal  ) - zdsad
                  zdsad = 0._wp
               END DO
            END DO
         END DO
#endif
         DO jk = jpkm1, 2, -1
            pts_ad(:,:,jk-1,jp_tem) = pts_ad(:,:,jk-1,jp_tem) + rn_alpha * pn2_ad(:,:,jk) &
                       & * grav / fse3w(:,:,jk) * tmask(:,:,jk)
            pts_ad(:,:,jk,jp_tem  ) = pts_ad(:,:,jk,jp_tem  ) - rn_alpha * pn2_ad(:,:,jk) &
                       & * grav / fse3w(:,:,jk) * tmask(:,:,jk)
            pts_ad(:,:,jk-1,jp_sal) = pts_ad(:,:,jk-1,jp_sal) - rn_beta  * pn2_ad(:,:,jk) &
                       & * grav / fse3w(:,:,jk) * tmask(:,:,jk)
            pts_ad(:,:,jk,jp_sal  ) = pts_ad(:,:,jk,jp_sal  ) + rn_beta  * pn2_ad(:,:,jk) &
                       & * grav / fse3w(:,:,jk) * tmask(:,:,jk)
            pn2_ad(:,:,jk) = 0.0_wp
         END DO
      END SELECT

   END SUBROUTINE eos_bn2_adj

   SUBROUTINE eos_alpbet_tan( pts, pts_tl, palpbet_tl, beta0_tl )
      !!----------------------------------------------------------------------
      !!                 ***  ROUTINE eos_alpbet_tan  ***
      !!
      !! ** Purpose of the direct routine :
      !!       Calculates the in situ thermal/haline expansion ratio at T-points
      !!
      !! ** Method  of the direct routine :
      !!       calculates alpha / beta ratio at T-points
      !!       * nn_eos = 0  : UNESCO sea water properties
      !!                       The alpha/beta ratio is returned as 3-D array palpbet using the polynomial
      !!                       polynomial expression of McDougall (1987).
      !!                       Scalar beta0 is returned = 1.
      !!       * nn_eos = 1  : linear equation of state (temperature only)
      !!                       The ratio is undefined, so we return alpha as palpbet
      !!                       Scalar beta0 is returned = 0.
      !!       * nn_eos = 2  : linear equation of state (temperature & salinity)
      !!                       The alpha/beta ratio is returned as ralpbet
      !!                       Scalar beta0 is returned = 1.
      !!
      !! ** Action  : - palpbet : thermal/haline expansion ratio at T-points
      !!            :   beta0   : 1. or 0.
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in   ) ::   pts_tl, &    ! linear tangent of pot. temperature & salinity
         &                                                      pts          ! pot. temperature & salinity
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(  out) ::   palpbet_tl   ! thermal/haline expansion ratio
      REAL(wp),                              INTENT(  out) ::   beta0_tl     ! set = 1 except with case 1 eos, rho=rho(T)
      !!
      INTEGER  ::   ji, jj, jk      ! dummy loop indices
      REAL(wp) ::   zt, zs, zh, &   ! local scalars
         &          zttl, zstl
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 ) CALL timing_start('eos_alpbet_tan')
      !
      SELECT CASE ( nn_eos )
      !
      CASE ( 0 )               ! Jackett and McDougall (1994) formulation
         DO jk = 1, jpk
            DO jj = 1, jpj
               DO ji = 1, jpi
                  zt = pts(ji,jj,jk,jp_tem)           ! potential temperature
                  zs = pts(ji,jj,jk,jp_sal) - 35._wp  ! salinity anomaly (s-35)
                  zh = fsdept(ji,jj,jk)               ! depth in meters
                  !! Tangent part
                  zttl = pts_tl(ji,jj,jk,jp_tem)           ! potential temperature
                  zstl = pts_tl(ji,jj,jk,jp_sal)           ! salinity anomaly (s-35)
                  palpbet_tl(ji,jj,jk) =                                           &
                     &     ( ( ( ( - 4. * 0.255019e-07_wp * zt                     &
                     &           + 3. * 0.298357e-05_wp ) * zt                     &
                     &           - 2. * 0.203814e-03_wp ) * zt                     &
                     &           + 0.170907e-01_wp * zt )                          &
                     &           - 0.846960e-04_wp * zs                            &
                     &     + ( ( 2. * 0.512857e-12_wp * zt ) * zh                  &
                     &     +   ( 2. * 0.791325e-08_wp * zt                         &
                     &           - 0.933746e-06_wp ) ) * zh ) * zttl               &
                     &     + ( ( - 2. * 0.678662e-05_wp * zs                       &
                     &           - 0.846960e-04_wp * zt                            &
                     &           + 0.378110e-02_wp )                               &
                     &       + ( - 0.251520e-11_wp  * zh                           &
                     &           - 0.164759e-06_wp ) * zh ) * zstl
               END DO
            END DO
         END DO
         beta0_tl = 0._wp
         !
      CASE ( 1 )              !==  Linear formulation = F( temperature )  ==!
         palpbet_tl(:,:,:) = 0._wp
         beta0_tl = 0._wp
         !
      CASE ( 2 )              !==  Linear formulation = F( temperature , salinity )  ==!
         palpbet_tl(:,:,:) = 0._wp
         beta0_tl = 0._wp
         !
      CASE DEFAULT
         IF(lwp) WRITE(numout,cform_err)
         IF(lwp) WRITE(numout,*) '          bad flag value for nn_eos = ', nn_eos
         nstop = nstop + 1
         !
      END SELECT
      !
      IF( nn_timing == 1 ) CALL timing_stop('eos_alpbet_tan')
      !
   END SUBROUTINE eos_alpbet_tan

   SUBROUTINE eos_alpbet_adj( pts, pts_ad, palpbet_ad, beta0_ad )
      !!----------------------------------------------------------------------
      !!                 ***  ROUTINE eos_alpbet_adj  ***
      !!
      !! ** Purpose of the direct routine :
      !!       Calculates the in situ thermal/haline expansion ratio at T-points
      !!
      !! ** Method  of the direct routine :
      !!       calculates alpha / beta ratio at T-points
      !!       * nn_eos = 0  : UNESCO sea water properties
      !!                       The alpha/beta ratio is returned as 3-D array palpbet using the polynomial
      !!                       polynomial expression of McDougall (1987).
      !!                       Scalar beta0 is returned = 1.
      !!       * nn_eos = 1  : linear equation of state (temperature only)
      !!                       The ratio is undefined, so we return alpha as palpbet
      !!                       Scalar beta0 is returned = 0.
      !!       * nn_eos = 2  : linear equation of state (temperature & salinity)
      !!                       The alpha/beta ratio is returned as ralpbet
      !!                       Scalar beta0 is returned = 1.
      !!
      !! ** Action  : - palpbet : thermal/haline expansion ratio at T-points
      !!            :   beta0   : 1. or 0.
      !!----------------------------------------------------------------------
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(inout) ::   pts_ad       ! linear tangent of pot. temperature & salinity
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in)    ::   pts          ! pot. temperature & salinity
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(inout) ::   palpbet_ad   ! thermal/haline expansion ratio
      REAL(wp),                              INTENT(inout) ::   beta0_ad     ! set = 1 except with case 1 eos, rho=rho(T)
      !!
      INTEGER  ::   ji, jj, jk      ! dummy loop indices
      REAL(wp) ::   zt, zs, zh, &   ! local scalars
         &          ztad, zsad
      !!----------------------------------------------------------------------
      !
      ztad = 0.0_wp
      zsad = 0.0_wp
      IF( nn_timing == 1 ) CALL timing_start('eos_alpbet_adj')
      !
      SELECT CASE ( nn_eos )
      !
      CASE ( 0 )               ! Jackett and McDougall (1994) formulation
         DO jk = jpk, 1, -1
            DO jj = jpj, 1, -1
               DO ji = jpi, 1, -1
                  zt = pts(ji,jj,jk,jp_tem)           ! potential temperature
                  zs = pts(ji,jj,jk,jp_sal) - 35._wp  ! salinity anomaly (s-35)
                  zh = fsdept(ji,jj,jk)               ! depth in meters
                  !! Adjoint part
                  ztad = ztad + ( ( ( ( - 4. * 0.255019e-07_wp * zt                &
                     &           + 3. * 0.298357e-05_wp ) * zt                     &
                     &           - 2. * 0.203814e-03_wp ) * zt                     &
                     &           + 0.170907e-01_wp * zt )                          &
                     &           - 0.846960e-04_wp * zs                            &
                     &     + ( ( 2. * 0.512857e-12_wp * zt ) * zh                  &
                     &     +   ( 2. * 0.791325e-08_wp * zt                         &
                     &           - 0.933746e-06_wp ) ) * zh ) * palpbet_ad(ji,jj,jk)
                  zsad = zsad + ( ( - 2. * 0.678662e-05_wp * zs                    &
                     &           - 0.846960e-04_wp * zt                            &
                     &           + 0.378110e-02_wp )                               &
                     &       + ( - 0.251520e-11_wp  * zh                           &
                     &           - 0.164759e-06_wp ) * zh ) * palpbet_ad(ji,jj,jk)
                  palpbet_ad(ji,jj,jk) = 0.0_wp
                  pts_ad(ji,jj,jk,jp_tem) = pts_ad(ji,jj,jk,jp_tem) + ztad
                  pts_ad(ji,jj,jk,jp_sal) = pts_ad(ji,jj,jk,jp_sal) + zsad
                  ztad = 0.0_wp
                  zsad = 0.0_wp
               END DO
            END DO
         END DO
         beta0_ad = 0._wp
         !
      CASE ( 1 )              !==  Linear formulation = F( temperature )  ==!
         palpbet_ad(:,:,:) = 0._wp
         beta0_ad = 0._wp
         !
      CASE ( 2 )              !==  Linear formulation = F( temperature , salinity )  ==!
         palpbet_ad(:,:,:) = 0._wp
         beta0_ad = 0._wp
         !
      CASE DEFAULT
         IF(lwp) WRITE(numout,cform_err)
         IF(lwp) WRITE(numout,*) '          bad flag value for nn_eos = ', nn_eos
         nstop = nstop + 1
         !
      END SELECT
      !
      IF( nn_timing == 1 ) CALL timing_stop('eos_alpbet_adj')
      !
   END SUBROUTINE eos_alpbet_adj

#if defined key_tam
   SUBROUTINE eos_insitu_adj_tst( kumadt )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE eos_adj_tst ***
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
      !!        ! 08-07 (A. Vidard)
      !!-----------------------------------------------------------------------
      !! * Modules used

      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt             ! Output unit
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         zts                       ! potential temperature
                                   ! salinity
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts_adout                 ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zrd_adin                 ! anomaly density
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts_tlin                 ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & znts                     ! potential temperature
                                   ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zrd_tlout                  ! anomaly density
      REAL(KIND=wp) ::       &
         & zsp1,             & ! scalar product involving the tangent routine
         & zsp2                ! scalar product involving the adjoint routine
      INTEGER :: &
         & ji, &
         & jj, &
         & jk, &
      & jn, &
      & jeos
      CHARACTER(LEN=14) :: cl_name
      ALLOCATE( &
         & zts(     jpi, jpj, jpk, jpts ),  &
         & znts(      jpi, jpj, jpk, jpts ),  &
         & zts_adout( jpi, jpj, jpk,jpts ),  &
         & zrd_adin( jpi, jpj, jpk ),  &
         & zts_tlin(  jpi, jpj, jpk,jpts ),  &
         & zrd_tlout(jpi, jpj, jpk )    )
      ! Initialize the reference state
      zts = tsn
      ! store initial nn_eos
      jeos = nn_eos
      DO jn = 0, 2
         nn_eos = jn
         !=============================================================
         ! 1) dx = ( T ) and dy = ( T )
         !=============================================================

         !--------------------------------------------------------------------
         ! Reset the tangent and adjoint variables
         !--------------------------------------------------------------------
         zts_tlin(:,:,:,:)     = 0.0_wp
         zrd_tlout(:,:,:)   = 0.0_wp
         zts_adout(:,:,:,:)    = 0.0_wp
         zrd_adin(:,:,:)    = 0.0_wp

         !--------------------------------------------------------------------
         ! Initialize the tangent input with random noise: dx
         !--------------------------------------------------------------------
         CALL grid_random( znts(:,:,:,jp_tem), 'T', 0.0_wp, stdt )
         CALL grid_random( znts(:,:,:,jp_sal), 'T', 0.0_wp, stds )

         DO jk = 1, jpk
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zts_tlin(ji,jj,jk,jp_tem) = znts(ji,jj,jk,jp_tem)
                  zts_tlin(ji,jj,jk,jp_sal) = znts(ji,jj,jk,jp_sal)
               END DO
            END DO
         END DO
         CALL eos_insitu_tan(zts, zts_tlin, zrd_tlout)

         DO jk = 1, jpk
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zrd_adin(ji,jj,jk)   = zrd_tlout(ji,jj,jk) &
                       &                 * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,jk)&
                       &                 * tmask(ji,jj,jk)
               END DO
            END DO
         END DO

         !--------------------------------------------------------------------
         ! Compute the scalar product: ( L dx )^T W dy
         !--------------------------------------------------------------------

         zsp1 = DOT_PRODUCT( zrd_tlout, zrd_adin )

         !--------------------------------------------------------------------
         ! Call the adjoint routine: dx^* = L^T dy^*
         !--------------------------------------------------------------------
         CALL eos_insitu_adj(zts, zts_adout, zrd_adin)
         zsp2 = DOT_PRODUCT( zts_tlin(:,:,:,jp_tem), zts_adout(:,:,:,jp_tem) ) + &
                & DOT_PRODUCT( zts_tlin(:,:,:,jp_sal), zts_adout(:,:,:,jp_sal) )

         ! Compare the scalar products

         ! Compare the scalar products
         ! 14 char:'12345678901234'
         SELECT CASE( jn )
         CASE (0) ; cl_name = 'eos_adj ins T1'
         CASE (1) ; cl_name = 'eos_adj ins T2'
         CASE (2) ; cl_name = 'eos_adj ins T3'
         END SELECT
         CALL prntst_adj( cl_name, kumadt, zsp1, zsp2 )
      ENDDO
      ! restore initial nn_eos
      nn_eos = jeos

      ! Deallocate memory

      DEALLOCATE(      &
         & zts,       &
         & zts_adout,   &
         & zrd_adin,   &
         & zts_tlin,    &
         & zrd_tlout,  &
         & znts        &
         & )
   END SUBROUTINE eos_insitu_adj_tst

   SUBROUTINE eos_insitu_pot_adj_tst( kumadt )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE eos_adj_tst ***
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
      !! ** Action  : Separate tests are applied for the following dx and dy:
      !!
      !!              1) dx = ( SSH ) and dy = ( SSH )
      !!
      !! History :
      !!        ! 08-07 (A. Vidard)
      !!-----------------------------------------------------------------------
      !! * Modules used

      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt             ! Output unit
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         zts                        ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts_adout                ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zrd_adin                 ! anomaly density
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zrhop_adin               ! volume mass
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts_tlin                  ! potential temperature
                                     ! salinity
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & znts                     ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zrd_tlout                  ! anomaly density
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zrhop_tlout                  ! volume mass
      REAL(KIND=wp) ::       &
         & zsp1,             & ! scalar product involving the tangent routine
         & zsp2                ! scalar product involving the adjoint routine
      INTEGER :: &
         & ji, &
         & jj, &
         & jk, &
         & jn, &
         & jeos
      CHARACTER(LEN=14) :: cl_name

       ! Allocate memory
      ALLOCATE( &
         & zts(     jpi, jpj, jpk, jpts ),  &
         & zts_adout( jpi, jpj, jpk, jpts ),  &
         & zrhop_adin( jpi, jpj, jpk ),  &
         & zrd_adin( jpi, jpj, jpk ),  &
         & zts_tlin(  jpi, jpj, jpk, jpts ),  &
         & znts(      jpi, jpj, jpk, jpts ),  &
         & zrd_tlout(jpi, jpj, jpk ),  &
         & zrhop_tlout(jpi, jpj, jpk )    )

      ! Initialize random field standard deviationsthe reference state
      zts = tsn

      ! store initial nn_eos
      jeos = nn_eos
      DO jn = 0, 2
         nn_eos = jn
         !=============================================
         !  testing of eos_insitu_pot
         !=============================================

         !=============================================================
         ! 1) dx = ( T ) and dy = ( T )
         !=============================================================

         !--------------------------------------------------------------------
         ! Reset the tangent and adjoint variables
         !--------------------------------------------------------------------
         zts_tlin(:,:,:,:)     = 0.0_wp
         zrd_tlout(:,:,:)   = 0.0_wp
         zrhop_tlout(:,:,:) = 0.0_wp
         zts_adout(:,:,:,:)    = 0.0_wp
         zrhop_adin(:,:,:)  = 0.0_wp
         zrd_adin(:,:,:)    = 0.0_wp

         !--------------------------------------------------------------------
         ! Initialize the tangent input with random noise: dx
         !--------------------------------------------------------------------
         CALL grid_random( znts(:,:,:,jp_tem), 'T', 0.0_wp, stdt )
         CALL grid_random( znts(:,:,:,jp_sal), 'T', 0.0_wp, stds )
         DO jk = 1, jpk
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zts_tlin(ji,jj,jk,:) = znts(ji,jj,jk,:)
               END DO
            END DO
         END DO
         !--------------------------------------------------------------------
         ! Call the tangent routine: dy = L dx
         !--------------------------------------------------------------------

         call eos_insitu_pot_tan ( zts, zts_tlin, zrd_tlout, zrhop_tlout )

         !--------------------------------------------------------------------
         ! Initialize the adjoint variables: dy^* = W dy
         !--------------------------------------------------------------------
         DO jk = 1, jpk
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zrd_adin(ji,jj,jk)   = zrd_tlout(ji,jj,jk) &
                       &                 * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,jk)&
                       &                 * tmask(ji,jj,jk)
                  zrhop_adin(ji,jj,jk) = zrhop_tlout(ji,jj,jk) &
                       &                 * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,jk)&
                       &                 * tmask(ji,jj,jk)
               END DO
            END DO
         END DO

         !--------------------------------------------------------------------
         ! Compute the scalar product: ( L dx )^T W dy
         !--------------------------------------------------------------------

         zsp1 =  DOT_PRODUCT( zrd_tlout  , zrd_adin   ) &
              &  + DOT_PRODUCT( zrhop_tlout, zrhop_adin )
         !--------------------------------------------------------------------
         ! Call the adjoint routine: dx^* = L^T dy^*
         !--------------------------------------------------------------------

         CALL eos_insitu_pot_adj( zts, zts_adout, zrd_adin, zrhop_adin )
         !--------------------------------------------------------------------
         ! Compute the scalar product: dx^T L^T W dy
         !--------------------------------------------------------------------

         zsp2 = DOT_PRODUCT( zts_tlin(:,:,:,jp_tem), zts_adout(:,:,:,jp_tem) ) + &
                & DOT_PRODUCT( zts_tlin(:,:,:,jp_sal), zts_adout(:,:,:,jp_sal) )
         ! Compare the scalar products

         ! 14 char:'12345678901234'
         SELECT CASE( jn )
         CASE (0) ; cl_name = 'eos_adj pot T1'
         CASE (1) ; cl_name = 'eos_adj pot T2'
         CASE (2) ; cl_name = 'eos_adj pot T3'
         END SELECT
         CALL prntst_adj( cl_name, kumadt, zsp1, zsp2 )

      ENDDO

      ! restore initial nn_eos
      nn_eos = jeos

      ! Deallocate memory
      DEALLOCATE(       &
         & zts,       &
         & zts_adout,   &
         & zrd_adin,   &
         & zrhop_adin, &
         & zts_tlin,    &
         & zrd_tlout,  &
         & zrhop_tlout,&
         & znts    )
   END SUBROUTINE eos_insitu_pot_adj_tst

   SUBROUTINE eos_alpbet_adj_tst(kumadt)
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE eos_adj_tst ***
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
      !! ** Action  : Separate tests are applied for the following dx and dy:
      !!
      !!              1) dx = ( SSH ) and dy = ( SSH )
      !!
      !! History :
      !!        ! 05-12 (P.-A. Bouttier)
      !!-----------------------------------------------------------------------
      !! * Modules used

      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt             ! Output unit
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts                        ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts_adout                ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zpalpbet_adin
      REAL(wp)                       ::   &
         & zbeta0_adin
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts_tlin                  ! potential temperature
                                     ! salinity
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & znts                     ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zpalpbet_tlout
      REAL(wp)                       ::   &
         & zbeta0_tlout
      REAL(KIND=wp) ::       &
         & zsp1,             & ! scalar product involving the tangent routine
         & zsp2                ! scalar product involving the adjoint routine
      INTEGER :: &
         & ji, &
         & jj, &
         & jk, &
         & jn, &
         & jeos
      CHARACTER(LEN=14) :: cl_name

       ! Allocate memory
      ALLOCATE( &
         & zts(     jpi, jpj, jpk, jpts ),  &
         & zts_adout( jpi, jpj, jpk, jpts ),  &
         & zpalpbet_adin( jpi, jpj, jpk ),  &
         & zts_tlin(  jpi, jpj, jpk, jpts ),  &
         & znts(      jpi, jpj, jpk, jpts ),  &
         & zpalpbet_tlout(jpi, jpj, jpk ) )


      ! Initialize random field standard deviationsthe reference state
      zts = tsn

      ! store initial nn_eos
      jeos = nn_eos
      DO jn = 0, 2
         nn_eos = jn
         !=============================================
         !  testing of eos_insitu_pot
         !=============================================

         !=============================================================
         ! 1) dx = ( T ) and dy = ( T )
         !=============================================================

         !--------------------------------------------------------------------
         ! Reset the tangent and adjoint variables
         !--------------------------------------------------------------------
         zts_tlin(:,:,:,:)     = 0.0_wp
         zpalpbet_tlout(:,:,:)   = 0.0_wp
         zbeta0_tlout = 0.0_wp
         zts_adout(:,:,:,:)    = 0.0_wp
         zbeta0_adin  = 0.0_wp
         zpalpbet_adin(:,:,:)    = 0.0_wp

         !--------------------------------------------------------------------
         ! Initialize the tangent input with random noise: dx
         !--------------------------------------------------------------------
         CALL grid_random( znts(:,:,:,jp_tem), 'T', 0.0_wp, stdt )
         CALL grid_random( znts(:,:,:,jp_sal), 'T', 0.0_wp, stds )
         DO jk = 1, jpk
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zts_tlin(ji,jj,jk,:) = znts(ji,jj,jk,:)
               END DO
            END DO
         END DO
         !--------------------------------------------------------------------
         ! Call the tangent routine: dy = L dx
         !--------------------------------------------------------------------

         call eos_alpbet_tan ( zts, zts_tlin, zpalpbet_tlout, zbeta0_tlout )

         !--------------------------------------------------------------------
         ! Initialize the adjoint variables: dy^* = W dy
         !--------------------------------------------------------------------
         DO jk = 1, jpk
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zpalpbet_adin(ji,jj,jk)   = zpalpbet_tlout(ji,jj,jk) &
                       &                 * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,jk)&
                       &                 * tmask(ji,jj,jk)
               END DO
            END DO
         END DO
         zbeta0_adin = zbeta0_tlout

         !--------------------------------------------------------------------
         ! Compute the scalar product: ( L dx )^T W dy
         !--------------------------------------------------------------------

         zsp1 =  DOT_PRODUCT( zpalpbet_tlout  , zpalpbet_adin   ) &
              &  + zbeta0_tlout *  zbeta0_adin
         !--------------------------------------------------------------------
         ! Call the adjoint routine: dx^* = L^T dy^*
         !--------------------------------------------------------------------

         CALL eos_alpbet_adj( zts, zts_adout, zpalpbet_adin, zbeta0_adin )

         !--------------------------------------------------------------------
         ! Compute the scalar product: dx^T L^T W dy
         !--------------------------------------------------------------------

         zsp2 = DOT_PRODUCT( zts_tlin(:,:,:,jp_tem), zts_adout(:,:,:,jp_tem) ) + &
                & DOT_PRODUCT( zts_tlin(:,:,:,jp_sal), zts_adout(:,:,:,jp_sal) )
         ! Compare the scalar products

         ! 14 char:'12345678901234'
         SELECT CASE( jn )
         CASE (0) ; cl_name = 'eos_adj ab  T1'
         CASE (1) ; cl_name = 'eos_adj ab  T2'
         CASE (2) ; cl_name = 'eos_adj ab  T3'
         END SELECT
         CALL prntst_adj( cl_name, kumadt, zsp1, zsp2 )

      ENDDO

      ! restore initial nn_eos
      nn_eos = jeos

      ! Deallocate memory
      DEALLOCATE(       &
         & zts,       &
         & zts_adout,   &
         & zpalpbet_adin,   &
         & zts_tlin,    &
         & zpalpbet_tlout,  &
         & znts    )

   END SUBROUTINE eos_alpbet_adj_tst

   SUBROUTINE eos_insitu_2d_adj_tst( kumadt )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE eos_adj_tst ***
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
      !! ** Action  : Separate tests are applied for the following dx and dy:
      !!
      !!              1) dx = ( SSH ) and dy = ( SSH )
      !!
      !! History :
      !!        ! 08-07 (A. Vidard)
      !!-----------------------------------------------------------------------
      !! * Modules used

      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt             ! Output unit
      REAL(wp), DIMENSION(:,:), ALLOCATABLE ::   &
         zdep                       ! depth
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         zts                        ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zts_adout                ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:), ALLOCATABLE ::   &
         & zrd_adin                 ! anomaly density
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zts_tlin                 ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & znts                     ! potential temperature
                                    ! salinity
      REAL(wp), DIMENSION(:,:), ALLOCATABLE ::   &
         & zrd_tlout                  ! anomaly density
      REAL(KIND=wp) ::       &
         & zsp1,             & ! scalar product involving the tangent routine
         & zsp2                ! scalar product involving the adjoint routine
      INTEGER :: &
         & ji, &
         & jj, &
         & jn, &
         & jeos
      CHARACTER(LEN=14) :: cl_name
      ! Allocate memory

      ALLOCATE( &
         & zdep(     jpi, jpj),  &
         & zts(     jpi, jpj, jpts ),  &
         & znts(      jpi, jpj, jpts ),  &
         & zts_adout( jpi, jpj, jpts ),  &
         & zrd_adin( jpi, jpj ),  &
         & zts_tlin(  jpi, jpj, jpts ),  &
         & zrd_tlout(jpi, jpj )    )

      ! Initialize the reference state
      zts(:,:,:) = tsn(:,:,2,:)
      zdep(:,:) = fsdept(:,:,2)

      ! store initial nn_eos
      jeos = nn_eos
      DO jn = 0, 2
         nn_eos = jn
         !=============================================================
         ! 1) dx = ( T ) and dy = ( T )
         !=============================================================

         !--------------------------------------------------------------------
         ! Reset the tangent and adjoint variables
         !--------------------------------------------------------------------
         zts_tlin(:,:,:)     = 0.0_wp
         zrd_tlout(:,:)   = 0.0_wp
         zts_adout(:,:,:)    = 0.0_wp
         zrd_adin(:,:)    = 0.0_wp

         !--------------------------------------------------------------------
         ! Initialize the tangent input with random noise: dx
         !--------------------------------------------------------------------
         CALL grid_random( znts(:,:,jp_tem), 'T', 0.0_wp, stdt )
         CALL grid_random( znts(:,:,jp_sal), 'T', 0.0_wp, stds )
         DO jj = nldj, nlej
            DO ji = nldi, nlei
               zts_tlin(ji,jj,:) = znts(ji,jj,:)
            END DO
         END DO

         CALL eos_insitu_2d_tan(zts, zdep, zts_tlin, zrd_tlout)

         DO jj = nldj, nlej
            DO ji = nldi, nlei
               zrd_adin(ji,jj)   = zrd_tlout(ji,jj) &
                    &                 * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,2)&
                    &                 * tmask(ji,jj,2)
            END DO
         END DO

         !--------------------------------------------------------------------
         ! Compute the scalar product: ( L dx )^T W dy
         !--------------------------------------------------------------------

         zsp1 = DOT_PRODUCT( zrd_tlout, zrd_adin )

         !--------------------------------------------------------------------
         ! Call the adjoint routine: dx^* = L^T dy^*
         !--------------------------------------------------------------------
         CALL eos_insitu_2d_adj(zts, zdep, zts_adout, zrd_adin)
         zsp2 = DOT_PRODUCT( zts_tlin(:,:,jp_tem), zts_adout(:,:,jp_tem) ) + &
              & DOT_PRODUCT( zts_tlin(:,:,jp_sal), zts_adout(:,:,jp_sal) )

         ! Compare the scalar products

         ! 14 char:'12345678901234'
         SELECT CASE( jn )
         CASE (0) ; cl_name = 'eos_adj 2d  T1'
         CASE (1) ; cl_name = 'eos_adj 2d  T2'
         CASE (2) ; cl_name = 'eos_adj 2d  T3'
         END SELECT
         CALL prntst_adj( cl_name, kumadt, zsp1, zsp2 )

      ENDDO

      ! restore initial nn_eos
      nn_eos = jeos

      ! Deallocate memory

      DEALLOCATE(      &
         & zdep,       &
         & zts,       &
         & zts_adout,   &
         & zrd_adin,   &
         & zts_tlin,    &
         & zrd_tlout,  &
         & znts   )
   END SUBROUTINE eos_insitu_2d_adj_tst

   SUBROUTINE eos_adj_tst( kumadt )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE eos_adj_tst ***
      !!
      !! ** Purpose : Test the adjoint routine.
      !!
      !! History :
      !!        ! 08-07 (A. Vidard)
      !!-----------------------------------------------------------------------
      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt             ! Output unit

      CALL eos_insitu_adj_tst( kumadt )
      CALL eos_insitu_pot_adj_tst( kumadt )
      CALL eos_insitu_2d_adj_tst( kumadt )
      CALL eos_alpbet_adj_tst( kumadt )

   END SUBROUTINE eos_adj_tst

   SUBROUTINE bn2_adj_tst( kumadt )
      !!-----------------------------------------------------------------------
      !!
      !!                  ***  ROUTINE bn2_adj_tst ***
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
      !! ** Action  : Separate tests are applied for the following dx and dy:
      !!
      !!              1) dx = ( SSH ) and dy = ( SSH )
      !!
      !! History :
      !!        ! 08-07 (A. Vidard)
      !!-----------------------------------------------------------------------
      !! * Modules used

      !! * Arguments
      INTEGER, INTENT(IN) :: &
         & kumadt             ! Output unit
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         zts                      ! potential temperature
                                  ! salinity
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts_adout              ! potential temperature
                                  ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zrd_adin,               &  ! potential density (surface referenced)
         & zrd_adout                  ! potential density (surface referenced)
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & zts_tlin,              &  ! potential temperature
                                     ! salinity
         & zts_tlout                 ! potential temperature
                                     ! salinity
      REAL(wp), DIMENSION(:,:,:), ALLOCATABLE ::   &
         & zrd_tlout                  ! potential density (surface referenced)
      REAL(KIND=wp) ::       &
         & zsp1,             & ! scalar product involving the tangent routine
         & zsp2                ! scalar product involving the adjoint routine
      REAL(wp), DIMENSION(:,:,:,:), ALLOCATABLE ::   &
         & znts                   ! potential temperature
                                  ! salinity
      INTEGER :: &
         & ji, &
         & jj, &
         & jk, &
         & jn, &
         & jeos
      CHARACTER(LEN=14) :: cl_name

      ! Allocate memory
      ALLOCATE( &
         & zts(     jpi, jpj, jpk, jpts ),  &
         & zts_adout( jpi, jpj, jpk, jpts ),  &
         & zrd_adin( jpi, jpj, jpk ),  &
         & zrd_adout(jpi, jpj, jpk ),  &
         & zts_tlin(  jpi, jpj, jpk, jpts ),  &
         & znts(      jpi, jpj, jpk, jpts ),  &
         & zts_tlout( jpi, jpj, jpk, jpts ),  &
         & zrd_tlout(jpi, jpj, jpk )    )

      ! Initialize random field standard deviationsthe reference state
      zts = tsn
      ! store initial nn_eos
      jeos = nn_eos
      DO jn = 0, 2
         nn_eos = jn
         !=============================================================
         ! 1) dx = ( T ) and dy = ( T )
         !=============================================================

         !--------------------------------------------------------------------
         ! Reset the tangent and adjoint variables
         !--------------------------------------------------------------------
         zts_tlin(:,:,:,:) = 0.0_wp
         zts_tlout(:,:,:,:) = 0.0_wp
         zrd_tlout(:,:,:) = 0.0_wp
         zts_adout(:,:,:,:) = 0.0_wp
         zrd_adin(:,:,:) = 0.0_wp
         zrd_adout(:,:,:) = 0.0_wp

         !--------------------------------------------------------------------
         ! Initialize the tangent input with random noise: dx
         !--------------------------------------------------------------------
         CALL grid_random( znts(:,:,:,jp_tem), 'T', 0.0_wp, stdt )
         CALL grid_random( znts(:,:,:,jp_sal), 'T', 0.0_wp, stds )
         DO jk = 1, jpk
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zts_tlin(ji,jj,jk,:) = znts(ji,jj,jk,:)
               END DO
            END DO
         END DO
         !--------------------------------------------------------------------
         ! Call the tangent routine: dy = L dx
         !--------------------------------------------------------------------
         zts_tlout(:,:,:,:) = zts_tlin
         CALL eos_bn2_tan( zts, zts_tlout, zrd_tlout )
         !--------------------------------------------------------------------
         ! Initialize the adjoint variables: dy^* = W dy
         !--------------------------------------------------------------------
         DO jk = 1, jpk
            DO jj = nldj, nlej
               DO ji = nldi, nlei
                  zrd_adin(ji,jj,jk) = zrd_tlout(ji,jj,jk) &
                       &               * e1t(ji,jj) * e2t(ji,jj) * fse3t(ji,jj,jk) &
                       &               * tmask(ji,jj,jk)
               END DO
            END DO
         END DO

         !--------------------------------------------------------------------
         ! Compute the scalar product: ( L dx )^T W dy
         !--------------------------------------------------------------------

         zsp1 = DOT_PRODUCT( zrd_tlout, zrd_adin )

         !--------------------------------------------------------------------
         ! Call the adjoint routine: dx^* = L^T dy^*
         !--------------------------------------------------------------------

         zrd_adout(:,:,:) = zrd_adin(:,:,:)
         CALL eos_bn2_adj( zts, zts_adout, zrd_adout )

         !--------------------------------------------------------------------
         ! Compute the scalar product: dx^T L^T W dy
         !--------------------------------------------------------------------
         zsp2 = DOT_PRODUCT( zts_tlin(:,:,:,jp_tem), zts_adout(:,:,:,jp_tem )) + &
              & DOT_PRODUCT( zts_tlin(:,:,:,jp_sal), zts_adout(:,:,:,jp_sal ))

         ! Compare the scalar products
         ! 14 char:'12345678901234'
         SELECT CASE( jn )
         CASE (0) ; cl_name = 'bn2_adj     T1'
         CASE (1) ; cl_name = 'bn2_adj     T2'
         CASE (2) ; cl_name = 'bn2_adj     T3'
         END SELECT
         CALL prntst_adj( cl_name, kumadt, zsp1, zsp2 )
      ENDDO
      ! restore initial nn_eos
      nn_eos = jeos

      ! Deallocate memory
      DEALLOCATE(      &
         & zts,       &
         & zts_adout,   &
         & zrd_adin,   &
         & zrd_adout,  &
         & zts_tlin,    &
         & zts_tlout,   &
         & zrd_tlout,  &
         & znts    )
   END SUBROUTINE bn2_adj_tst
#endif
   !!======================================================================
END MODULE eosbn2_tam
