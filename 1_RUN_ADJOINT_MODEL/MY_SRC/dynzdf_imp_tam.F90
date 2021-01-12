!!! 20191004I - by SAM - essential modifications throughout for NEMOTAM operation
! See details and diffs at http://forge.ipsl.jussieu.fr/nemo/attachment/ticket/1362/dynzdf_imp_tam.F90.diff
!!! /20191004I
MODULE dynzdf_imp_tam
#if defined key_tam
   !!==============================================================================
   !!                     ***  MODULE  dynzdf_imp_tam  ***
   !! Ocean dynamics:  vertical component(s) of the momentum mixing trend
   !!                 Tangent and Adjoint Module
   !!==============================================================================
   !! History of the direct module:
   !!                !  90-10  (B. Blanke)  Original code
   !!                !  97-05  (G. Madec)  vertical component of isopycnal
   !!           8.5  !  02-08  (G. Madec)  F90: Free form and module
   !! History of the TAM module:
   !!           9.0  !  09-01  (A. Vidard) TAM of the 02-08 version
   !!----------------------------------------------------------------------

   !!----------------------------------------------------------------------
   !!   dyn_zdf_imp  : update the momentum trend with the vertical diffu-
   !!                  sion using an implicit time-stepping scheme.
   !!----------------------------------------------------------------------
   !! * Modules used
   USE par_oce
   USE oce_tam
   USE zdf_oce
   USE dom_oce
   USE phycst
   USE in_out_manager
   USE lib_mpp         ! MPP library
   USE zdfbfr          ! Bottom friction setup
   USE wrk_nemo        ! Memory Allocation
   USE timing          ! Timing

   IMPLICIT NONE
   PRIVATE

   !! * Routine accessibility
   PUBLIC dyn_zdf_imp_tan     ! called by dynzdf_tam.F90
   PUBLIC dyn_zdf_imp_adj     ! called by dynzdf_tam.F90

   !! * Substitutions
#  include "domzgr_substitute.h90"
#  include "vectopt_loop_substitute.h90"
   !!----------------------------------------------------------------------

CONTAINS

   SUBROUTINE dyn_zdf_imp_tan( kt, p2dt )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dyn_zdf_imp_tan  ***
      !!
      !! ** Purpose of the direct routine:
      !!      Compute the trend due to the vert. momentum diffusion
      !!      and the surface forcing, and add it to the general trend of
      !!      the momentum equations.
      !!
      !! ** Method of the direct routine:
      !!      The vertical momentum mixing trend is given by :
      !!             dz( avmu dz(u) ) = 1/e3u dk+1( avmu/e3uw dk(ua) )
      !!      backward time stepping
      !!      Surface boundary conditions: wind stress input
      !!      Bottom boundary conditions : bottom stress (cf zdfbfr.F)
      !!      Add this trend to the general trend ua :
      !!         ua = ua + dz( avmu dz(u) )E
      !!
      !! ** Action : - Update (ua,va) arrays with the after vertical diffusive
      !!               mixing trend.
      !!---------------------------------------------------------------------
      !! * Modules used
      !! * Arguments
      INTEGER , INTENT( in ) ::   kt                           ! ocean time-step index
      REAL(wp), INTENT( in ) ::   p2dt                         ! time-step

      !! * Local declarations
      INTEGER ::   ji, jj, jk, ikbu, ikbv                          ! dummy loop indices
      REAL(wp) ::  z1_p2dt, z2dtf, zcoef, zzws, zrhstl ! temporary scalars
      REAL(wp), POINTER, DIMENSION(:,:,:):: zwi, zws, zwd ! temporary workspace arrays
      REAL(wp), POINTER, DIMENSION(:,:):: zavmu, zavmv      ! temporary workspace arrays
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )  CALL timing_start('dyn_zdf_imp_tan')
      !
      CALL wrk_alloc( jpi,jpj,jpk, zwi, zwd, zws )
      CALL wrk_alloc( jpi,jpj, zavmu, zavmv)
      !
      IF( kt == nit000 ) THEN
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) 'dyn_zdf_imp_tan : vertical momentum diffusion explicit operator'
         IF(lwp) WRITE(numout,*) '~~~~~~~~~~~~~~~ '
      ENDIF
      ! 0. Local constant initialization
      ! --------------------------------
      z1_p2dt = 1._wp / p2dt      ! inverse of the timestep

      ! 1. Apply semi-implicit bottom friction
      ! --------------------------------------
      ! Only needed for semi-implicit bottom friction setup. The explicit
      ! bottom friction has been included in "u(v)a" which act as the R.H.S
      ! column vector of the tri-diagonal matrix equation
      !
      IF( ln_bfrimp ) THEN
         !!!!!!!!!!!!!!!!!!!!!!!!!!
         ! avm* are unactivated for the current TAM
         !!!!!!!!!!!!!!!!!!!!!!!!!!
# if defined key_vectopt_loop
      DO jj = 1, 1
         DO ji = jpi+2, jpij-jpi-1   ! vector opt. (forced unrolling)
# else
      DO jj = 2, jpjm1
         DO ji = 2, jpim1
# endif
            ikbu = mbku(ji,jj)         ! ocean bottom level at u- and v-points
            ikbv = mbkv(ji,jj)         ! (deepest ocean u- and v-points)
            zavmu(ji,jj) = avmu(ji,jj,ikbu+1)
            zavmv(ji,jj) = avmv(ji,jj,ikbv+1)
            avmu(ji,jj,ikbu+1) = -bfrua(ji,jj) * fse3uw(ji,jj,ikbu+1)
            avmv(ji,jj,ikbv+1) = -bfrva(ji,jj) * fse3vw(ji,jj,ikbv+1)
         END DO
      END DO

      ENDIF

      ! 2. Vertical diffusion on u
      ! ---------------------------
      ! Matrix and second member construction
      ! bottom boundary condition: both zwi and zws must be masked as avmu can take
      ! non zero value at the ocean bottom depending on the bottom friction used.
      DO jk = 1, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zcoef = - p2dt / fse3u(ji,jj,jk)
               zwi(ji,jj,jk) = zcoef * avmu(ji,jj,jk  ) / fse3uw(ji,jj,jk  ) * umask(ji,jj,jk)
               zzws          = zcoef * avmu(ji,jj,jk+1) / fse3uw(ji,jj,jk+1)
               zws(ji,jj,jk) = zzws  * umask(ji,jj,jk+1)
               zwd(ji,jj,jk) = 1._wp - zwi(ji,jj,jk) - zzws
            END DO
         END DO
      END DO

      ! Surface boudary conditions
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            zwi(ji,jj,1) = 0.0_wp
            zwd(ji,jj,1) = 1._wp - zws(ji,jj,1)
         END DO
      END DO

      ! Matrix inversion starting from the first level
      !-----------------------------------------------------------------------
      !   solve m.x = y  where m is a tri diagonal matrix ( jpk*jpk )
      !
      !        ( zwd1 zws1   0    0    0  )( zwx1 ) ( zwy1 )
      !        ( zwi2 zwd2 zws2   0    0  )( zwx2 ) ( zwy2 )
      !        (  0   zwi3 zwd3 zws3   0  )( zwx3 )=( zwy3 )
      !        (        ...               )( ...  ) ( ...  )
      !        (  0    0    0   zwik zwdk )( zwxk ) ( zwyk )
      !
      !   m is decomposed in the product of an upper and a lower triangular matrix
      !   The 3 diagonal terms are in 2d arrays: zwd, zws, zwi
      !   The solution (the after velocity) is in ua
      !-----------------------------------------------------------------------

      ! First recurrence : Dk = Dk - Lk * Uk-1 / Dk-1   (increasing k)
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zwd(ji,jj,jk) = zwd(ji,jj,jk) - zwi(ji,jj,jk) * zws(ji,jj,jk-1) / zwd(ji,jj,jk-1)
            END DO
         END DO
      END DO

      ! second recurrence:    SOLk = RHSk - Lk / Dk-1  Lk-1
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            ua_tl(ji,jj,1) = ub_tl(ji,jj,1)  &
                           + p2dt * ua_tl(ji,jj,1)
         END DO
      END DO
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zrhstl          = ub_tl(ji,jj,jk) + p2dt * ua_tl(ji,jj,jk)   ! zrhs=right hand side
               ua_tl(ji,jj,jk) = zrhstl - zwi(ji,jj,jk) / zwd(ji,jj,jk-1) * ua_tl(ji,jj,jk-1)
            END DO
         END DO
      END DO

      ! thrid recurrence : SOLk = ( Lk - Uk * Ek+1 ) / Dk
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            ua_tl(ji,jj,jpkm1) = ua_tl(ji,jj,jpkm1) / zwd(ji,jj,jpkm1)
         END DO
      END DO
      DO jk = jpk-2, 1, -1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               ua_tl(ji,jj,jk) = ( ua_tl(ji,jj,jk) - zws(ji,jj,jk) * ua_tl(ji,jj,jk+1) ) / zwd(ji,jj,jk)
            END DO
         END DO
      END DO

      ! Normalization to obtain the general momentum trend ua
      DO jk = 1, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               ua_tl(ji,jj,jk) = ( ua_tl(ji,jj,jk) - ub_tl(ji,jj,jk) ) * z1_p2dt
            END DO
         END DO
      END DO

      ! 2. Vertical diffusion on v
      ! ---------------------------
      ! Matrix and second member construction
      ! bottom boundary condition: both zwi and zws must be masked as avmv can take
      ! non zero value at the ocean bottom depending on the bottom friction
      ! used but the bottom velocities have already been updated with the bottom
      ! friction velocity in dyn_bfr using values from the previous timestep. There
      ! is no need to include these in the implicit calculation.
      DO jk = 1, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zcoef          = -p2dt / fse3v(ji,jj,jk)
               zwi(ji,jj,jk) = zcoef * avmv(ji,jj,jk  ) / fse3vw(ji,jj,jk  ) * vmask(ji,jj,jk)
               zzws          = zcoef * avmv(ji,jj,jk+1) / fse3vw(ji,jj,jk+1)
               zws(ji,jj,jk) =  zzws * vmask(ji,jj,jk+1)
               zwd(ji,jj,jk) = 1._wp - zwi(ji,jj,jk) - zzws
            END DO
         END DO
      END DO

      ! Surface boudary conditions
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            zwi(ji,jj,1) = 0._wp
            zwd(ji,jj,1) = 1._wp - zws(ji,jj,1)
         END DO
      END DO

      ! Matrix inversion
      !-----------------------------------------------------------------------
      !   solve m.x = y  where m is a tri diagonal matrix ( jpk*jpk )
      !
      !        ( zwd1 zws1   0    0    0  )( zwx1 ) ( zwy1 )
      !        ( zwi2 zwd2 zws2   0    0  )( zwx2 ) ( zwy2 )
      !        (  0   zwi3 zwd3 zws3   0  )( zwx3 )=( zwy3 )
      !        (        ...               )( ...  ) ( ...  )
      !        (  0    0    0   zwik zwdk )( zwxk ) ( zwyk )
      !
      !   m is decomposed in the product of an upper and lower triangular
      !   matrix
      !   The 3 diagonal terms are in 2d arrays: zwd, zws, zwi
      !   The solution (after velocity) is in 2d array va
      !-----------------------------------------------------------------------

      ! First recurrence : Dk = Dk - Lk * Uk-1 / Dk-1   (increasing k)
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zwd(ji,jj,jk) = zwd(ji,jj,jk) - zwi(ji,jj,jk) * zws(ji,jj,jk-1) / zwd(ji,jj,jk-1)
            END DO
         END DO
      END DO

      ! second recurrence:    SOLk = RHSk - Lk / Dk-1  Lk-1
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            va_tl(ji,jj,1) = vb_tl(ji,jj,1)  &
                           + p2dt * va_tl(ji,jj,1)
         END DO
      END DO
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zrhstl          = vb_tl(ji,jj,jk) + p2dt * va_tl(ji,jj,jk)   ! zrhs=right hand side
               va_tl(ji,jj,jk) = zrhstl - zwi(ji,jj,jk) / zwd(ji,jj,jk-1) * va_tl(ji,jj,jk-1)
            END DO
         END DO
      END DO

      ! thrid recurrence : SOLk = ( Lk - Uk * SOLk+1 ) / Dk
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            va_tl(ji,jj,jpkm1) = va_tl(ji,jj,jpkm1) / zwd(ji,jj,jpkm1)
         END DO
      END DO
      DO jk = jpk-2, 1, -1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               va_tl(ji,jj,jk) = ( va_tl(ji,jj,jk) - zws(ji,jj,jk) * va_tl(ji,jj,jk+1) ) / zwd(ji,jj,jk)
            END DO
         END DO
      END DO

      ! Normalization to obtain the general momentum trend va
      DO jk = 1, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               va_tl(ji,jj,jk) = ( va_tl(ji,jj,jk) - vb_tl(ji,jj,jk) ) * z1_p2dt
            END DO
         END DO
      END DO

      !! restore bottom layer avmu(v)
      IF( ln_bfrimp ) THEN
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
         ! avm* are unactivated in the current TAM
         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# if defined key_vectopt_loop
      DO jj = 1, 1
         DO ji = jpi+2, jpij-jpi-1   ! vector opt. (forced unrolling)
# else
      DO jj = 2, jpjm1
         DO ji = 2, jpim1
# endif
            ikbu = mbku(ji,jj)         ! ocean bottom level at u- and v-points
            ikbv = mbkv(ji,jj)         ! (deepest ocean u- and v-points)
            avmu(ji,jj,ikbu+1) = zavmu(ji,jj)
            avmv(ji,jj,ikbv+1) = zavmv(ji,jj)
         END DO
      END DO
      ENDIF
      !
      CALL wrk_dealloc( jpi,jpj,jpk, zwi, zwd, zws)
      CALL wrk_dealloc( jpi,jpj, zavmu, zavmv)
      !
      IF( nn_timing == 1 )  CALL timing_stop('dyn_zdf_imp_tan')
      !
   END SUBROUTINE dyn_zdf_imp_tan
   SUBROUTINE dyn_zdf_imp_adj( kt, p2dt )
      !!----------------------------------------------------------------------
      !!                  ***  ROUTINE dyn_zdf_imp_adj  ***
      !!
      !! ** Purpose of the direct routine:
      !!      Compute the trend due to the vert. momentum diffusion
      !!      and the surface forcing, and add it to the general trend of
      !!      the momentum equations.
      !!
      !! ** Method of the direct routine:
      !!      The vertical momentum mixing trend is given by :
      !!             dz( avmu dz(u) ) = 1/e3u dk+1( avmu/e3uw dk(ua) )
      !!      backward time stepping
      !!      Surface boundary conditions: wind stress input
      !!      Bottom boundary conditions : bottom stress (cf zdfbfr.F)
      !!      Add this trend to the general trend ua :
      !!         ua = ua + dz( avmu dz(u) )E
      !!
      !! ** Action : - Update (ua,va) arrays with the after vertical diffusive
      !!               mixing trend.
      !!---------------------------------------------------------------------
      !! * Modules used
      !! * Arguments
      INTEGER , INTENT( in ) ::   kt                           ! ocean time-step index
      REAL(wp), INTENT( in ) ::   p2dt                         ! time-step

      !! * Local declarations
      !! * Local declarations
      INTEGER ::   ji, jj, jk, ikbu, ikbv                          ! dummy loop indices
      REAL(wp) ::   z1_p2dt, z2dtf, zcoef, zzws, zrhsad ! temporary scalars
      REAL(wp), POINTER, DIMENSION(:,:,:) :: zwi, zws, zwd! temporary workspace arrays
      REAL(wp), POINTER, DIMENSION(:,:):: zavmu, zavmv      ! temporary workspace arrays
      !!----------------------------------------------------------------------
      !
      IF( nn_timing == 1 )  CALL timing_start('dyn_zdf_imp_adj')
      !
      CALL wrk_alloc( jpi,jpj,jpk, zwi, zwd, zws )
      CALL wrk_alloc( jpi,jpj, zavmu, zavmv )
      !
      IF( kt == nitend ) THEN
         IF(lwp) WRITE(numout,*)
         IF(lwp) WRITE(numout,*) 'dyn_zdf_imp_adj : vertical momentum diffusion explicit operator'
         IF(lwp) WRITE(numout,*) '~~~~~~~~~~~~~~~ '
      ENDIF
      ! 0. Local constant initialization
      ! --------------------------------
      z1_p2dt = 1._wp / p2dt      ! inverse of the timestep
      zrhsad = 0.0_wp
      !
      !
      ! 1. Apply semi-implicit bottom friction
      ! --------------------------------------
      ! Only needed for semi-implicit bottom friction setup. The explicit
      ! bottom friction has been included in "u(v)a" which act as the R.H.S
      ! column vector of the tri-diagonal matrix equation
      !
      IF( ln_bfrimp ) THEN
# if defined key_vectopt_loop
      DO jj = 1, 1
         DO ji = jpi+2, jpij-jpi-1   ! vector opt. (forced unrolling)
# else
      DO jj = 2, jpjm1
         DO ji = 2, jpim1
# endif
            ikbu = mbku(ji,jj)         ! ocean bottom level at u- and v-points
            ikbv = mbkv(ji,jj)         ! (deepest ocean u- and v-points)
            zavmu(ji,jj) = avmu(ji,jj,ikbu+1)
            zavmv(ji,jj) = avmv(ji,jj,ikbv+1)
            avmu(ji,jj,ikbu+1) = -bfrua(ji,jj) * fse3uw(ji,jj,ikbu+1)
            avmv(ji,jj,ikbv+1) = -bfrva(ji,jj) * fse3vw(ji,jj,ikbv+1)
         END DO
      END DO
      ENDIF
      !
      ! 2. Vertical diffusion on v
      ! ---------------------------
      ! Matrix and second member construction
      ! bottom boundary condition: both zwi and zws must be masked as avmv can take
      ! non zero value at the ocean bottom depending on the bottom friction
      ! used but the bottom velocities have already been updated with the bottom
      ! friction velocity in dyn_bfr using values from the previous timestep. There
      ! is no need to include these in the implicit calculation.
      DO jk = 1, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zcoef         = -p2dt / fse3v(ji,jj,jk)
               zwi(ji,jj,jk) = zcoef * avmv(ji,jj,jk  ) / fse3vw(ji,jj,jk  ) * vmask(ji,jj,jk)
               zzws          = zcoef * avmv(ji,jj,jk+1) / fse3vw(ji,jj,jk+1)
               zws(ji,jj,jk) =  zzws * vmask(ji,jj,jk+1)
               zwd(ji,jj,jk) = 1._wp - zwi(ji,jj,jk) - zzws
            END DO
         END DO
      END DO

      ! Surface boudary conditions
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            zwi(ji,jj,1) = 0._wp
            zwd(ji,jj,1) = 1._wp - zws(ji,jj,1)
         END DO
      END DO

      ! Matrix inversion
      !-----------------------------------------------------------------------
      !   solve m.x = y  where m is a tri diagonal matrix ( jpk*jpk )
      !
      !        ( zwd1 zws1   0    0    0  )( zwx1 ) ( zwy1 )
      !        ( zwi2 zwd2 zws2   0    0  )( zwx2 ) ( zwy2 )
      !        (  0   zwi3 zwd3 zws3   0  )( zwx3 )=( zwy3 )
      !        (        ...               )( ...  ) ( ...  )
      !        (  0    0    0   zwik zwdk )( zwxk ) ( zwyk )
      !
      !   m is decomposed in the product of an upper and lower triangular
      !   matrix
      !   The 3 diagonal terms are in 2d arrays: zwd, zws, zwi
      !   The solution (after velocity) is in 2d array va
      !-----------------------------------------------------------------------

      ! First recurrence : Dk = Dk - Lk * Uk-1 / Dk-1   (increasing k)
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zwd(ji,jj,jk) = zwd(ji,jj,jk) - zwi(ji,jj,jk) * zws(ji,jj,jk-1) / zwd(ji,jj,jk-1)
            END DO
         END DO
      END DO

      ! Normalization to obtain the general momentum trend va
      DO jk = jpkm1, 1, -1
         DO jj = jpjm1, 2, -1
            DO ji = fs_jpim1, fs_2, -1   ! vector opt.
               vb_ad(ji,jj,jk) = vb_ad(ji,jj,jk) - va_ad(ji,jj,jk) / p2dt
               va_ad(ji,jj,jk) = va_ad(ji,jj,jk) / p2dt
            END DO
         END DO
      END DO
      ! thrid recurrence : SOLk = ( Lk - Uk * SOLk+1 ) / Dk
      DO jk = 1, jpk-2
         DO jj = jpjm1, 2, -1
            DO ji = fs_jpim1, fs_2, -1   ! vector opt.
               va_ad(ji,jj,jk+1) = va_ad(ji,jj,jk+1) - zws(ji,jj,jk) * va_ad(ji,jj,jk) / zwd(ji,jj,jk)
               va_ad(ji,jj,jk  ) = va_ad(ji,jj,jk  ) / zwd(ji,jj,jk)
            END DO
         END DO
      END DO
      DO jj = jpjm1, 2, -1
         DO ji = fs_jpim1, fs_2, -1   ! vector opt.
            va_ad(ji,jj,jpkm1) = va_ad(ji,jj,jpkm1) / zwd(ji,jj,jpkm1)
         END DO
      END DO
      ! second recurrence:    SOLk = RHSk - Lk / Dk-1  Lk-1
      DO jk = jpkm1, 2, -1
         DO jj = jpjm1, 2, -1
            DO ji = fs_jpim1, fs_2, -1   ! vector opt.
               zrhsad            = zrhsad + va_ad(ji,jj,jk)
               va_ad(ji,jj,jk-1) = va_ad(ji,jj,jk-1) - zwi(ji,jj,jk) / zwd(ji,jj,jk-1) * va_ad(ji,jj,jk)
               va_ad(ji,jj,jk  ) = 0.0_wp
               vb_ad(ji,jj,jk)   = vb_ad(ji,jj,jk) + zrhsad
               va_ad(ji,jj,jk)   = va_ad(ji,jj,jk) + p2dt * zrhsad
               zrhsad            = 0.0_wp
            END DO
         END DO
      END DO
      DO jj = jpjm1, 2, -1
         DO ji = fs_jpim1, fs_2, -1   ! vector opt.
            vb_ad(ji,jj,1) = vb_ad(ji,jj,1) + va_ad(ji,jj,1)
            va_ad(ji,jj,1) = va_ad(ji,jj,1) * p2dt
         END DO
      END DO

      ! 1. Vertical diffusion on u
      ! ---------------------------
      ! Matrix and second member construction
      ! bottom boundary condition: both zwi and zws must be masked as avmu can take
      ! non zero value at the ocean bottom depending on the bottom friction
      ! used but the bottom velocities have already been updated with the bottom
      ! friction velocity in dyn_bfr using values from the previous timestep. There
      ! is no need to include these in the implicit calculation.
      DO jk = 1, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zcoef = - p2dt / fse3u(ji,jj,jk)
               zwi(ji,jj,jk) = zcoef * avmu(ji,jj,jk  ) / fse3uw(ji,jj,jk  ) * umask(ji,jj,jk)
               zzws          = zcoef * avmu(ji,jj,jk+1) / fse3uw(ji,jj,jk+1)
               zws(ji,jj,jk) = zzws  * umask(ji,jj,jk+1)
               zwd(ji,jj,jk) = 1._wp - zwi(ji,jj,jk) - zzws
            END DO
         END DO
      END DO

      ! Surface boudary conditions
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            zwi(ji,jj,1) = 0._wp
            zwd(ji,jj,1) = 1._wp - zws(ji,jj,1)
         END DO
      END DO

      ! Matrix inversion starting from the first level
      !-----------------------------------------------------------------------
      !   solve m.x = y  where m is a tri diagonal matrix ( jpk*jpk )
      !
      !        ( zwd1 zws1   0    0    0  )( zwx1 ) ( zwy1 )
      !        ( zwi2 zwd2 zws2   0    0  )( zwx2 ) ( zwy2 )
      !        (  0   zwi3 zwd3 zws3   0  )( zwx3 )=( zwy3 )
      !        (        ...               )( ...  ) ( ...  )
      !        (  0    0    0   zwik zwdk )( zwxk ) ( zwyk )
      !
      !   m is decomposed in the product of an upper and a lower triangular matrix
      !   The 3 diagonal terms are in 2d arrays: zwd, zws, zwi
      !   The solution (the after velocity) is in ua
      !-----------------------------------------------------------------------

     ! First recurrence : Dk = Dk - Lk * Uk-1 / Dk-1   (increasing k)
      DO jk = 2, jpkm1
         DO jj = 2, jpjm1
            DO ji = fs_2, fs_jpim1   ! vector opt.
               zwd(ji,jj,jk) = zwd(ji,jj,jk) - zwi(ji,jj,jk) * zws(ji,jj,jk-1) / zwd(ji,jj,jk-1)
            END DO
         END DO
      END DO
      ! Normalization to obtain the general momentum trend ua
      DO jk = jpkm1, 1, -1
         DO jj = jpjm1, 2, -1
            DO ji = fs_jpim1, fs_2, -1   ! vector opt.
               ub_ad(ji,jj,jk) = ub_ad(ji,jj,jk) - ua_ad(ji,jj,jk) / p2dt
               ua_ad(ji,jj,jk) = ua_ad(ji,jj,jk) / p2dt
            END DO
         END DO
      END DO
      ! thrid recurrence : SOLk = ( Lk - Uk * Ek+1 ) / Dk
      DO jk = 1, jpk-2
         DO jj = jpjm1, 2, -1
            DO ji = fs_jpim1, fs_2, -1   ! vector opt.
               ua_ad(ji,jj,jk+1) = ua_ad(ji,jj,jk+1) - ua_ad(ji,jj,jk) * zws(ji,jj,jk) / zwd(ji,jj,jk)
               ua_ad(ji,jj,jk)   = ua_ad(ji,jj,jk) / zwd(ji,jj,jk)
            END DO
         END DO
      END DO
      DO jj = jpjm1, 2, -1
         DO ji = fs_jpim1, fs_2, -1
            ua_ad(ji,jj,jpkm1) = ua_ad(ji,jj,jpkm1) / zwd(ji,jj,jpkm1)
         END DO
      END DO
      DO jk = jpkm1, 2, -1
         DO jj = jpjm1, 2, -1
            DO ji = fs_jpim1, fs_2, -1   ! vector opt.
               zrhsad            = zrhsad + ua_ad(ji,jj,jk)
               ua_ad(ji,jj,jk-1) = ua_ad(ji,jj,jk-1) - zwi(ji,jj,jk) / zwd(ji,jj,jk-1) * ua_ad(ji,jj,jk)
               ua_ad(ji,jj,jk)   = 0.0_wp
               ub_ad(ji,jj,jk)   = ub_ad(ji,jj,jk) + zrhsad
               ua_ad(ji,jj,jk)   = ua_ad(ji,jj,jk) + zrhsad * p2dt
               zrhsad            = 0.0_wp
            END DO
         END DO
      END DO
      ! second recurrence:    SOLk = RHSk - Lk / Dk-1  Lk-1
      DO jj = 2, jpjm1
         DO ji = fs_2, fs_jpim1   ! vector opt.
            ub_ad(ji,jj,1) = ub_ad(ji,jj,1) + ua_ad(ji,jj,1)
            ua_ad(ji,jj,1) = p2dt * ua_ad(ji,jj,1)
         END DO
      END DO
      !! restore bottom layer avmu(v)
      IF( ln_bfrimp ) THEN
# if defined key_vectopt_loop
      DO jj = 1, 1
         DO ji = jpi+2, jpij-jpi-1   ! vector opt. (forced unrolling)
# else
      DO jj = 2, jpjm1
         DO ji = 2, jpim1
# endif
            ikbu = mbku(ji,jj)         ! ocean bottom level at u- and v-points
            ikbv = mbkv(ji,jj)         ! (deepest ocean u- and v-points)
            avmu(ji,jj,ikbu+1) = zavmu(ji,jj)
            avmv(ji,jj,ikbv+1) = zavmv(ji,jj)
         END DO
      END DO
      ENDIF
      !
      CALL wrk_dealloc( jpi,jpj,jpk, zwi, zwd, zws)
      CALL wrk_dealloc( jpi,jpj, zavmu, zavmv)
      !
      IF( nn_timing == 1 )  CALL timing_stop('dyn_zdf_imp_adj')
      !
   END SUBROUTINE dyn_zdf_imp_adj
#endif
   !!==============================================================================
END MODULE dynzdf_imp_tam
