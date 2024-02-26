#!/bin/bash

echo Initial size
read size
echo Max Iteration
read ITE
echo Lagrange multiplier
read LAGMUL

declare -i ITE2=$ITE*30

mpiexec -n 1 speight -tao_view -tao_smonitor -tao_ls_view -tao_ls_max_funcs 100 -tao_ls_rtol 1.e-3 -tao_ls_ftol 0.1 -tao_ls_stepmax 0.1 -tao_gatol 1.e-4 -tao_type lmvm -tao_max_funcs 400000 -tao_max_it $ITE -mx 801 -my 801 -lambda $size -paramlag $LAGMUL -parc0 0.26 -parc2 0.5 -parc4 1.0 -h 0.1 -paramnewlag 0.0 -itrmax $ITE2 #-tao_ls_stepmax 0.00001 -tao_ls_ftol 0.00001-start_in_debugger #-tao_ls_type armijo -tao_ls_armijo_beta 0.1 #-tao_ls_stepinit 0.0001 #-periodic #-tao_ls_type unit -tao_ls_stepinit 0.01 -periodic # -tao_bnk_as_type bertsekas -tao_bnk_as_step 0.01 -update_type step -tao_bnk_eta1 0.1 -tao_bnk_eta2 0.2 -tao_bnk_eta3 0.3 -tao_bnk_eta4 0.4 -tao_bnk_omega5 0 -tao_bnk_omega4 0 
python3 ToArray.py
python3 vecarray.py
gnuplot 3dplots.plt

echo Initial size : $size --- Max Iteration : $ITE --- Lagrange multiplier : $LAGMUL 
# memo (phi1,phi2,sqrt(1 - phi1^2 - phi2^2))
#nolagmuls itr 21でぶっ壊れる