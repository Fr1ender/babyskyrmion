#!/bin/bash

echo Initial size
read size
echo Max Iteration
read ITE
echo Lagrange multiplier
read LAGMUL

declare -i ITE2=$ITE*30

mpiexec -n 1 speight -tao_view -tao_smonitor -tao_gatol 1.e-4 -tao_type lmvm -tao_max_funcs 200000 -tao_max_it $ITE -mx 101 -my 101 -lambda $size -paramlag $LAGMUL -parc0 1.0 -parc4 1.0 -h 0.1 -itrmax $ITE2 
python3 ToArray.py
python3 VecArray.py
gnuplot 3dplots.plt

echo Initial size : $size --- Max Iteration : $ITE --- Lagrange multiplier : $LAGMUL 