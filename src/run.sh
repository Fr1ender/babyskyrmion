#!/bin/bash
echo Max Iteration
read ITE
echo Lagrange multiplier
read LAGMUL
mpiexec -n 1 speight -tao_view -tao_smonitor -tao_gatol 1.e-4 -tao_type lmvm -tao_max_funcs 200000 -tao_max_it $ITE -mx 101 -my 101 -lambda 100 -paramlag $LAGMUL -parc0 0.0
python3 ToArray.py
gnuplot 3dplots.plt
echo Max Iteration : $ITE --- Lagrange multiplier : $LAGMUL