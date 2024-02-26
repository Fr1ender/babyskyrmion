cd "../data"
set terminal x11
set encoding utf8
#set key left top  #凡例の位置
#set key at -4 , 4 
set xrange [-6:6] #プロット範囲
set yrange [-6:6]
#set zrange [-6:6]
#set xrange [-0.1:0.1] #プロット範囲
#set yrange [-0.1:0.1]
#set zrange[-0.5:0.5]
set xlabel 'x軸ラベル' 
set ylabel 'y軸ラベル'

#set format x "$%1.3t\\times10^{%L}$" #目盛りの話

set mxtics 2 #ちっちゃい目盛りの数が変わるよ
set mytics 2 
#set logscale x   #対数軸

#プロット内容
set size square            # same side lengths for x and y
set xlabel 'x axis'             # x-axis
set ylabel 'y axis'             # y-axis
#set view x軸まわりの回転角, z軸まわりの回転角, グラフの拡大率, z軸の拡大率となるx軸,z軸まわりの回転角度は度数で指定する
#set view 90,90,1,1 # yz
#set view 90,0,1,1 # xz
set view 0,0,1,1 # xy

#splot "initial=1_N=101.dat" with vectors lt 6 title "initial vec"
#pause 5
splot "solN=801.dat" with vectors lt 6 title "solution vec"
pause 10
splot "gradN=801.dat" with vectors lt 6 title "gradient vec"
pause 300

#output
#set terminal pngcairo
#set encoding utf8
#set output "itr=1_xy.png"
replot

#画像出力用
#set terminal pngcairo
#set encoding utf8
#set output "initialexcontour.png"
#replot
