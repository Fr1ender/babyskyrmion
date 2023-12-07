cd "../data"
set terminal x11
set encoding utf8
#set key left top  #凡例の位置
#set key at -4 , 4 
set xrange [-6:6] #プロット範囲
set yrange [-6:6]
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
#set palette defined (0 'blue', 0.15 'red', 0.3 'yellow')
#unset ztics                # do not show z-tics
#set pm3d at b              # draw with colored contour 
#set view 1,1               # view from the due north
#set view x軸まわりの回転角, z軸まわりの回転角, グラフの拡大率, z軸の拡大率となるx軸,z軸まわりの回転角度は度数で指定する
#set view 90,90,1,1

splot "initialN=101.dat" with vectors lt 6 title "initial vec"
#pause 5
splot "solN=101.dat" with vectors lt 6 title "solution vec"
pause 100
#splot "gradN=101.dat" with vectors lt 6 title "gradient vec"
#pause 30

#output
set terminal pngcairo
set encoding utf8
set output "initial=1.png"
replot

#画像出力用
#set terminal pngcairo
#set encoding utf8
#set output "initialexcontour.png"
#replot
