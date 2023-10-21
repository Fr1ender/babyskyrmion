set terminal x11
set encoding utf8
set key left top  #凡例の位置
# set key at x , y 
#set xrange [-5:5] #プロット範囲
#set yrange [-2:2]
set xlabel 'x軸ラベル' 
set ylabel 'y軸ラベル'

#set format x "$%1.3t\\times10^{%L}$" #目盛りの話

set mxtics 2 #ちっちゃい目盛りの数が変わるよ
set mytics 2 
#set logscale x   #対数軸

#画像出力用
#set terminal pngcairo
#set encoding utf8
#set output "outputname.png"

#TikZで出力用 (もろもろはgnuplot tikz で検索)
#set terminal tikz
#set encoding utf8
#set output "outputname.tex"

# fitting
#f(x) = a*x + b
#fit f(x) "data.dat" using 1:2 via a,b

#プロット内容
set size square            # same side lengths for x and y
set xlabel 'i'             # x-axis
set ylabel 'j'             # y-axis
#set palette defined (0 'blue', 0.15 'red', 0.3 'yellow')
set nosurface              # do not show surface plot
#unset ztics                # do not show z-tics
#set pm3d at b              # draw with colored contour 
#set view 0,0               # view from the due north
set contour
set view 0,0,1,1
splot "densityplotcancelN=51.dat" using 1:2:3 w l title "neko"
#plot "data.dat" pt 7 lc 8 title "データ"  #実験データのプロットとか

pause 5