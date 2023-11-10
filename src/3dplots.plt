cd "../data"
set terminal x11
set encoding utf8
#set key left top  #凡例の位置
#set key at -4 , 4 
#set xrange [-2:2] #プロット範囲
#set yrange [-2:2]
set xlabel 'x軸ラベル' 
set ylabel 'y軸ラベル'

#set format x "$%1.3t\\times10^{%L}$" #目盛りの話

set mxtics 2 #ちっちゃい目盛りの数が変わるよ
set mytics 2 
#set logscale x   #対数軸


#TikZで出力用 (もろもろはgnuplot tikz で検索)
#set terminal tikz
#set encoding utf8
#set output "outputname.tex"

# fitting
#f(x) = a*x + b
#fit f(x) "data.dat" using 1:2 via a,b

#プロット内容
set size square            # same side lengths for x and y
set xlabel 'x axis'             # x-axis
set ylabel 'y axis'             # y-axis
#set palette defined (0 'blue', 0.15 'red', 0.3 'yellow')
#unset ztics                # do not show z-tics
#set pm3d at b              # draw with colored contour 
#set view 0,0               # view from the due north

set contour
set cntrparam levels 15     #等高線の本数
#set nosurface              # do not show surface plot
#set view 0,0,1,1
splot "densityplotN=101.dat" using 1:2:3 w l title "Energy density"
pause 5

#output
set terminal pngcairo
set encoding utf8
set output "initialex.png"
replot

# contourのみ
set terminal x11
set nosurface              # do not show surface plot
set view 0,0,1,1
replot
#画像出力用
#set terminal pngcairo
#set encoding utf8
#set output "initialexcontour.png"
#replot

pause 5