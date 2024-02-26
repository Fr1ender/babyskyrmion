cd "../data"
set terminal x11
set encoding utf8
#set key left top  #凡例の位置
#set key at -4 , 4 
set xrange [-5:5] #プロット範囲
set yrange [-5:5]
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


set contour base
set cntrparam levels 8    #等高線の本数
#set nosurface              # do not show surface plot
#set view 0,0,1,1
splot "densityplotN=501.dat" using 1:2:3 w l title "Energy density"
pause 10

#splot "phi3N=800.dat" using 1:2:3 w l title "Energy density"
#pause 10

set terminal pngcairo
set encoding utf8
#set output "N=801_itr=26_f=35.3_res9.7_easyplain_nolag=0.5_1.png"
#replot

#output

# contourのみ
set terminal x11
set size square            # same side lengths for x and y
set nosurface              # do not show surface plot
set size square
#set view 0,0,1,1
set view map
replot
pause 3
#画像出力用
set terminal pngcairo
set encoding utf8
#set output "N=801_itr=26_f=35.3_res9.7_easyplain_nolag=0.5_2.png"
#replot

set terminal x11
set encoding utf8
set pm3d
set pm3d map
unset contour
#set cbrange[-1:1]
set palette defined ( 0 '#000090',1 '#000fff',2 '#0090ff',3 '#0fffee',4 '#90ff70',5 '#ffee00',6 '#ff7000',7 '#ee0000',8 '#7f0000')
splot "densityplotN=501.dat" using 1:2:3 with pm3d title "Energy density"
pause 10

set terminal pngcairo
set encoding utf8
#set output "N=801_itr=26_f=35.3_res9.7_easyplain_nolag=0.5_3.png"
#replot