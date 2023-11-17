#cd "../data"
set terminal x11
set encoding utf8
#set key left top  #凡例の位置
#set key at -4 , 4 
#set xrange [-2:2] #プロット範囲
set xlabel 'x' 
set ylabel 'y'

#set format x "$%1.3t\\times10^{%L}$" #目盛りの話

set mxtics 3 #ちっちゃい目盛りの数が変わるよ
set mytics 3 
#set logscale x   #対数軸


#TikZで出力用 (もろもろはgnuplot tikz で検索)
#set terminal tikz
#set encoding utf8
#set output "outputname.tex"

#output
#set terminal pngcairo
#set encoding utf8
#set output "derrickex.png"
#replot

#set logscale y
set yrange [0:10]
plot "derrickitr.dat" every ::2::25
#pause 5
#plot "chargeitr.dat" every ::2::30
# latest result L = 10 ini = 1 itr 30

#set terminal pngcairo
#set encoding utf8
#set output "chargeL=10Ini=1.png"
#replot
pause 10