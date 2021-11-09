#Euler for diff h
set term pngcairo enhanced size 1280,1024
set datafile separator ","
set output 'euler_diff_h.png'
set border linewidth 2
set grid lt 1 lw 0.2 lc palette frac 0.5
set xtics 10 
set multiplot layout 2,1 
set key top left Left box title 
set xlabel 'Time(Years)' 
set ylabel 'Population'
set label 1 "(a)" at 96,346 font "enhanced [,20]"
set title "Euler Method(Predator) for different step sizes(h)"  font "enhanced [,20]"
unset label 1
unset title
# ----------------------
set label 2 "(b)" at 96,396 font "enhanced [,20]"
plot "data_1000.csv" u 1:3 title "n = 1000" , "data_10000.csv" u 1:3 title "n=10000" , "data_100000.csv" u 1:3 title "n = 100000", "data_1000000.csv" u 1:3 title "n = 1000000"
plot "data_1000.csv" u 1:7 title "n = 1000" , "data_10000.csv" u 1:7 title "n=10000" , "data_100000.csv" u 1:7 title "n = 100000", "data_1000000.csv" u 1:7 title "n = 1000000"
unset label 2
unset multiplot
