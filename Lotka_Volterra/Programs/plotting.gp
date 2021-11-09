#Euler for diff h
set term pngcairo enhanced size 1280,1024
set datafile separator ","
set output 'euler_diff_h.png'
set border linewidth 2
set grid lt 1 lw 0.2 lc palette frac 0.5
set xtics 10 
set multiplot layout 2,1 
set key top left Left box title 
set xlabel 'Time(Years)' font "enhanced [,15]" 
set ylabel 'Population (Predator) x 1000' font "enhanced [,15]"
set label 1 "(a)" at 95,337 font "enhanced [,20]"
set title "Euler Method for different step sizes"  font "enhanced [,20]"
plot "data_1000.csv" u 1:3 title "n = 1000" , "data_10000.csv" u 1:3 title "n=10000" , "data_100000.csv" u 1:3 title "n = 100000", "data_1000000.csv" u 1:3 title "n = 1000000"
unset label 1
unset title
unset ylabel
# ----------------------
set ylabel 'Population (Prey) x 1000' font "enhanced [,15]"
set label 2 "(b)" at 95,387 font "enhanced [,20]"
plot "data_1000.csv" u 1:7 title "n = 1000" , "data_10000.csv" u 1:7 title "n=10000" , "data_100000.csv" u 1:7 title "n = 100000", "data_1000000.csv" u 1:7 title "n = 1000000"
unset label 2
unset ylabel
unset multiplot

#RK2 for diff h
set term pngcairo enhanced size 1280,1024
set datafile separator ","
set output 'rk2_diff_h.png'
set border linewidth 2
set grid lt 1 lw 0.2 lc palette frac 0.5
set multiplot layout 2,1 
set key top left Left box title 
set xlabel 'Time(Years)' font "enhanced [,15]" 
set ylabel 'Population (Predator) x 1000' font "enhanced [,15]"
set label 1 "(a)" at 90,80 font "enhanced [,25]"
set title "RK2 Method for different step sizes"  font "enhanced [,20]"
plot "data_1000.csv" u 1:4 title "n = 1000" , "data_10000.csv" u 1:4 title "n=10000" , "data_100000.csv" u 1:4 title "n = 100000", "data_1000000.csv" u 1:4 title "n = 1000000"
unset label 1
unset title
unset ylabel
# ----------------------
set ylabel 'Population (Prey) x 1000' font "enhanced [,15]"
set label 2 "(b)" at 90,65 font "enhanced [,25]"
plot "data_1000.csv" u 1:8 title "n = 1000" , "data_10000.csv" u 1:8 title "n=10000" , "data_100000.csv" u 1:8 title "n = 100000", "data_1000000.csv" u 1:8 title "n = 1000000"
unset label 2
unset ylabel
unset multiplot

#RK4 for diff h
set term pngcairo enhanced size 1280,1024
set datafile separator ","
set output 'rk4_diff_h.png'
set border linewidth 2
set grid lt 1 lw 0.2 lc palette frac 0.5
set multiplot layout 2,1 
set key top left Left box title 
set xlabel 'Time(Years)' font "enhanced [,15]" 
set ylabel 'Population (Predator) x 1000' font "enhanced [,15]"
set label 1 "(a)" at 90,80 font "enhanced [,25]"
set title "RK4 Method for different step sizes"  font "enhanced [,20]"
plot "data_1000.csv" u 1:5 title "n = 1000" , "data_10000.csv" u 1:5 title "n=10000" , "data_100000.csv" u 1:5 title "n = 100000", "data_1000000.csv" u 1:5 title "n = 1000000"
unset label 1
unset title
unset ylabel
# ----------------------
set ylabel 'Population (Prey) x 1000' font "enhanced [,15]"
set label 2 "(b)" at 90,65 font "enhanced [,25]"
plot "data_1000.csv" u 1:9 title "n = 1000" , "data_10000.csv" u 1:9 title "n=10000" , "data_100000.csv" u 1:9 title "n = 100000", "data_1000000.csv" u 1:9 title "n = 1000000"
unset label 2
unset ylabel
unset multiplot



#Plotting time v/s population
set term pngcairo enhanced size 1280,1024
set datafile separator ","
set output 'time_vs_population.png'
set grid lt 1 lw 0.2 lc palette frac 0.5
set border linewidth 2
set xtics 10 
set style line 1 linecolor rgb 'blue' linetype 1 linewidth 8
set style line 1 linecolor rgb 'green' linetype 1 linewidth 8
set multiplot layout 2,2 #title "Time v/s Population" font "enhanced [,30]"
set key top left Left box title 
set xlabel 'Time(Years)' font "enhanced [,15]" 
set ylabel 'Population x 1000' font "enhanced [,15]"
set label 1 "(a)" at 90,85 font "enhanced [,25]"
set title "OdeInt"  font "enhanced [,20]"
plot "data_1000.csv" u 1:2 title "Predator" w l , "data_1000.csv" u 1:6 title "Prey"  w l
unset label 1

# ----------------------
set label 1 "(b)" at 90,85 font "enhanced [,25]"
set title "Euler Method"  font "enhanced [,20]"
plot "data_100000.csv" u 1:3 title "Predator"  w l, "data_100000.csv" u 1:7  title "Prey" w l
unset label 1
# ----------------------
set label 1 "(c)" at 90,85 font "enhanced [,25]"
set title "Rk2 Method"  font "enhanced [,20]"
plot "data_10000.csv" u 1:4  title "Predator" w l, "data_10000.csv" u 1:8  title "Prey" w l
unset label 1
# ----------------------
set label 1 "(d)" at 90,85 font "enhanced [,25]"
set title "Rk4 Method"  font "enhanced [,20]"
plot "data_1000.csv" u 1:5  title "Predator" w l, "data_1000.csv" u 1:9  title "Prey" w l
unset label 1
unset multiplot


#Relative Error between Scipy's Odeint Numerical method and Euler, RK2, RK4 Methods for predator and prey 
set term pngcairo enhanced size 1280,1024
set datafile separator ","
set output 'error_plot.png'
set border linewidth 2
unset key
set multiplot layout 2,1 #title "Time v/s Population" font "enhanced [,30]"
set key bottom left Left box title 
set xlabel 'Time(Years)' font "enhanced [,15]"
# ------------------
set ylabel 'Relative Error (Predator)' font "enhanced [,15]"
set title "Relative Error between Scipy's Odeint Numerical method and Euler, RK2, RK4 Methods"  font "enhanced [,20]"
set label 1 "(a)" at 2,0.08 font "enhanced [,20]"
plot "data_100000.csv" u 1:10 pt 4 title "Euler Method" , "data_10000.csv" u 1:12 pt 4 ps 3 title "Rk2 Method" , "data_1000.csv" u 1:14 pt 4 title "Rk4 Method"
unset label 1
unset title
# ------------------
set ylabel 'Relative Error (Prey)' font "enhanced [,15]"
set label 2 "(b)" at 2,0.18 font "enhanced [,20]"
plot "data_100000.csv" u 1:11 pt 4 title "Euler Method"  , "data_10000.csv" u 1:13 pt 4 ps 3 title "Rk2 Method" , "data_1000.csv" u 1:15 pt 4 title "Rk4 Method"

unset label 2
unset multiplot