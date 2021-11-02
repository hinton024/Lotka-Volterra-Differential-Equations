#Euler for diff h
set term pngcairo enhanced size 1280,1024
set datafile separator ","
set output 'euler_diff_h.png'
set border linewidth 2

set multiplot layout 2,1 
set key top left Left box title 
set xlabel 'Time(Years)' 
set ylabel 'Population'
set title "Euler Method(Predator) for different step sizes(h)"  font "enhanced [,20]"
plot "Euler_data.csv" u 1:2 title "n = 1000" , "Euler_data.csv" u 4:5 title "n=10000" , "Euler_data.csv" u 7:8 title "n = 100000", "Euler_data.csv" u 10:11 title "n = 1000000"
set title "Euler Method(Prey) for different step sizes(h)"  font "enhanced [,20]"
plot "Euler_data.csv" u 1:3 title "n = 1000" , "Euler_data.csv" u 4:6 title "n=10000" , "Euler_data.csv" u 7:9 title "n = 100000", "Euler_data.csv" u 10:12 title "n = 1000000"
unset multiplot

#Plotting time v/s population
set term pngcairo enhanced size 1280,1024


set datafile separator ","
set output 'time_vs_population.png'
set border linewidth 2

set style line 1 linecolor rgb 'blue' linetype 1 linewidth 8
set style line 1 linecolor rgb 'green' linetype 1 linewidth 8
set multiplot layout 2,2 #title "Time v/s Population" font "enhanced [,30]"
set key top left Left box title 
set xlabel 'Time(Years)' 
set ylabel 'Population'
set title "OdeInt"  font "enhanced [,20]"
plot "test.csv" u 1:2 title "Predator" w l , "test.csv" u 1:6 title "Prey"  w l
set title "Euler Method"  font "enhanced [,20]"
plot "test.csv" u 1:3 title "Predator"  w l, "test.csv" u 1:7  title "Prey" w l
set title "Rk2 Method"  font "enhanced [,20]"
plot "test.csv" u 1:4  title "Predator" w l, "test.csv" u 1:8  title "Prey" w l
set title "Rk4 Method"  font "enhanced [,20]"
plot "test.csv" u 1:5  title "Predator" w l, "test.csv" u 1:9  title "Prey" w l
set title "rk4"

unset multiplot



#Error Plot for predator and prey
set term pngcairo enhanced size 1280,1024
set datafile separator ","
set output 'error_plot.png'
set border linewidth 2

set multiplot layout 2,1 #title "Time v/s Population" font "enhanced [,30]"
set key bottom left Left box title 
set xlabel 'Time(Years)' 
set ylabel 'Absolute Error'
set title "Error Plot(Predator)"  font "enhanced [,20]"
plot "test.csv" u 1:10 pt 4 title "Euler Method" , "test.csv" u 1:13 pt 4 ps 3 title "Rk2 Method" , "test.csv" u 1:15 pt 4 title "Rk4 Method"
set title "Error Plot(Prey)"  font "enhanced [,20]"
plot "test.csv" u 1:11 pt 4 title "Euler Method"  , "test.csv" u 1:13 pt 4 ps 3 title "Rk2 Method" , "test.csv" u 1:15 pt 4 title "Rk4 Method"
unset multiplot