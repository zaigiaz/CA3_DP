set terminal pngcairo size 1024,768
set output 'time_complexity_linear.png'

set title 'Time Complexity: Effect of Number of Grandkids' font ',14'
set xlabel 'Array Size (N)' font ',12'
set ylabel 'Execution Time (seconds)' font ',12'

set grid
set key top left

set style data linespoints
set pointsize 1.5

plot 'time.dat' using 1:2 title '1 Grandkid' with linespoints pt 7 lc rgb 'blue', \
     'time.dat' using 1:3 title '2 Grandkids' with linespoints pt 5 lc rgb 'green', \
     'time.dat' using 1:4 title '3 Grandkids' with linespoints pt 9 lc rgb 'red'
