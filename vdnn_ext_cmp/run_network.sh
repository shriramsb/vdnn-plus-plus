echo dyn 1>> ../stderr.dat
echo dyn 1>> ../stdout.dat
./alexnet_test.out dyn p 2>> ../stderr.dat 1>> ../stdout.dat
echo conv_p 1>> ../stderr.dat
echo conv_p 1>> ../stdout.dat
./alexnet_test.out conv p 2>> ../stderr.dat 1>> ../stdout.dat
echo conv_m 1>> ../stderr.dat
echo conv_m 1>> ../stdout.dat
./alexnet_test.out conv m 2>> ../stderr.dat 1>> ../stdout.dat
echo all_p 1>> ../stderr.dat
echo all_p 1>> ../stdout.dat
./alexnet_test.out all p 2>> ../stderr.dat 1>> ../stdout.dat
echo all_m 1>> ../stderr.dat
echo all_m 1>> ../stdout.dat
./alexnet_test.out all m 2>> ../stderr.dat 1>> ../stdout.dat
mkdir first_run
for f in *.dat*; do
	mv $f first_run/$f
done;
