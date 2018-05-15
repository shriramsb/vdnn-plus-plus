echo base_m 1>> ../stderr.dat
echo base_m 1>> ../stdout.dat
./alexnet_test m 2>> ../stderr.dat 1>> ../stdout.dat
echo base_p 1>> ../stderr.dat
echo base_p 1>> ../stdout.dat
./alexnet_test p 2>> ../stderr.dat 1>> ../stdout.dat
mkdir second_run
for f in *.dat; do
	mv $f second_run/$f
done;