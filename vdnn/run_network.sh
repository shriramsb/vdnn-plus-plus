echo vdnn_dyn 1>> ../stderr.dat
echo vdnn_dyn 1>> ../stdout.dat
./alexnet_test dyn p 2>> ../stderr.dat 1>> ../stdout.dat
echo vdnn_conv_p 1>> ../stderr.dat
echo vdnn_conv_p 1>> ../stdout.dat
./alexnet_test conv p 2>> ../stderr.dat 1>> ../stdout.dat
echo vdnn_conv_m 1>> ../stderr.dat
echo vdnn_conv_m 1>> ../stdout.dat
./alexnet_test conv m 2>> ../stderr.dat 1>> ../stdout.dat
echo vdnn_all_p 1>> ../stderr.dat
echo vdnn_all_p 1>> ../stdout.dat
./alexnet_test all p 2>> ../stderr.dat 1>> ../stdout.dat
echo vdnn_all_m 1>> ../stderr.dat
echo vdnn_all_m 1>> ../stdout.dat
./alexnet_test all m 2>> ../stderr.dat 1>> ../stdout.dat
mkdir first_run
for f in *.dat*; do
	mv $f first_run/$f
done;
