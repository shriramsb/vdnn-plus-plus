echo vdnn_dyn 1>> ../stderr.dat
echo vdnn_dyn 1>> ../stdout.dat
./vgg_test.out dyn p 2>> ../stderr.dat 1>> ../stdout.dat
echo vdnn_conv_p 1>> ../stderr.dat
echo vdnn_conv_p 1>> ../stdout.dat
./vgg_test.out conv p 2>> ../stderr.dat 1>> ../stdout.dat
echo vdnn_conv_m 1>> ../stderr.dat
echo vdnn_conv_m 1>> ../stdout.dat
./vgg_test.out conv m 2>> ../stderr.dat 1>> ../stdout.dat
echo vdnn_all_p 1>> ../stderr.dat
echo vdnn_all_p 1>> ../stdout.dat
./vgg_test.out all p 2>> ../stderr.dat 1>> ../stdout.dat
echo vdnn_all_m 1>> ../stderr.dat
echo vdnn_all_m 1>> ../stdout.dat
./vgg_test.out all m 2>> ../stderr.dat 1>> ../stdout.dat
mkdir expt
cd expt
mkdir VGG_64
cd ..
for f in *.dat*; do
	mv $f expt/VGG_64/$f
done;

