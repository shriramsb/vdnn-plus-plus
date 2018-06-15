echo base_p 1>> stderr.dat
echo base_p 1>> stdout.dat
./vgg_test.out p 2>> stderr.dat 1>> stdout.dat
echo base_m 1>> stderr.dat
echo base_m 1>> stdout.dat
./vgg_test.out m 2>> stderr.dat 1>> stdout.dat
mkdir first_run
for f in *.dat*; do
	mv $f first_run/$f
done;