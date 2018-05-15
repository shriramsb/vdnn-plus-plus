./vgg_test m
./vgg_test p
./vgg_test m
mkdir first_run
for f in *.dat; do
	mv $f first_run/$f
done;