./vgg_test m
./vgg_test p
mkdir second_run
for f in *.dat; do
	mv $f second_run/$f
done;