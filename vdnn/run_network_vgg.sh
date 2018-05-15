./vgg_test dyn p
./vgg_test conv p
./vgg_test conv m
./vgg_test all p
./vgg_test all m
mkdir first_run
for f in *.dat; do
	mv $f first_run/$f
done;
