./vgg_test all m
./vgg_test all p
./vgg_test conv m
./vgg_test conv p
./vgg_test dyn p
mkdir second_run
for f in *.dat; do
	mv $f second_run/$f
done;