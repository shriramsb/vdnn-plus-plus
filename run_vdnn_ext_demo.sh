cd vdnn
./vdnn_ext_demo.out conv p $1 $3 1> /dev/null 2> /dev/null
cd ../base
./vdnn_ext_demo.out p $1 $3 1> stdout.dat 2> stderr.dat
mkdir vdnn_ext_demo_1
mv *.dat vdnn_ext_demo/
cd ../vdnn
./vdnn_ext_demo.out conv p $1 $3 1> stdout.dat 2> stderr.dat
mkdir vdnn_ext_demo_1
mv *.dat vdnn_ext_demo/
cd ../vdnn_ext
timeout $2 ./vdnn_ext_demo.out conv p $1 $3 1> stdout.dat 2> stderr.dat
mkdir vdnn_ext_demo_1
mv *.dat vdnn_ext_demo/
timeout $2 ./vdnn_ext_demo.out conv p $1 $3 1>> stdout.dat 2>> stderr.dat
mkdir vdnn_ext_demo_2
mv *.dat vdnn_ext_demo/
cd ../vdnn
./vdnn_ext_demo.out conv p $1 $3 1>> stdout.dat 2>> stderr.dat
mkdir vdnn_ext_demo_2
mv *.dat vdnn_ext_demo/
cd ../base
./vdnn_ext_demo.out p $1 $3 1>> stdout.dat 2>> stderr.dat
mkdir vdnn_ext_demo_2
mv *.dat vdnn_ext_demo/