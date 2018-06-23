cd vdnn
./vdnn_ext_demo.out conv m $1 $3 1> /dev/null 2> /dev/null
cd ../base
./vdnn_ext_demo.out m $1 $3 1> stdout.dat 2> stderr.dat
mkdir vdnn_ext_demo_1
mv *.dat vdnn_ext_demo_1/
cd ../vdnn
./vdnn_ext_demo.out conv m $1 $3 1> stdout.dat 2> stderr.dat
mkdir vdnn_ext_demo_1
mv *.dat vdnn_ext_demo_1/
cd ../vdnn_ext
timeout $2 ./vdnn_ext_demo.out conv m $1 $3 1> stdout.dat 2> stderr.dat
mkdir vdnn_ext_demo_1
mv *.dat vdnn_ext_demo_1/
timeout $2 ./vdnn_ext_demo.out conv m $1 $3 1>> stdout.dat 2>> stderr.dat
mkdir vdnn_ext_demo_2
mv *.dat vdnn_ext_demo_2/
cd ../vdnn
./vdnn_ext_demo.out conv m $1 $3 1>> stdout.dat 2>> stderr.dat
mkdir vdnn_ext_demo_2
mv *.dat vdnn_ext_demo_2/
cd ../base
./vdnn_ext_demo.out m $1 $3 1>> stdout.dat 2>> stderr.dat
mkdir vdnn_ext_demo_2
mv *.dat vdnn_ext_demo_2/