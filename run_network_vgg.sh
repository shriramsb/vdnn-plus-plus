cd vdnn
./burn_run_vgg.sh $1
cd ..
cd base
./run_network_vgg.sh $1
cd ../vdnn
./run_network_vgg.sh $1
cd ../vdnn_ext
./run_network_vgg.sh $1 $2
./run_network_reverse_vgg.sh $1 $2
cd ../vdnn
./run_network_reverse_vgg.sh $1
cd ../base
./run_network_reverse_vgg.sh $1
