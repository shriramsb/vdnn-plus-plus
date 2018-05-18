cd vdnn
./burn_run_vgg.sh
cd ..
cd base
./run_network_vgg.sh
cd ../vdnn
./run_network_vgg.sh
cd ../vdnn_ext
./run_network_vgg.sh
./run_network_reverse_vgg.sh
cd ../vdnn
./run_network_reverse_vgg.sh
cd ../base
./run_network_reverse_vgg.sh