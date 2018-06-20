cd vdnn
./burn_run.sh $1
cd ..
cd base
./run_network.sh $1
cd ../vdnn
./run_network.sh $1
cd ../vdnn_ext
./run_network.sh $1 $2
./run_network_reverse.sh $1 $2
cd ../vdnn
./run_network_reverse.sh $1
cd ../base
./run_network_reverse.sh $1