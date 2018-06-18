./run_network.sh 128
cd final_experiments
./copy_res.sh
mv temp AlexNet_128_data
cd ..
./run_network_vgg.sh 16
cd final_experiments
./copy_res.sh
mv temp VGG_16_data