./run_network.sh
cd final_experiments
./copy_res.sh
mv temp AlexNet_256_data
cd ..
./run_network_vgg.sh
cd final_experiments
./copy_res.sh
mv temp VGG_64_data