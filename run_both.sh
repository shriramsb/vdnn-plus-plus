./run_network.sh
cd final_experiments
./copy_res.sh
mv temp AlexNet_512_data
cd ..
rm stdout.dat
rm stderr.dat
./run_network_vgg.sh
cd final_experiments
./copy_res.sh
mv temp VGG_42_data