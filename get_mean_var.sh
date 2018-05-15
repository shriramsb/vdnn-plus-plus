echo base_p
./mean_var basememorymanager/first_run/base_p.dat.bin basememorymanager/second_run/base_p.dat.bin
echo base_m
./mean_var basememorymanager/first_run/base_m.dat.bin basememorymanager/second_run/base_m.dat.bin
echo vdnn_dyn
./mean_var vdnnmemorymanager/first_run/vdnn_dyn.dat.bin vdnnmemorymanager/second_run/vdnn_dyn.dat.bin
echo vdnn_conv_p
./mean_var vdnnmemorymanager/first_run/vdnn_conv_p.dat.bin vdnnmemorymanager/second_run/vdnn_conv_p.dat.bin
echo vdnn_conv_m
./mean_var vdnnmemorymanager/first_run/vdnn_conv_m.dat.bin vdnnmemorymanager/second_run/vdnn_conv_m.dat.bin
echo vdnn_all_p
./mean_var vdnnmemorymanager/first_run/vdnn_all_p.dat.bin vdnnmemorymanager/second_run/vdnn_all_p.dat.bin
echo vdnn_all_m
./mean_var vdnnmemorymanager/first_run/vdnn_all_m.dat.bin vdnnmemorymanager/second_run/vdnn_all_m.dat.bin

echo vdnnext_dyn
./mean_var vdnnextmemorymanager/first_run/vdnn_dyn.dat.bin vdnnextmemorymanager/second_run/vdnn_dyn.dat.bin
echo vdnnext_conv_p
./mean_var vdnnextmemorymanager/first_run/vdnn_conv_p.dat.bin vdnnextmemorymanager/second_run/vdnn_conv_p.dat.bin
echo vdnnext_conv_m
./mean_var vdnnextmemorymanager/first_run/vdnn_conv_m.dat.bin vdnnextmemorymanager/second_run/vdnn_conv_m.dat.bin
echo vdnnext_all_p
./mean_var vdnnextmemorymanager/first_run/vdnn_all_p.dat.bin vdnnextmemorymanager/second_run/vdnn_all_p.dat.bin
echo vdnnext_all_m
./mean_var vdnnextmemorymanager/first_run/vdnn_all_m.dat.bin vdnnextmemorymanager/second_run/vdnn_all_m.dat.bin
