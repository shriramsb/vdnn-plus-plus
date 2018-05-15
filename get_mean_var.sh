echo base_p
./mean_var.out base/first_run/base_p.dat.bin base/second_run/base_p.dat.bin
echo base_m
./mean_var.out base/first_run/base_m.dat.bin base/second_run/base_m.dat.bin
echo vdnn_dyn
./mean_var.out vdnn/first_run/vdnn_dyn.dat.bin vdnn/second_run/vdnn_dyn.dat.bin
echo vdnn_conv_p
./mean_var.out vdnn/first_run/vdnn_conv_p.dat.bin vdnn/second_run/vdnn_conv_p.dat.bin
echo vdnn_conv_m
./mean_var.out vdnn/first_run/vdnn_conv_m.dat.bin vdnn/second_run/vdnn_conv_m.dat.bin
echo vdnn_all_p
./mean_var.out vdnn/first_run/vdnn_all_p.dat.bin vdnn/second_run/vdnn_all_p.dat.bin
echo vdnn_all_m
./mean_var.out vdnn/first_run/vdnn_all_m.dat.bin vdnn/second_run/vdnn_all_m.dat.bin

echo vdnnext_dyn
./mean_var.out vdnn_ext/first_run/vdnn_dyn.dat.bin vdnn_ext/second_run/vdnn_dyn.dat.bin
echo vdnnext_conv_p
./mean_var.out vdnn_ext/first_run/vdnn_conv_p.dat.bin vdnn_ext/second_run/vdnn_conv_p.dat.bin
echo vdnnext_conv_m
./mean_var.out vdnn_ext/first_run/vdnn_conv_m.dat.bin vdnn_ext/second_run/vdnn_conv_m.dat.bin
echo vdnnext_all_p
./mean_var.out vdnn_ext/first_run/vdnn_all_p.dat.bin vdnn_ext/second_run/vdnn_all_p.dat.bin
echo vdnnext_all_m
./mean_var.out vdnn_ext/first_run/vdnn_all_m.dat.bin vdnn_ext/second_run/vdnn_all_m.dat.bin
