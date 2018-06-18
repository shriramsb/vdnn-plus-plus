mkdir temp
cd temp
mkdir base
mkdir vdnn
mkdir vdnn_ext
cd ..

cp -r ../base/first_run temp/base
cp -r ../base/second_run temp/base

cp -r ../vdnn/first_run temp/vdnn
cp -r ../vdnn/second_run temp/vdnn

cp -r ../vdnn_ext/first_run temp/vdnn_ext
cp -r ../vdnn_ext/second_run temp/vdnn_ext

cp ../stderr.dat temp
cp ../stdout.dat temp