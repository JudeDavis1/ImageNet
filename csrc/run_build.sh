cd build
cmake -DCMAKE_PREFIX_PATH=../include/libtorch -DUSE_CUDA=OFF ..
cmake --build . --config Release
cp imagenet ..
cd ..
