g++ tvm_online_compile.cc -o tvm_online_compile -ltvm -L./build-adreno -I ./include/ -I./3rdparty/tvm-ffi/include/ -I./3rdparty/tvm-ffi/3rdparty/dlpack/include/ -I./3rdparty/dmlc-core/include/ -DDMLC_USE_LOGGING_LIBRARY="<tvm/runtime/logging.h>" -std=c++17 -ltvm_ffi -L./3rdparty/tvm-ffi/build/lib

LD_LIBRARY_PATH=./build-adreno:./3rdparty/tvm-ffi/build/lib  ./tvm_online_compile



