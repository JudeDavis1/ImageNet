cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(imagenet)


find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
add_executable(imagenet main.cc model.cc)
target_link_libraries(imagenet "${TORCH_LIBRARIES}")

set_property(TARGET imagenet PROPERTY CXX_STANDARD 17)

