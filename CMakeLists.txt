cmake_minimum_required (VERSION 2.6)
project (train_HOG_ANN)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++11")

LINK_DIRECTORIES("/usr/local/lib")

set(OPENCV_LIB opencv_core opencv_imgcodecs opencv_improc opencv_objectdetect opencv_ml)

add_executable(train_ANN_HOG opencv_train_ann_hog.cpp)
target_link_libraries(train_ANN_HOG ${OPENCV_LIB})
