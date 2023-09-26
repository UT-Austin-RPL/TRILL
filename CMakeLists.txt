cmake_minimum_required(VERSION 2.8.12)
project(VRInterface)

add_definitions("-std=c++11")

include (${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Protobuf REQUIRED)
include_directories(${PROTOBUF_INCLUDE_DIR})


add_library(rpc-draco-msg SHARED IMPORTED) # or STATIC instead of SHARED
set_target_properties(rpc-draco-msg PROPERTIES
  IMPORTED_LOCATION "${CMAKE_SOURCE_DIR}/messages/librpc-draco-msg.a"
  INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_SOURCE_DIR}/messages/"
)

add_executable(vr vr_teleop.cpp)
target_link_libraries(vr rpc-draco-msg ${CONAN_LIBS} ${PROTOBUF_LIBRARY})


#add_executable(vr_control vr_control.cpp)
#target_link_libraries(vr_control ${CONAN_LIBS})
#add_executable(vr_request test_request.cpp)
#target_link_libraries(vr_request ${CONAN_LIBS})

