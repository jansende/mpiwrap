#this project is valid for cmake 3.1 to 3.11
cmake_minimum_required(VERSION 3.1...3.11)
#fix the above syntax for cmake <3.12
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_VERSION})
endif()

#create the mpiwrap project
project(mpiwrap VERSION 0.1.0
                DESCRIPTION "mpiwrap is a C++ wrapper for the horrible C interface of MPI libaries."
                LANGUAGES CXX)

#add library interface
add_library(mpiwrap INTERFACE)
#add header files
target_include_directories(mpiwrap INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)

#add mpi
find_package(MPI REQUIRED)
#fix for cmake <3.9 --> https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()
#add mpi link
target_link_libraries(mpiwrap INTERFACE MPI::MPI_CXX)

#add examples
add_executable(hello_mpi examples/hello_mpi.cpp)
target_link_libraries(hello_mpi PRIVATE mpiwrap)
add_executable(send_recv examples/send_recv.cpp)
target_link_libraries(send_recv PRIVATE mpiwrap)