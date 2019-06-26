#this project is valid for cmake 3.1 to 3.11
cmake_minimum_required(VERSION 3.1...3.11)
#fix the above syntax for cmake <3.12
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_VERSION})
endif()

#create the mpiwrap project
project(mpiwrap VERSION 0.1.0
                DESCRIPTION "mpiwrap is a C++ wrapper for the horrible C interface of MPI libraries."
                LANGUAGES CXX)

#add library
add_library(mpiwrap src/mpi.cpp src/scheduler.cpp)
#add header files
target_include_directories(mpiwrap PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

#add mpi
find_package(MPI REQUIRED)
#fix for cmake <3.9 --> https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED PUBLIC)
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()
#add mpi link
target_link_libraries(mpiwrap PUBLIC MPI::MPI_CXX)
#add additional build options
option(MPIWRAP_EXAMPLES_ENABLED "Build examples" OFF)
option(MPIWRAP_PARANOID_ASSERTIONS "Use paranoid assertions" OFF)

#enter paranoia
if (MPIWRAP_PARANOID_ASSERTIONS)
    target_compile_definitions(mpiwrap PUBLIC BE_PARANOID)
endif()

#add examples
if (MPIWRAP_EXAMPLES_ENABLED)
    add_executable(allgather examples/allgather.cpp)
    target_link_libraries(allgather PRIVATE mpiwrap)
    add_executable(alltoall examples/alltoall.cpp)
    target_link_libraries(alltoall PRIVATE mpiwrap)
    add_executable(bcast examples/bcast.cpp)
    target_link_libraries(bcast PRIVATE mpiwrap)
    add_executable(hello_mpi examples/hello_mpi.cpp)
    target_link_libraries(hello_mpi PRIVATE mpiwrap)
    add_executable(ibcast examples/ibcast.cpp)
    target_link_libraries(ibcast PRIVATE mpiwrap)
    add_executable(isend_irecv examples/isend_irecv.cpp)
    target_link_libraries(isend_irecv PRIVATE mpiwrap)
    add_executable(isend_irecv_alt examples/isend_irecv_alt.cpp)
    target_link_libraries(isend_irecv_alt PRIVATE mpiwrap)
    add_executable(make_op examples/make_op.cpp)
    target_link_libraries(make_op PRIVATE mpiwrap)
    add_executable(reduce examples/reduce.cpp)
    target_link_libraries(reduce PRIVATE mpiwrap)
    add_executable(scatter_gather examples/scatter_gather.cpp)
    target_link_libraries(scatter_gather PRIVATE mpiwrap)
    add_executable(scheduler examples/scheduler.cpp)
    target_link_libraries(scheduler PRIVATE mpiwrap)
    add_executable(send_recv examples/send_recv.cpp)
    target_link_libraries(send_recv PRIVATE mpiwrap)
endif()