#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto world_size = mpi::comm("world")->size();
    auto world_rank = mpi::comm("world")->rank();

    auto processor_name = mpi::processor_name();
    std::cout << "Hello world from processor " << processor_name << ", rank " << world_rank << " out of " << world_size << " processors.\n";

    return 0;
}