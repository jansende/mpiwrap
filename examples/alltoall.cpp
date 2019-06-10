#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();

    auto numbers = (rank == 0) ? std::vector<int>{1, 2, 3, 4, 5, 1, 2, 3, 4, 5} : std::vector<int>{6, 7, 8, 9, 10, 6, 7, 8, 9, 10};

    std::cout << "rank " << rank << " here, with: ";
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    numbers = mpi::comm("world")->alltoall(numbers, 5);

    std::cout << "rank " << rank << " here, with: ";
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    return 0;
}