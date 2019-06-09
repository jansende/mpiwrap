#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();

    auto number = mpi::comm("world")->dest(0)->reduce(rank, MPI_SUM);
    if (rank == 0)
        std::cout << number << '\n';

    number = mpi::comm("world")->dest(0)->reduce(rank, MPI_MIN);
    if (rank == 0)
        std::cout << number << '\n';

    number = mpi::comm("world")->dest(0)->reduce(rank, MPI_MAX);
    if (rank == 0)
        std::cout << number << '\n';

    auto numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, MPI_SUM);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, MPI_MIN);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, MPI_MAX);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }
    numbers = mpi::comm("world")->allreduce(std::vector<int>{rank, 5 - rank}, MPI_MAX);
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    return 0;
}