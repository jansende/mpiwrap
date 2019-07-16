#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();

    auto number = mpi::comm("world")->dest(0)->ireduce(rank, mpi::sum)->get();
    if (rank == 0)
        std::cout << number << '\n';

    number = mpi::comm("world")->dest(0)->ireduce(rank, mpi::min)->get();
    if (rank == 0)
        std::cout << number << '\n';

    number = mpi::comm("world")->dest(0)->ireduce(rank, mpi::max)->get();
    if (rank == 0)
        std::cout << number << '\n';

    auto numbers = mpi::comm("world")->dest(0)->ireduce(std::vector<int>{rank, 5 - rank}, mpi::sum)->get();
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    numbers = mpi::comm("world")->dest(0)->ireduce(std::vector<int>{rank, 5 - rank}, mpi::min)->get();
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    numbers = mpi::comm("world")->dest(0)->ireduce(std::vector<int>{rank, 5 - rank}, mpi::max)->get();
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }
    numbers = mpi::comm("world")->iallreduce(std::vector<int>{rank, 5 - rank}, mpi::max)->get();
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    return 0;
}