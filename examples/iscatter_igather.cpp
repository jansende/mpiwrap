#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();

    auto numbers = std::vector<double>{};
    if (rank == 0)
    {
        numbers.emplace_back(4.0);
        numbers.emplace_back(7.1);
        numbers.emplace_back(8.9);
        numbers.emplace_back(42.);
        numbers.emplace_back(3.4);
        numbers.emplace_back(-11.);
    }
    numbers = mpi::comm("world")->source(0)->iscatter(numbers, 2)->get();
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';
    numbers = mpi::comm("world")->dest(0)->igather(numbers)->get();
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';
    numbers = mpi::comm("world")->source(0)->iscatter(numbers, 2)->get();
    numbers = mpi::comm("world")->iallgather(numbers)->get();
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    auto message = std::string{};
    if (rank == 0)
    {
        message = "Hello MPI";
    }
    message = mpi::comm("world")->source(0)->iscatter(message, 3)->get();
    std::cout << message << '\n';
    message = mpi::comm("world")->dest(0)->igather(message)->get();
    std::cout << message << '\n';
    message = mpi::comm("world")->source(0)->iscatter(message, 3)->get();
    message = mpi::comm("world")->iallgather(message)->get();
    std::cout << message << '\n';

    return 0;
}