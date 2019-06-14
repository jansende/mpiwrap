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
    }
    mpi::comm("world")->source(0)->bcast(numbers);
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    auto message = std::string{};
    if (rank == 0)
    {
        message = "Hello MPI";
    }
    message = mpi::comm("world")->source(0)->bcast<std::string>(message.c_str());
    std::cout << message << '\n';

    return 0;
}