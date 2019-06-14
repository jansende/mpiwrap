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
    numbers = mpi::comm("world")->source(0)->scatter(numbers, 2);
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';
    numbers = mpi::comm("world")->dest(0)->gather(numbers);
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';
    numbers = mpi::comm("world")->source(0)->scatter(numbers, 2);
    numbers = mpi::comm("world")->allgather(numbers);
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    auto message = std::string{};
    if (rank == 0)
    {
        message = "Hello MPI";
    }
    message = mpi::comm("world")->source(0)->scatter(message, 3);
    std::cout << message << '\n';
    message = mpi::comm("world")->dest(0)->gather(message);
    std::cout << message << '\n';
    message = mpi::comm("world")->source(0)->scatter(message, 3);
    message = mpi::comm("world")->allgather(message);
    std::cout << message << '\n';

    return 0;
}