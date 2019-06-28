#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();

    auto numbers = std::vector<double>{};
    numbers = mpi::comm("world")->dest(0)->gather(42.0);
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';
    numbers = mpi::comm("world")->source(0)->scatter(numbers, 2);
    numbers = mpi::comm("world")->iallgather(7.0)->get();
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    auto message = std::string{};
    message = mpi::comm("world")->dest(0)->gather('d');
    std::cout << message << '\n';
    message = mpi::comm("world")->source(0)->scatter(message, 3);
    message = mpi::comm("world")->iallgather('c')->get();
    std::cout << message << '\n';

    message = mpi::comm("world")->dest(0)->gather("hi");
    std::cout << message << '\n';
    message = mpi::comm("world")->source(0)->scatter(message, 3);
    message = mpi::comm("world")->iallgather("ho")->get();
    std::cout << message << '\n';

    return 0;
}