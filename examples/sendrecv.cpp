#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();

    if (rank == 0)
    {
        auto number = mpi::comm("world")->source(1)->dest(1)->sendrecv<int>("Hello World");
        std::cout << "Number: " << number << '\n';
        auto numbers = mpi::comm("world")->source(1)->dest(1)->sendrecv<std::vector<int>>(7);
        std::cout << "Data: ";
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
        auto message = mpi::comm("world")->source(1)->dest(1)->sendrecv<std::string>(std::vector<int>{});
        std::cout << "Message: " << message << '\n';
    }
    else if (rank == 1)
    {
        auto message = mpi::comm("world")->source(0)->dest(0)->sendrecv<std::string>(42);
        std::cout << "Message: " << message << '\n';
        auto number = mpi::comm("world")->source(0)->dest(0)->sendrecv<int>(std::vector<int>{1, 2, 3, 4, 5});
        std::cout << "Number: " << number << '\n';
        auto numbers = mpi::comm("world")->source(0)->dest(0)->sendrecv<std::vector<int>>("Bye Bye");
        std::cout << "Data: ";
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    return 0;
}