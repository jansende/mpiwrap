#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();
    if (rank == 0)
    {
        auto number_request = mpi::comm("world")->dest(1)->isend(9);
        number_request->wait();

        auto message_request = mpi::comm("world")->dest(1)->isend("Hello, there\n");
        message_request->wait();

        auto numbers_request = mpi::comm("world")->dest(1)->isend(std::vector<double>{1.0, 2.0, 3.0});
        numbers_request->wait();
    }
    else if (rank == 1)
    {
        auto number_request = mpi::comm("world")->source(0)->irecv<int>();
        auto number = number_request->get();
        std::cout << "Number: " << number << '\n';

        auto message_request = mpi::comm("world")->source(0)->irecv<std::string>();
        auto message = message_request->get();
        std::cout << "Message: " << message;

        auto numbers_request = mpi::comm("world")->source(0)->irecv<std::vector<double>>();
        auto numbers = numbers_request->get();
        std::cout << "Data: ";
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    return 0;
}