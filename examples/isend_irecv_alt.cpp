#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();
    if (rank == 0)
    {
        auto number_request = mpi::comm("world")->dest(1)->isend(9);
        auto message_request = mpi::comm("world")->dest(1)->isend("Hello, there\n");
        auto numbers_request = mpi::comm("world")->dest(1)->isend(std::vector<double>{1.0, 2.0, 3.0});

        mpi::waitall(number_request, message_request, numbers_request);
    }
    else if (rank == 1)
    {
        auto number_request = mpi::comm("world")->source(0)->irecv<int>();
        auto message_request = mpi::comm("world")->source(0)->irecv<std::string>();
        auto numbers_request = mpi::comm("world")->source(0)->irecv<std::vector<double>>();
        mpi::waitall(number_request, message_request, numbers_request);

        auto number = number_request->get();
        std::cout << "Number: " << number << '\n';
        auto message = message_request->get();
        std::cout << "Message: " << message;
        auto numbers = numbers_request->get();
        std::cout << "Data: ";
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    return 0;
}