#include <mpiwrap/mpi.h>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();
    if (rank == 0)
    {
        mpi::comm("world")->dest(1)->send("Hello, there\n");
        mpi::comm("world")->dest(1)->send(std::vector<double>{1.0, 2.0, 3.0});
    }
    else if (rank == 1)
    {
        auto message = mpi::comm("world")->source(0)->recv<std::string>();
        std::cout << "Message: " << message;

        auto numbers = mpi::comm("world")->source(0)->recv<std::vector<double>>();
        std::cout << "Data: ";
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    return 0;
}