#include <mpiwrap/mpi.h>

//goal 1
int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto world_size = mpi::comm("world")->size();
    auto world_rank = mpi::comm("world")->rank();
    auto name = mpi::comm("world")->name();

    auto processor_name = mpi::processor_name();
    std::cout << "Hello world from processor " << processor_name << ", rank " << world_rank << " out of " << world_size << " processors,\n"
              << "with comm " << name << '\n';

    return 0;
}

//goal 2
// int main(int argc, char **argv)
// {
//     mpi::mpi init{argc, argv};

//     auto rank = mpi::comm("world")->rank();
//     if (rank == 0)
//     {
//         mpi::comm("world")->dest(1)->send("Hello, there\n");
//         mpi::comm("world")->dest(1)->send(size_t{5});
//         mpi::comm("world")->dest(2)->send("Hi, there\n");
//         mpi::comm("world")->dest(2)->send(std::vector<double>{3.4, 5.0, 6.9});
//     }
//     else if (rank == 2)
//     {
//         //result
//         auto message = mpi::comm("world")->source(0)->recv<std::string>();
//         std::cout << message;
//         auto numbers = mpi::comm("world")->source(0)->recv<std::vector<double>>();
//         for (auto &&number : numbers)
//             std::cout << number << ' ';
//         std::cout << '\n';
//     }
//     else if (rank == 1)
//     {
//         //output argument
//         auto message = std::string{};
//         mpi::comm("world")->source(0)->recv(message);
//         std::cout << message;
//         auto number = size_t{};
//         mpi::comm("world")->source(0)->recv(number);
//         std::cout << number << '\n';
//     }

//     return 0;
// }


//goal 6
// int main(int argc, char **argv)
// {
//     mpi::mpi init{argc, argv};

//     auto rank = mpi::comm("world")->rank();

//     auto number = mpi::comm("world")->dest(0)->reduce(rank, MPI_SUM);
//     if (rank == 0)
//         std::cout << number << '\n';

//     number = mpi::comm("world")->dest(0)->reduce(rank, MPI_MIN);
//     if (rank == 0)
//         std::cout << number << '\n';

//     number = mpi::comm("world")->dest(0)->reduce(rank, MPI_MAX);
//     if (rank == 0)
//         std::cout << number << '\n';

//     auto numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, MPI_SUM);
//     if (rank == 0)
//     {
//         for (auto &&number : numbers)
//             std::cout << number << ' ';
//         std::cout << '\n';
//     }

//     numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, MPI_MIN);
//     if (rank == 0)
//     {
//         for (auto &&number : numbers)
//             std::cout << number << ' ';
//         std::cout << '\n';
//     }

//     numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, MPI_MAX);
//     if (rank == 0)
//     {
//         for (auto &&number : numbers)
//             std::cout << number << ' ';
//         std::cout << '\n';
//     }
//     numbers = mpi::comm("world")->allreduce(std::vector<int>{rank, 5 - rank}, MPI_MAX);
//     for (auto &&number : numbers)
//         std::cout << number << ' ';
//     std::cout << '\n';

//     return 0;
// }
