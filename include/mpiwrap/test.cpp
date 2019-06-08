#include "mpiwrap.h"

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

//goal 3
// int main(int argc, char **argv)
// {
//     mpi::mpi init{argc, argv};

//     auto rank = mpi::comm("world")->rank();

//     auto numbers = std::vector<double>{};
//     if (rank == 0)
//     {
//         numbers.emplace_back(4.0);
//         numbers.emplace_back(7.1);
//         numbers.emplace_back(8.9);
//         numbers.emplace_back(42.);
//     }
//     mpi::comm("world")->source(0)->bcast(numbers);
//     for (auto &&number : numbers)
//         std::cout << number << ' ';
//     std::cout << '\n';

//     auto message = std::string{};
//     if (rank == 0)
//     {
//         message = "Hello MPI";
//     }
//     mpi::comm("world")->source(0)->bcast(message);
//     std::cout << message << '\n';

//     return 0;
// }

// //goal 4
// int main(int argc, char **argv)
// {
//     mpi::mpi init{argc, argv};

//     auto rank = mpi::comm("world")->rank();

//     auto numbers = std::vector<double>{};
//     if (rank == 0)
//     {
//         numbers.emplace_back(4.0);
//         numbers.emplace_back(7.1);
//         numbers.emplace_back(8.9);
//         numbers.emplace_back(42.);
//         numbers.emplace_back(3.4);
//         numbers.emplace_back(-11.);
//     }
//     numbers = mpi::comm("world")->source(0)->scatter(numbers, 2);
//     for (auto &&number : numbers)
//         std::cout << number << ' ';
//     std::cout << '\n';
//     numbers = mpi::comm("world")->dest(0)->gather(numbers);
//     for (auto &&number : numbers)
//         std::cout << number << ' ';
//     std::cout << '\n';
//     numbers = mpi::comm("world")->source(0)->scatter(numbers, 2);
//     numbers = mpi::comm("world")->allgather(numbers);
//     for (auto &&number : numbers)
//         std::cout << number << ' ';
//     std::cout << '\n';

//     auto message = std::string{};
//     if (rank == 0)
//     {
//         message = "Hello MPI";
//     }
//     message = mpi::comm("world")->source(0)->scatter(message, 3);
//     std::cout << message << '\n';
//     message = mpi::comm("world")->dest(0)->gather(message);
//     std::cout << message << '\n';
//     message = mpi::comm("world")->source(0)->scatter(message, 3);
//     message = mpi::comm("world")->allgather(message);
//     std::cout << message << '\n';

//     return 0;
// }

//goal 5
// int main(int argc, char **argv)
// {
//     mpi::mpi init{argc, argv};

//     auto rank = mpi::comm("world")->rank();

//     auto numbers = std::vector<double>{};
//     numbers = mpi::comm("world")->dest(0)->gather(42.0);
//     for (auto &&number : numbers)
//         std::cout << number << ' ';
//     std::cout << '\n';
//     numbers = mpi::comm("world")->source(0)->scatter(numbers, 2);
//     numbers = mpi::comm("world")->allgather(7.0);
//     for (auto &&number : numbers)
//         std::cout << number << ' ';
//     std::cout << '\n';

//     auto message = std::string{};
//     message = mpi::comm("world")->dest(0)->gather('d');
//     std::cout << message << '\n';
//     message = mpi::comm("world")->source(0)->scatter(message, 3);
//     message = mpi::comm("world")->allgather('c');
//     std::cout << message << '\n';

//     message = mpi::comm("world")->dest(0)->gather("hi");
//     std::cout << message << '\n';
//     message = mpi::comm("world")->source(0)->scatter(message, 3);
//     message = mpi::comm("world")->allgather("ho");
//     std::cout << message << '\n';

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
