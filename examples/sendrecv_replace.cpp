#include <mpiwrap/mpi.h>
#include <iostream>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();
    auto size = mpi::comm("world")->size();
    auto left = (rank > 0) ? (rank - 1) : (size - 1);
    auto right = (rank + 1) % size;

    auto number = (rank == 0) ? 1 : 0;
    if (number == 1)
        std::cout << "Process: " << rank << "   Number: " << number << '\n';
    mpi::comm("world")->source(left)->dest(right)->sendrecv_replace(number);
    if (number == 1)
        std::cout << "Process: " << rank << "   Number: " << number << '\n';
    mpi::comm("world")->source(left)->dest(right)->sendrecv_replace(number);
    if (number == 1)
        std::cout << "Process: " << rank << "   Number: " << number << '\n';

    auto numbers = (rank == 0) ? std::vector<int>{1, 2, 3, 4, 5} : std::vector<int>{};
    if (!numbers.empty())
    {
        std::cout << "Process: " << rank << "   Data: ";
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }
    mpi::comm("world")->dest(left)->source(right)->sendrecv_replace(numbers);
    if (!numbers.empty())
    {
        std::cout << "Process: " << rank << "   Data: ";
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }
    mpi::comm("world")->dest(left)->source(right)->sendrecv_replace(numbers);
    if (!numbers.empty())
    {
        std::cout << "Process: " << rank << "   Data: ";
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    auto message = (rank == 0) ? std::string{"Hello, there"} : std::string{};
    if (!message.empty())
        std::cout << "Process: " << rank << "   Message: " << message << '\n';
    mpi::comm("world")->source(left)->dest(right)->sendrecv_replace(message);
    if (!message.empty())
        std::cout << "Process: " << rank << "   Message: " << message << '\n';
    mpi::comm("world")->source(left)->dest(right)->sendrecv_replace(message);
    if (!message.empty())
        std::cout << "Process: " << rank << "   Message: " << message << '\n';

    return 0;
}