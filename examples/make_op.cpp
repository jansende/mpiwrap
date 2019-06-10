#include <mpiwrap/mpi.h>
#include <iostream>

int sum(int a, int b)
{
    return a + b;
}
int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();

    //pure lambda operation
    auto lambda1 = mpi::make_op<int>([](auto a, auto b) { return a + b; });
    auto op = [](auto a, auto b) { return a + b; };
    auto lambda2 = mpi::make_op<int>(op);
    //function access
    auto function1 = mpi::make_op<int>([](auto a, auto b) { return sum(a, b); });
    auto function2 = mpi::make_op<int>(mpi::wrap<int, sum>);

    auto numbers = std::vector<int>{};
    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, lambda1.get());
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, lambda2.get());
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, function1.get());
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, function2.get());
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    return 0;
}