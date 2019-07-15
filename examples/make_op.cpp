#include <mpiwrap/mpi.h>
#include <iostream>
#include <functional>

int sum(int a, int b)
{
    return a + b;
}
struct plus
{
    template <class T>
    auto operator()(T &&lhs, T &&rhs) const -> decltype(std::forward<T>(lhs) + std::forward<T>(rhs))
    {
        return std::forward<T>(lhs) + std::forward<T>(rhs);
    }
};
int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();
    auto numbers = std::vector<int>{};

    //pure lambda operation
    auto lambda = [](auto a, auto b) { return a + b; };
    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, lambda);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }
    
    //standard library operation
    auto stdlib = std::plus<int>{}; //we can even use std::plus{} from c++17 onwards
    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, stdlib);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    //function access
    auto function1 = [](auto a, auto b) { return sum(a, b); };
    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, function1);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }
    auto function2 = mpi::wrap<int, sum>;
    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, function2);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    //functors
    auto functor = plus{};
    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, functor);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }

    //builtin
    auto builtin = mpi::sum;
    numbers = mpi::comm("world")->dest(0)->reduce(std::vector<int>{rank, 5 - rank}, builtin);
    if (rank == 0)
    {
        for (auto &&number : numbers)
            std::cout << number << ' ';
        std::cout << '\n';
    }
    
    return 0;
}