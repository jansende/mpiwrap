#include "mpi.h"
#include <string>
#include <memory>
#include <iostream>
#include <array>
#include <vector>
#include <cstring>

namespace mpi
{
//NOTE: strings might not properly work for mixed string implementations
class sender;
class receiver;
class communicator
{
private:
    MPI_Comm _comm;

public:
    communicator(MPI_Comm _comm) : _comm(_comm) {}

    auto size() -> int
    {
        auto _size = int{};
        //add error checking
        MPI_Comm_size(_comm, &_size);
        return _size;
    }
    auto rank() -> int
    {
        auto _rank = int{};
        //add error checking
        MPI_Comm_rank(_comm, &_rank);
        return _rank;
    }
    auto dest(int _dest)
    {
        auto _tag = 0;
        return std::make_unique<sender>(_dest, _tag, _comm);
    }
    auto source(int _source)
    {
        auto _tag = 0;
        return std::make_unique<receiver>(_source, _tag, _comm);
    }
    //allgather
};
auto comm(MPI_Comm _comm)
{
    return std::make_unique<communicator>(_comm);
}
auto comm(const std::string &_name)
{
    if (_name == std::string{"world"})
        return comm(MPI_COMM_WORLD);
    else
        throw;
}
auto processor_name()
{
    auto name = std::array<char, MPI_MAX_PROCESSOR_NAME>{};
    auto size = static_cast<int>(name.size());
    //add error checking
    MPI_Get_processor_name(name.data(), &size);
    return std::string{name.data()};
}

class mpi
{
public:
    mpi(const mpi &) = delete;
    mpi(mpi &&) = delete;
    mpi &operator=(const mpi &) = delete;
    mpi(int argc, char **argv)
    {
        //add error checking
        MPI_Init(&argc, &argv);
    }
    ~mpi()
    {
        //add error checking
        MPI_Finalize();
    }
};
template <class>
struct type_wrapper
{
    operator MPI_Datatype() const { return MPI_DATATYPE_NULL; }
};
template <>
struct type_wrapper<bool>
{
    operator MPI_Datatype() const { return MPI_CXX_BOOL; }
};
template <>
struct type_wrapper<char>
{
    operator MPI_Datatype() const { return MPI_CHAR; }
};
template <>
struct type_wrapper<signed char>
{
    operator MPI_Datatype() const { return MPI_SIGNED_CHAR; }
};
template <>
struct type_wrapper<unsigned char>
{
    operator MPI_Datatype() const { return MPI_UNSIGNED_CHAR; }
};
template <>
struct type_wrapper<short int>
{
    operator MPI_Datatype() const { return MPI_SHORT; }
};
template <>
struct type_wrapper<unsigned short int>
{
    operator MPI_Datatype() const { return MPI_UNSIGNED_SHORT; }
};
template <>
struct type_wrapper<int>
{
    operator MPI_Datatype() const { return MPI_INT; }
};
template <>
struct type_wrapper<unsigned int>
{
    operator MPI_Datatype() const { return MPI_UNSIGNED; }
};
template <>
struct type_wrapper<long int>
{
    operator MPI_Datatype() const { return MPI_LONG; }
};
template <>
struct type_wrapper<unsigned long int>
{
    operator MPI_Datatype() const { return MPI_UNSIGNED_LONG; }
};
template <>
struct type_wrapper<long long int>
{
    operator MPI_Datatype() const { return MPI_LONG_LONG; }
};
template <>
struct type_wrapper<unsigned long long int>
{
    operator MPI_Datatype() const { return MPI_UNSIGNED_LONG_LONG; }
};
template <>
struct type_wrapper<float>
{
    operator MPI_Datatype() const { return MPI_FLOAT; }
};
template <>
struct type_wrapper<double>
{
    operator MPI_Datatype() const { return MPI_DOUBLE; }
};
template <>
struct type_wrapper<long double>
{
    operator MPI_Datatype() const { return MPI_LONG_DOUBLE; }
};

template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const T _value)
{
    MPI_Send(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm);
}
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value)
{
    MPI_Send(_value.c_str(), _value.size() + 1, MPI_CHAR, _dest, _tag, _comm);
}
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value)
{
    send_impl(_dest, _tag, _comm, std::string{_value});
}
template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value)
{
    MPI_Send(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm);
}

class sender
{
private:
    int _dest;
    int _tag;
    MPI_Comm _comm;

public:
    template <class T>
    auto send(const T _value)
    {
        send_impl(_dest, _tag, _comm, _value);
    }

    sender(int _dest, int _tag, MPI_Comm _comm) : _dest(_dest), _tag(_tag), _comm(_comm) {}
    //gather
};

template <class T>
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, T &_value)
{
    MPI_Recv(&_value, 1, type_wrapper<T>{}, _source, _tag, _comm, _status);
}
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::string &_value)
{
    //we need to finde the proper size of the incoming string
    MPI_Probe(_source, _tag, _comm, _status);
    auto _size = int{};
    MPI_Get_count(_status, MPI_CHAR, &_size);
    //we need to allocate some memory for it
    auto _c_str = std::make_unique<char[]>(_size);
    //we need to receive it
    MPI_Recv(_c_str.get(), _size, MPI_CHAR, _source, _tag, _comm, _status);
    //we need to write the value back
    _value = std::string{_c_str.get()};
}
template <class T>
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::vector<T> &_value)
{
    //we need to finde the proper size of the incoming data
    MPI_Probe(_source, _tag, _comm, _status);
    auto _size = int{};
    MPI_Get_count(_status, type_wrapper<T>{}, &_size);
    //we need to allocate some memory for it
    _value.resize(_size);
    //we need to receive it
    MPI_Recv(_value.data(), _size, type_wrapper<T>{}, _source, _tag, _comm, _status);
}
template <class T>
auto bcast_impl(int _source, MPI_Comm _comm, T &_value)
{
    MPI_Bcast(&_value, 1, type_wrapper<T>{}, _source, _comm);
}
auto bcast_impl(int _source, MPI_Comm _comm, std::string &_value)
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the size before the string
    auto _size = (_rank == _source) ? static_cast<int>(_value.size()) + 1 : int{};
    bcast_impl(_source, _comm, _size);
    //we need to allocate some memory for it
    auto _c_str = std::make_unique<char[]>(_size);
    //we need to copy the data into the right place
    if (_rank == _source)
        std::strcpy(_c_str.get(), _value.c_str());
    //broadcast the data
    MPI_Bcast(_c_str.get(), _size, MPI_CHAR, _source, _comm);
    //at we need to write the value back
    _value = std::string{_c_str.get()};
}
template <class T>
auto bcast_impl(int _source, MPI_Comm _comm, std::vector<T> &_value)
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the size before the data
    auto _size = (_rank == _source) ? static_cast<int>(_value.size()) : int{};
    bcast_impl(_source, _comm, _size);
    //resize the vector if not the sender
    if (_rank != _source)
        _value.resize(_size);
    //broadcast the data
    MPI_Bcast(_value.data(), _size, type_wrapper<T>{}, _source, _comm);
}

class receiver
{
private:
    int _source;
    int _tag;
    MPI_Comm _comm;
    MPI_Status _status;

public:
    receiver(int _source, int _tag, MPI_Comm _comm) : _source(_source), _tag(_tag), _comm(_comm) {}

    template <class T>
    auto recv()
    {
        auto _value = T{};
        recv(_value);
        return _value;
    }
    template <class T>
    auto recv(T &_value)
    {
        recv_impl(_source, _tag, _comm, &_status, _value);
    }
    template <class T>
    auto bcast(T &_value)
    {
        bcast_impl(_source, _comm, _value);
    }
    //scatter
};

} // namespace mpi

//goal 1
// int main(int argc, char **argv)
// {
//     mpi::mpi init{argc, argv};

//     auto world_size = mpi::comm("world")->size();
//     auto world_rank = mpi::comm("world")->rank();

//     auto processor_name = mpi::processor_name();
//     std::cout << "Hello world from processor " << processor_name << ", rank " << world_rank << " out of " << world_size << " processors\n";

//     return 0;
// }

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
int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto rank = mpi::comm("world")->rank();

    auto numbers = std::vector<double>{};
    if (rank == 0)
    {
        numbers.emplace_back(4.0);
        numbers.emplace_back(7.1);
        numbers.emplace_back(8.9);
        numbers.emplace_back(42.);
    }
    mpi::comm("world")->source(0)->bcast(numbers);
    for (auto &&number : numbers)
        std::cout << number << ' ';
    std::cout << '\n';

    auto message = std::string{};
    if (rank == 0)
    {
        message = "Hello MPI";
    }
    mpi::comm("world")->source(0)->bcast(message);
    std::cout << message << '\n';

    return 0;
}

scatter
gather
allgather