#pragma once
#include "mpi.h"
#include <string>
#include <memory>
#include <iostream>
#include <array>
#include <vector>
#include <cstring>
#include <cassert>

namespace mpi
{
auto processor_name() -> std::string
{
    auto name = std::array<char, MPI_MAX_PROCESSOR_NAME>{};
    auto size = static_cast<int>(name.size());
    //add error checking
    MPI_Get_processor_name(name.data(), &size);
    return std::string{name.data()};
}
enum class op
{
    max,    //Returns the maximum element.
    min,    //Returns the minimum element.
    sum,    //Sums the elements.
    prod,   //Multiplies all elements.
    land,   //Performs a logical and across the elements.
    lor,    //Performs a logical or across the elements.
    band,   //Performs a bitwise and across the bits of the elements.
    bor,    //Performs a bitwise or across the bits of the elements.
    maxloc, //Returns the maximum value and the rank of the process that owns it.
    minloc, //Returns the minimum value and the rank of the process that owns it.
};
//NOTE: strings might not properly work for mixed string implementations
class sender;
class receiver;

auto allgather_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void;
auto allgather_impl(MPI_Comm _comm, const char *_value, std::string &_bucket) -> void;
template <class T>
auto allgather_impl(MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket) -> void;

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

    auto dest(int _dest) -> std::unique_ptr<sender>
    {
        auto _tag = 0;
        return std::make_unique<sender>(_dest, _tag, _comm);
    }
    auto source(int _source) -> std::unique_ptr<receiver>
    {
        auto _tag = 0;
        return std::make_unique<receiver>(_source, _tag, _comm);
    }

    template <class T>
    auto allgather(const T &_value, T &_bucket) -> void
    {
        allgather_impl(_comm, _value, _bucket);
    }
    template <class T>
    auto allgather(const T _value, std::vector<T> &_bucket) -> void
    {
        allgather({_value}, _bucket);
    }
    template <class T>
    auto allgather(const std::vector<T> &_value) -> std::vector<T>
    {
        auto _bucket = std::vector<T>{};
        allgather(_value, _bucket);
        return _bucket;
    }
    auto allgather(const std::string &_value) -> std::string
    {
        auto _bucket = std::string{};
        allgather(_value, _bucket);
        return _bucket;
    }
    template <class T>
    auto allgather(const T _value) -> std::vector<T>
    {
        return allgather(std::vector<T>{_value});
    }
    auto allgather(const char *_value) -> std::string
    {
        return allgather(std::string{_value});
    }
    auto allgather(const char _value) -> std::string
    {
        return allgather(std::string{_value});
    }

    auto barrier() -> void
    {
        MPI_Barrier(_comm);
    }
};
auto comm(MPI_Comm _comm) -> std::unique_ptr<communicator>
{
    return std::make_unique<communicator>(_comm);
}
auto comm(const std::string &_name) -> std::unique_ptr<communicator>
{
    if (_name == std::string{"world"})
        return comm(MPI_COMM_WORLD);
    else
        throw;
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
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, T &_value) -> void;
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::string &_value) -> void;
template <class T>
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::vector<T> &_value) -> void;

template <class T>
auto bcast_impl(int _source, MPI_Comm _comm, T &_value) -> void;
auto bcast_impl(int _source, MPI_Comm _comm, std::string &_value) -> void;
template <class T>
auto bcast_impl(int _source, MPI_Comm _comm, std::vector<T> &_value) -> void;

auto scatter_impl(int _source, MPI_Comm _comm, const std::string &_value, const size_t _chunk_size) -> std::string;
template <class T>
auto scatter_impl(int _source, MPI_Comm _comm, const std::vector<T> &_value, const size_t _chunk_size) -> std::vector<T>;

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
    auto recv() -> T
    {
        auto _value = T{};
        recv(_value);
        return _value;
    }
    template <class T>
    auto recv(T &_value) -> void
    {
        recv_impl(_source, _tag, _comm, &_status, _value);
    }

    template <class T>
    auto bcast(T &_value) -> void
    {
        bcast_impl(_source, _comm, _value);
    }

    template <class T>
    auto scatter(T &_value, const size_t _chunk_size) -> T
    {
        return scatter_impl(_source, _comm, _value, _chunk_size);
    }
};

template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void;
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void;
template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void;

auto gather_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void;
auto gather_impl(int _dest, MPI_Comm _comm, const char *_value, std::string &_bucket) -> void;
template <class T>
auto gather_impl(int _dest, MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket) -> void;

class sender
{
private:
    int _dest;
    int _tag;
    MPI_Comm _comm;

public:
    sender(int _dest, int _tag, MPI_Comm _comm) : _dest(_dest), _tag(_tag), _comm(_comm) {}

    template <class T>
    auto send(const T _value) -> void
    {
        send_impl(_dest, _tag, _comm, _value);
    }

    template <class T>
    auto gather(const T &_value, T &_bucket) -> void
    {
        gather_impl(_dest, _comm, _value, _bucket);
    }
    template <class T>
    auto gather(const T _value, std::vector<T> &_bucket) -> void
    {
        gather({_value}, _bucket);
    }
    template <class T>
    auto gather(const std::vector<T> &_value) -> std::vector<T>
    {
        auto _bucket = std::vector<T>{};
        gather(_value, _bucket);
        return _bucket;
    }
    auto gather(const std::string &_value) -> std::string
    {
        auto _bucket = std::string{};
        gather(_value, _bucket);
        return _bucket;
    }
    template <class T>
    auto gather(const T _value) -> std::vector<T>
    {
        return gather(std::vector<T>{_value});
    }
    auto gather(const char *_value) -> std::string
    {
        return gather(std::string{_value});
    }
    auto gather(const char _value) -> std::string
    {
        return gather(std::string{_value});
    }
};

auto allgather_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //get world size
    auto _size = comm(_comm)->size();
    //broadcast the chunk_size before the data is gathered
    auto _chunk_size = (_rank == 0) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(0)->bcast(_chunk_size);
    //check size
    assert((_value.size() == _chunk_size));
    //we need to allocate some memory for the result
    auto _c_str = std::make_unique<char[]>(_size * _chunk_size + 1);
    //gather the data
    MPI_Allgather(_value.c_str(), _chunk_size, MPI_CHAR, _c_str.get(), _chunk_size, MPI_CHAR, _comm);
    //we need to write the value back
    _bucket = std::string{_c_str.get()};
}
auto allgather_impl(MPI_Comm _comm, const char *_value, std::string &_bucket) -> void
{
    allgather_impl(_comm, std::string{_value}, _bucket);
}
template <class T>
auto allgather_impl(MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket) -> void
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //get world size
    auto _size = comm(_comm)->size();
    //broadcast the chunk_size before the data is gathered
    auto _chunk_size = (_rank == 0) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(0)->bcast(_chunk_size);
    //check size
    assert((_value.size() == _chunk_size));
    //resize bucket to take all elements
    if (_size * _chunk_size != _bucket.size())
        _bucket.resize(_size * _chunk_size);
    //gather the data
    MPI_Allgather(_value.data(), _chunk_size, type_wrapper<T>{}, _bucket.data(), _chunk_size, type_wrapper<T>{}, _comm);
}

template <class T>
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, T &_value) -> void
{
    MPI_Recv(&_value, 1, type_wrapper<T>{}, _source, _tag, _comm, _status);
}
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::string &_value) -> void
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
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::vector<T> &_value) -> void
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
auto bcast_impl(int _source, MPI_Comm _comm, T &_value) -> void
{
    MPI_Bcast(&_value, 1, type_wrapper<T>{}, _source, _comm);
}
auto bcast_impl(int _source, MPI_Comm _comm, std::string &_value) -> void
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the size before the string
    auto _size = (_rank == _source) ? static_cast<int>(_value.size()) + 1 : int{};
    comm(_comm)->source(_source)->bcast(_size);
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
auto bcast_impl(int _source, MPI_Comm _comm, std::vector<T> &_value) -> void
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the size before the data
    auto _size = (_rank == _source) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(_source)->bcast(_size);
    //resize the vector if not the sender
    if (_rank != _source)
        _value.resize(_size);
    //broadcast the data
    MPI_Bcast(_value.data(), _size, type_wrapper<T>{}, _source, _comm);
}

auto scatter_impl(int _source, MPI_Comm _comm, const std::string &_value, const size_t _chunk_size) -> std::string
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //we need a copy of the original data
    auto _c_str = std::make_unique<char[]>(0);
    if (_rank == _source)
        std::strcpy(_c_str.get(), _value.c_str());
    //create result container
    auto _chunk = std::make_unique<char[]>(_chunk_size + 1);
    //scatter the data
    MPI_Scatter(_c_str.get(), _chunk_size, MPI_CHAR, _chunk.get(), _chunk_size + 1, MPI_CHAR, _source, _comm);
    //at we need to write the value back
    return std::string{_chunk.get()};
}
template <class T>
auto scatter_impl(int _source, MPI_Comm _comm, const std::vector<T> &_value, const size_t _chunk_size) -> std::vector<T>
{
    //create result container
    auto _chunk = std::vector<T>(_chunk_size);
    //scatter the data
    MPI_Scatter(_value.data(), _chunk_size, type_wrapper<T>{}, _chunk.data(), _chunk_size, type_wrapper<T>{}, _source, _comm);
    return _chunk;
}

template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void
{
    MPI_Send(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm);
}
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void
{
    MPI_Send(_value.c_str(), _value.size() + 1, MPI_CHAR, _dest, _tag, _comm);
}
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void
{
    send_impl(_dest, _tag, _comm, std::string{_value});
}
template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void
{
    MPI_Send(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm);
}

auto gather_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //get world size
    auto _size = comm(_comm)->size();
    //broadcast the chunk_size before the string is gathered
    auto _chunk_size = (_rank == _dest) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(_dest)->bcast(_chunk_size);
    //check size
    assert((_value.size() == _chunk_size));
    //we need to allocate some memory for the result
    auto _c_str = std::make_unique<char[]>((_rank == _dest) ? _size * _chunk_size + 1 : 0);
    //gather the data
    MPI_Gather(_value.c_str(), _chunk_size, MPI_CHAR, _c_str.get(), _chunk_size, MPI_CHAR, _dest, _comm);
    //we need to write the value back
    if (_rank == _dest)
        _bucket = std::string{_c_str.get()};
}
auto gather_impl(int _dest, MPI_Comm _comm, const char *_value, std::string &_bucket) -> void
{
    gather_impl(_dest, _comm, std::string{_value}, _bucket);
}
template <class T>
auto gather_impl(int _dest, MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket) -> void
{
    //get current rank
    auto _rank = comm(_comm)->rank();
    //get world size
    auto _size = comm(_comm)->size();
    //broadcast the chunk_size before the data is gathered
    auto _chunk_size = (_rank == _dest) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(_dest)->bcast(_chunk_size);
    //check size
    assert((_value.size() == _chunk_size));
    //resize bucket to take all elements
    if (_rank == _dest)
    {
        if (_size * _chunk_size != _bucket.size())
            _bucket.resize(_size * _chunk_size);
    }
    //gather the data
    MPI_Gather(_value.data(), _chunk_size, type_wrapper<T>{}, _bucket.data(), _chunk_size, type_wrapper<T>{}, _dest, _comm);
}

} // namespace mpi