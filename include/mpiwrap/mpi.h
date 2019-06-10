#pragma once
#include <mpi.h>
#include <string>
#include <memory>
#include <array>
#include <vector>
#include <cstring>
#include <cassert>
#include <functional>
#include <algorithm>
#include <mpiwrap/impl/lambda_hack.h>

#ifdef BE_PARANOID
#define paranoidly_assert(condition) assert(condition)
#else
#define paranoidly_assert(condition) ((void)0)
#endif

namespace mpi
{
class version_info
{
private:
    int _version;
    int _subversion;

public:
    version_info()
    {
        paranoidly_assert((initialized()));
        paranoidly_assert((!finalized()));

        MPI_Get_version(&_version, &_subversion);
    }
    auto version() { return _version; }
    auto subversion() { return _subversion; }
};
auto version()
{
    return version_info{};
}
template <class T, class Op>
class op_proxy
{
private:
    MPI_Op _operation;
    const bool _commute;

    static auto wrapper(void *void_a, void *void_b, int *len, MPI_Datatype *) -> void
    {
        auto a = static_cast<T *>(void_a);
        auto b = static_cast<T *>(void_b);
        auto op = impl::lambda_hack_impl<Op>{}.get();
        std::transform(a, a + *len, b, b, op);
    }

public:
    op_proxy(const op_proxy &) = delete;
    op_proxy(op_proxy &&) = delete;
    op_proxy &operator=(const op_proxy &) = delete;

    op_proxy(Op, const bool _commute) : _commute(_commute)
    {
        paranoidly_assert((initialized()));
        paranoidly_assert((!finalized()));

        MPI_Op_create(op_proxy<T, Op>::wrapper, _commute, &_operation);
    }
    ~op_proxy()
    {
        paranoidly_assert((initialized()));
        paranoidly_assert((!finalized()));

        MPI_Op_free(&_operation);
    }

    auto op() const -> const MPI_Op &
    {
        return _operation;
    }
    auto commutes() const
    {
        return _commute;
    }
};

template <class T, class Op>
auto make_op(Op _func, const bool _commute = false) -> std::unique_ptr<op_proxy<T, Op>>
{
    return std::make_unique<op_proxy<T, Op>>(_func, _commute);
}

auto initialized() -> bool
{
    auto flag = int{};
    MPI_Initialized(&flag);
    return flag == true;
}
auto finalized() -> bool
{
    auto flag = int{};
    MPI_Finalized(&flag);
    return flag == true;
}
auto processor_name() -> std::string
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    auto name = std::array<char, MPI_MAX_PROCESSOR_NAME>{};
    auto size = static_cast<int>(name.size());
    //add error checking
    MPI_Get_processor_name(name.data(), &size);
    return std::string{name.data()};
}
//NOTE: strings might not properly work for mixed string implementations
class sender;
class receiver;

auto allgather_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void;
auto allgather_impl(MPI_Comm _comm, const char *_value, std::string &_bucket) -> void;
template <class T>
auto allgather_impl(MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket) -> void;

template <class T>
auto allreduce_impl(MPI_Comm _comm, const T &_value, T &_bucket, MPI_Op _operation) -> void;
auto allreduce_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void;
auto allreduce_impl(MPI_Comm _comm, const char *_value, std::string &_bucket, MPI_Op _operation) -> void;
template <class T>
auto allreduce_impl(MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket, MPI_Op _operation) -> void;

class communicator
{
protected:
    MPI_Comm _comm;

public:
    communicator(MPI_Comm _comm) : _comm(_comm) {}

    enum class comp
    {
        ident,
        congruent,
        similar,
        unequal,
    };
    static constexpr auto ident = comp::ident;
    static constexpr auto congruent = comp::congruent;
    static constexpr auto similar = comp::similar;
    static constexpr auto unequal = comp::unequal;

    friend auto compare(const communicator &lhs, const MPI_Comm &rhs) -> comp;
    friend auto compare(const MPI_Comm &lhs, const communicator &rhs) -> comp;
    friend auto compare(const communicator &lhs, const communicator &rhs) -> comp;

    auto operator==(const communicator &rhs) -> bool;
    auto operator!=(const communicator &rhs) -> bool;
    auto operator==(const MPI_Comm &rhs) -> bool;
    auto operator!=(const MPI_Comm &rhs) -> bool;

    auto size() -> int
    {
        paranoidly_assert((initialized()));
        paranoidly_assert((!finalized()));
        auto _size = int{};
        //add error checking
        MPI_Comm_size(_comm, &_size);
        return _size;
    }
    auto rank() -> int
    {
        paranoidly_assert((initialized()));
        paranoidly_assert((!finalized()));
        auto _rank = int{};
        //add error checking
        MPI_Comm_rank(_comm, &_rank);
        return _rank;
    }
    auto name() -> std::string
    {
        paranoidly_assert((initialized()));
        paranoidly_assert((!finalized()));
        auto _name = std::make_unique<char[]>(MPI_MAX_OBJECT_NAME);
        auto _size = MPI_MAX_OBJECT_NAME;
        //add error checking
        MPI_Comm_get_name(_comm, _name.get(), &_size);
        return std::string{_name.get()};
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
        paranoidly_assert((initialized()));
        paranoidly_assert((!finalized()));
        MPI_Barrier(_comm);
    }

    template <class T>
    auto allreduce(const T &_value, T &_bucket, MPI_Op _operation) -> void
    {
        allreduce_impl(_comm, _value, _bucket, _operation);
    }
    template <class T>
    auto allreduce(const std::vector<T> &_value, MPI_Op _operation) -> std::vector<T>
    {
        auto _bucket = std::vector<T>{};
        allreduce(_value, _bucket, _operation);
        return _bucket;
    }
    auto allreduce(const std::string &_value, MPI_Op _operation) -> std::string
    {
        auto _bucket = std::string{};
        allreduce(_value, _bucket, _operation);
        return _bucket;
    }
    template <class T>
    auto allreduce(const T _value, MPI_Op _operation) -> T
    {
        auto _bucket = T{};
        allreduce(_value, _bucket, _operation);
        return _bucket;
    }
    auto allreduce(const char *_value, MPI_Op _operation) -> std::string
    {
        return allreduce(std::string{_value}, _operation);
    }

    template <class T, class Op>
    auto allreduce(const T &_value, T &_bucket, const op_proxy<T, Op> *_operation) -> void
    {
        return allreduce(_value, _bucket, _operation->op());
    }
    template <class T, class Op>
    auto allreduce(const std::vector<T> &_value, const op_proxy<T, Op> *_operation) -> std::vector<T>
    {
        return allreduce(_value, _operation->op());
    }
    template <class Op>
    auto allreduce(const std::string &_value, const op_proxy<std::string, Op> *_operation) -> std::string
    {
        return allreduce(_value, _operation->op());
    }
    template <class T, class Op>
    auto allreduce(const T _value, const op_proxy<T, Op> *_operation) -> T
    {
        return allreduce(_value, _operation->op());
    }
    template <class Op>
    auto allreduce(const char *_value, const op_proxy<std::string, Op> *_operation) -> std::string
    {
        return allreduce(_value, _operation->op());
    }
};
auto compare(const MPI_Comm &lhs, const MPI_Comm &rhs) -> communicator::comp
{
    auto _result = int{};
    MPI_Comm_compare(lhs, rhs, &_result);
    switch (_result)
    {
    case MPI_IDENT:
        return communicator::ident;
    case MPI_CONGRUENT:
        return communicator::congruent;
    case MPI_SIMILAR:
        return communicator::similar;
    default:
        return communicator::unequal;
    }
}
auto compare(const communicator &lhs, const MPI_Comm &rhs) -> communicator::comp
{

    return compare(lhs._comm, rhs);
}
auto compare(const MPI_Comm &lhs, const communicator &rhs) -> communicator::comp
{
    return compare(lhs, rhs._comm);
}
auto compare(const communicator &lhs, const communicator &rhs) -> communicator::comp
{
    return compare(lhs._comm, rhs._comm);
}
auto communicator::operator==(const communicator &rhs) -> bool
{
    return compare(*this, rhs) == communicator::ident;
}
auto communicator::operator!=(const communicator &rhs) -> bool
{
    return !(*this == rhs);
}
auto communicator::operator==(const MPI_Comm &rhs) -> bool
{
    return compare(*this, rhs) == communicator::ident;
}
auto communicator::operator!=(const MPI_Comm &rhs) -> bool
{
    return !(*this == rhs);
}
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
        paranoidly_assert((!initialized()));
        paranoidly_assert((!finalized()));
        //add error checking
        MPI_Init(&argc, &argv);
    }
    ~mpi()
    {
        paranoidly_assert((initialized()));
        paranoidly_assert((!finalized()));
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

    auto operator==(const receiver &rhs) -> bool
    {
        return _source == rhs._source && _tag == rhs._tag && _comm == rhs._comm;
    }
    auto operator!=(const receiver &rhs) -> bool
    {
        return !(*this == rhs);
    }

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
template <class T>
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void;
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void;
template <class T>
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void;
template <class T>
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void;
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void;
template <class T>
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void;

auto gather_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void;
auto gather_impl(int _dest, MPI_Comm _comm, const char *_value, std::string &_bucket) -> void;
template <class T>
auto gather_impl(int _dest, MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket) -> void;

template <class T>
auto reduce_impl(int _dest, MPI_Comm _comm, const T &_value, T &_bucket, MPI_Op _operation) -> void;
auto reduce_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void;
auto reduce_impl(int _dest, MPI_Comm _comm, const char *_value, std::string &_bucket, MPI_Op _operation) -> void;
template <class T>
auto reduce_impl(int _dest, MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket, MPI_Op _operation) -> void;

class sender
{
private:
    int _dest;
    int _tag;
    MPI_Comm _comm;

public:
    sender(int _dest, int _tag, MPI_Comm _comm) : _dest(_dest), _tag(_tag), _comm(_comm) {}

    auto operator==(const sender &rhs) -> bool
    {
        return _dest == rhs._dest && _tag == rhs._tag && _comm == rhs._comm;
    }
    auto operator!=(const sender &rhs) -> bool
    {
        return !(*this == rhs);
    }

    template <class T>
    auto send(const T _value) -> void
    {
        send_impl(_dest, _tag, _comm, _value);
    }
    template <class T>
    auto ssend(const T _value) -> void
    {
        ssend_impl(_dest, _tag, _comm, _value);
    }
    template <class T>
    auto rsend(const T _value) -> void
    {
        rsend_impl(_dest, _tag, _comm, _value);
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

    template <class T>
    auto reduce(const T &_value, T &_bucket, MPI_Op _operation) -> void
    {
        reduce_impl(_dest, _comm, _value, _bucket, _operation);
    }
    template <class T>
    auto reduce(const std::vector<T> &_value, MPI_Op _operation) -> std::vector<T>
    {
        auto _bucket = std::vector<T>{};
        reduce(_value, _bucket, _operation);
        return _bucket;
    }
    auto reduce(const std::string &_value, MPI_Op _operation) -> std::string
    {
        auto _bucket = std::string{};
        reduce(_value, _bucket, _operation);
        return _bucket;
    }
    template <class T>
    auto reduce(const T _value, MPI_Op _operation) -> T
    {
        auto _bucket = T{};
        reduce(_value, _bucket, _operation);
        return _bucket;
    }
    auto reduce(const char *_value, MPI_Op _operation) -> std::string
    {
        return reduce(std::string{_value}, _operation);
    }

    template <class T, class Op>
    auto reduce(const T &_value, T &_bucket, const op_proxy<T, Op> *_operation) -> void
    {
        return reduce(_value, _bucket, _operation->op());
    }
    template <class T, class Op>
    auto reduce(const std::vector<T> &_value, const op_proxy<T, Op> *_operation) -> std::vector<T>
    {
        return reduce(_value, _operation->op());
    }
    template <class Op>
    auto reduce(const std::string &_value, const op_proxy<std::string, Op> *_operation) -> std::string
    {
        return reduce(_value, _operation->op());
    }
    template <class T, class Op>
    auto reduce(const T _value, const op_proxy<T, Op> *_operation) -> T
    {
        return reduce(_value, _operation->op());
    }
    template <class Op>
    auto reduce(const char *_value, const op_proxy<std::string, Op> *_operation) -> std::string
    {
        return reduce(_value, _operation->op());
    }
};

auto allgather_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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
auto allreduce_impl(MPI_Comm _comm, const T &_value, T &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce(&_value, &_bucket, 1, type_wrapper<T>{}, _operation, _comm);
}
auto allreduce_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the chunk_size before the string is reduced
    auto _size = (_rank == 0) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(0)->bcast(_size);
    //check size
    assert((_value.size() == _size));
    //we need to allocate some memory for the result
    auto _c_str = std::make_unique<char[]>((_rank == 0) ? _size + 1 : 0);
    //reduce the data
    MPI_Allreduce(_value.c_str(), _c_str.get(), _size, MPI_CHAR, _operation, _comm);
    //we need to write the value back
    _bucket = std::string{_c_str.get()};
}
auto allreduce_impl(MPI_Comm _comm, const char *_value, std::string &_bucket, MPI_Op _operation) -> void
{
    allreduce_impl(_comm, std::string{_value}, _bucket, _operation);
}
template <class T>
auto allreduce_impl(MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the chunk_size before the data is reduced
    auto _size = (_rank == 0) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(0)->bcast(_size);
    //check size
    assert((_value.size() == _size));
    //resize bucket to take all elements
    if (_size != _bucket.size())
        _bucket.resize(_size);
    //reduce the data
    MPI_Allreduce(_value.data(), _bucket.data(), _size, type_wrapper<T>{}, _operation, _comm);
}

template <class T>
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, T &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Recv(&_value, 1, type_wrapper<T>{}, _source, _tag, _comm, _status);
}
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Bcast(&_value, 1, type_wrapper<T>{}, _source, _comm);
}
auto bcast_impl(int _source, MPI_Comm _comm, std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //create result container
    auto _chunk = std::vector<T>(_chunk_size);
    //scatter the data
    MPI_Scatter(_value.data(), _chunk_size, type_wrapper<T>{}, _chunk.data(), _chunk_size, type_wrapper<T>{}, _source, _comm);
    return _chunk;
}

template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Send(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm);
}
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Send(_value.c_str(), _value.size() + 1, MPI_CHAR, _dest, _tag, _comm);
}
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    send_impl(_dest, _tag, _comm, std::string{_value});
}
template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Send(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm);
}
template <class T>
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Ssend(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm);
}
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Ssend(_value.c_str(), _value.size() + 1, MPI_CHAR, _dest, _tag, _comm);
}
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void
{
    ssend_impl(_dest, _tag, _comm, std::string{_value});
}
template <class T>
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Ssend(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm);
}
template <class T>
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Rsend(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm);
}
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Rsend(_value.c_str(), _value.size() + 1, MPI_CHAR, _dest, _tag, _comm);
}
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void
{
    rsend_impl(_dest, _tag, _comm, std::string{_value});
}
template <class T>
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Rsend(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm);
}

auto gather_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
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

template <class T>
auto reduce_impl(int _dest, MPI_Comm _comm, const T &_value, T &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce(&_value, &_bucket, 1, type_wrapper<T>{}, _operation, _dest, _comm);
}
auto reduce_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the chunk_size before the string is reduced
    auto _size = (_rank == _dest) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(_dest)->bcast(_size);
    //check size
    assert((_value.size() == _size));
    //we need to allocate some memory for the result
    auto _c_str = std::make_unique<char[]>((_rank == _dest) ? _size + 1 : 0);
    //reduce the data
    MPI_Reduce(_value.c_str(), _c_str.get(), _size, MPI_CHAR, _operation, _dest, _comm);
    //we need to write the value back
    if (_rank == _dest)
        _bucket = std::string{_c_str.get()};
}
auto reduce_impl(int _dest, MPI_Comm _comm, const char *_value, std::string &_bucket, MPI_Op _operation) -> void
{
    reduce_impl(_dest, _comm, std::string{_value}, _bucket, _operation);
}
template <class T>
auto reduce_impl(int _dest, MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the chunk_size before the data is reduced
    auto _size = (_rank == _dest) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(_dest)->bcast(_size);
    //check size
    assert((_value.size() == _size));
    //resize bucket to take all elements
    if (_rank == _dest)
    {
        if (_size != _bucket.size())
            _bucket.resize(_size);
    }
    //reduce the data
    MPI_Reduce(_value.data(), _bucket.data(), _size, type_wrapper<T>{}, _operation, _dest, _comm);
}

} // namespace mpi