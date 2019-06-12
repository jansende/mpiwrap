#pragma once
#include <algorithm>
#include <cassert>
#include <cstring>
#include <mpiwrap/impl/lambda_hack.h>

namespace mpi
{
#pragma region MPI all to all
//declarations
auto alltoall_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void;
auto alltoall_impl(MPI_Comm _comm, const char *_value, std::string &_bucket, const size_t _chunk_size) -> void;
//templates
template <class T>
auto alltoall_impl(MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the size before the data is gathered
    auto _size = (_rank == 0) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(0)->bcast(_size);
    //check size
    assert((_value.size() == _size));
    //double check size
    assert((_size >= _chunk_size * comm(_comm)->size()));
    //resize bucket to take all elements
    if (_size != _bucket.size())
        _bucket.resize(_size);
    //gather the data
    MPI_Alltoall(_value.data(), _chunk_size, type_wrapper<T>{}, _bucket.data(), _chunk_size, type_wrapper<T>{}, _comm);
}
#pragma endregion
#pragma region MPI reduce
//declarations
auto reduce_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void;
auto reduce_impl(int _dest, MPI_Comm _comm, const char *_value, std::string &_bucket, MPI_Op _operation) -> void;
//templates
template <class T>
auto reduce_impl(int _dest, MPI_Comm _comm, const T &_value, T &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce(&_value, &_bucket, 1, type_wrapper<T>{}, _operation, _dest, _comm);
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
#pragma endregion
#pragma region MPI local reduce
//declarations
auto reduce(const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void;
auto reduce(const char *_value, std::string &_bucket, MPI_Op _operation) -> void;

auto reduce(const std::string &_value, MPI_Op _operation) -> std::string;
auto reduce(const char *_value, MPI_Op _operation) -> std::string;
//templates
template <class T>
auto reduce(const T &_value, T &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce_local(&_value, &_bucket, 1, type_wrapper<T>{}, _operation);
}
template <class T>
auto reduce(const std::vector<T> &_value, std::vector<T> &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //resize bucket to take all elements
    if (_value.size() != _bucket.size())
        _bucket.resize(_value.size());
    //reduce the data
    MPI_Reduce_local(_value.data(), _bucket.data(), _value.size(), type_wrapper<T>{}, _operation);
}
template <class T>
auto reduce(const std::vector<T> &_value, MPI_Op _operation) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
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

template <class T, class Op>
auto reduce(const T &_value, T &_bucket, const op_proxy<T, Op> *_operation) -> void
{
    return reduce(_value, _bucket, _operation->op());
}
template <class Op>
auto reduce(const std::string &_value, std::string &_bucket, const op_proxy<std::string, Op> *_operation) -> void
{
    return reduce(_value, _bucket, _operation->op());
}
template <class Op>
auto reduce(const char *_value, std::string &_bucket, const op_proxy<std::string, Op> *_operation) -> void
{
    return reduce(_value, _bucket, _operation->op());
}
template <class T, class Op>
auto reduce(const std::vector<T> &_value, std::vector<T> &_bucket, const op_proxy<T, Op> *_operation) -> void
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
#pragma endregion
#pragma region MPI allreduce
//declarations
auto allreduce_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void;
auto allreduce_impl(MPI_Comm _comm, const char *_value, std::string &_bucket, MPI_Op _operation) -> void;
//templates
template <class T>
auto allreduce_impl(MPI_Comm _comm, const T &_value, T &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce(&_value, &_bucket, 1, type_wrapper<T>{}, _operation, _comm);
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
#pragma endregion
#pragma region MPI gather
//declarations
auto gather_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void;
auto gather_impl(int _dest, MPI_Comm _comm, const char *_value, std::string &_bucket) -> void;
//templates
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
#pragma endregion
#pragma region MPI allgather
//declarations
auto allgather_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void;
auto allgather_impl(MPI_Comm _comm, const char *_value, std::string &_bucket) -> void;
//templates
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
#pragma endregion
#pragma region MPI scatter
//declarations
auto scatter_impl(int _source, MPI_Comm _comm, const std::string &_value, const size_t _chunk_size) -> std::string;
//templates
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
#pragma endregion
#pragma region MPI send
//declarations
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void;
//templates
template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Send(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm);
}
template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Send(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm);
}
#pragma endregion
#pragma region MPI synchronized send
//declarations
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void;
//templates
template <class T>
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Ssend(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm);
}
template <class T>
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Ssend(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm);
}
#pragma endregion
#pragma region MPI ready mode send
//declarations
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void;
//templates
template <class T>
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const T _value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Rsend(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm);
}
template <class T>
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Rsend(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm);
}
#pragma endregion
#pragma region MPI receive
//declarations
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::string &_value) -> void;
//templates
template <class T>
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, T &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Recv(&_value, 1, type_wrapper<T>{}, _source, _tag, _comm, _status);
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
#pragma endregion
#pragma region MPI broadcast
//declarations
auto bcast_impl(int _source, MPI_Comm _comm, std::string &_value) -> void;
//templates
template <class T>
auto bcast_impl(int _source, MPI_Comm _comm, T &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Bcast(&_value, 1, type_wrapper<T>{}, _source, _comm);
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
#pragma endregion
#pragma region MPI Operation wrapper
template <class T, class Op>
auto op_proxy<T, Op>::wrapper(void *void_a, void *void_b, int *len, MPI_Datatype *) -> void
{
    auto a = static_cast<T *>(void_a);
    auto b = static_cast<T *>(void_b);
    auto op = impl::lambda_hack_impl<Op>{}.get();
    std::transform(a, a + *len, b, b, op);
}
template <class T, class Op>
op_proxy<T, Op>::op_proxy(Op, const bool _commute) : _commute(_commute)
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));

    MPI_Op_create(op_proxy<T, Op>::wrapper, _commute, &_operation);
}
template <class T, class Op>
op_proxy<T, Op>::~op_proxy()
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));

    MPI_Op_free(&_operation);
}
template <class T, class Op>
auto op_proxy<T, Op>::op() const -> const MPI_Op &
{
    return _operation;
}
template <class T, class Op>
auto op_proxy<T, Op>::commutes() const -> bool
{
    return _commute;
}

template <class T, class Op>
auto make_op(Op _func, const bool _commute) -> std::unique_ptr<op_proxy<T, Op>>
{
    return std::make_unique<op_proxy<T, Op>>(_func, _commute);
}
#pragma endregion
#pragma region MPI communicator
template <class T>
auto communicator::allgather(const T &_value, T &_bucket) -> void
{
    allgather_impl(_comm, _value, _bucket);
}
template <class T>
auto communicator::allgather(const T _value, std::vector<T> &_bucket) -> void
{
    allgather({_value}, _bucket);
}
template <class T>
auto communicator::allgather(const std::vector<T> &_value) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    allgather(_value, _bucket);
    return _bucket;
}

template <class T>
auto communicator::allgather(const T _value) -> std::vector<T>
{
    return allgather(std::vector<T>{_value});
}

template <class T>
auto communicator::alltoall(const T &_value, T &_bucket, const size_t _chunk_size) -> void
{
    alltoall_impl(_comm, _value, _bucket, _chunk_size);
}
template <class T>
auto communicator::alltoall(const T _value, std::vector<T> &_bucket, const size_t _chunk_size) -> void
{
    alltoall({_value}, _bucket, _chunk_size);
}
template <class T>
auto communicator::alltoall(const std::vector<T> &_value, const size_t _chunk_size) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    alltoall(_value, _bucket, _chunk_size);
    return _bucket;
}

template <class T>
auto communicator::alltoall(const T _value, const size_t _chunk_size) -> std::vector<T>
{
    return alltoall(std::vector<T>{_value}, _chunk_size);
}

template <class T>
auto communicator::allreduce(const T &_value, T &_bucket, MPI_Op _operation) -> void
{
    allreduce_impl(_comm, _value, _bucket, _operation);
}
template <class T>
auto communicator::allreduce(const std::vector<T> &_value, MPI_Op _operation) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    allreduce(_value, _bucket, _operation);
    return _bucket;
}

template <class T>
auto communicator::allreduce(const T _value, MPI_Op _operation) -> T
{
    auto _bucket = T{};
    allreduce(_value, _bucket, _operation);
    return _bucket;
}

template <class T, class Op>
auto communicator::allreduce(const T &_value, T &_bucket, const op_proxy<T, Op> *_operation) -> void
{
    return allreduce(_value, _bucket, _operation->op());
}
template <class T, class Op>
auto communicator::allreduce(const std::vector<T> &_value, const op_proxy<T, Op> *_operation) -> std::vector<T>
{
    return allreduce(_value, _operation->op());
}
template <class Op>
auto communicator::allreduce(const std::string &_value, const op_proxy<std::string, Op> *_operation) -> std::string
{
    return allreduce(_value, _operation->op());
}
template <class T, class Op>
auto communicator::allreduce(const T _value, const op_proxy<T, Op> *_operation) -> T
{
    return allreduce(_value, _operation->op());
}
template <class Op>
auto communicator::allreduce(const char *_value, const op_proxy<std::string, Op> *_operation) -> std::string
{
    return allreduce(_value, _operation->op());
}
#pragma endregion
#pragma region MPI sender
template <class T>
auto sender::send(const T _value) -> void
{
    send_impl(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::ssend(const T _value) -> void
{
    ssend_impl(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::rsend(const T _value) -> void
{
    rsend_impl(_dest, _tag, _comm, _value);
}

template <class T>
auto sender::gather(const T &_value, T &_bucket) -> void
{
    gather_impl(_dest, _comm, _value, _bucket);
}
template <class T>
auto sender::gather(const T _value, std::vector<T> &_bucket) -> void
{
    gather({_value}, _bucket);
}
template <class T>
auto sender::gather(const std::vector<T> &_value) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    gather(_value, _bucket);
    return _bucket;
}
template <class T>
auto sender::gather(const T _value) -> std::vector<T>
{
    return gather(std::vector<T>{_value});
}

template <class T>
auto sender::reduce(const T &_value, T &_bucket, MPI_Op _operation) -> void
{
    reduce_impl(_dest, _comm, _value, _bucket, _operation);
}
template <class T>
auto sender::reduce(const std::vector<T> &_value, MPI_Op _operation) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    reduce(_value, _bucket, _operation);
    return _bucket;
}
template <class T>
auto sender::reduce(const T _value, MPI_Op _operation) -> T
{
    auto _bucket = T{};
    reduce(_value, _bucket, _operation);
    return _bucket;
}

template <class T, class Op>
auto sender::reduce(const T &_value, T &_bucket, const op_proxy<T, Op> *_operation) -> void
{
    return reduce(_value, _bucket, _operation->op());
}
template <class T, class Op>
auto sender::reduce(const std::vector<T> &_value, const op_proxy<T, Op> *_operation) -> std::vector<T>
{
    return reduce(_value, _operation->op());
}
template <class Op>
auto sender::reduce(const std::string &_value, const op_proxy<std::string, Op> *_operation) -> std::string
{
    return reduce(_value, _operation->op());
}
template <class T, class Op>
auto sender::reduce(const T _value, const op_proxy<T, Op> *_operation) -> T
{
    return reduce(_value, _operation->op());
}
template <class Op>
auto sender::reduce(const char *_value, const op_proxy<std::string, Op> *_operation) -> std::string
{
    return reduce(_value, _operation->op());
}
#pragma endregion
#pragma region MPI receiver
template <class T>
auto receiver::recv() -> T
{
    auto _value = T{};
    recv(_value);
    return _value;
}
template <class T>
auto receiver::recv(T &_value) -> void
{
    recv_impl(_source, _tag, _comm, &_status, _value);
}

template <class T>
auto receiver::bcast(T &_value) -> void
{
    bcast_impl(_source, _comm, _value);
}

template <class T>
auto receiver::scatter(T &_value, const size_t _chunk_size) -> T
{
    return scatter_impl(_source, _comm, _value, _chunk_size);
}
#pragma endregion
} // namespace mpi