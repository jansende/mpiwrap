#pragma once
#include <algorithm>
#include <cassert>
#include <cstring>
#include <mpiwrap/impl/lambda_hack.h>

namespace mpi
{
#pragma region allgather
//declarations
auto allgather_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void;
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
#pragma region allreduce
//declarations
auto allreduce_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket, op *_operation) -> void;
//templates
template <class T>
auto allreduce_impl(MPI_Comm _comm, const T &_value, T &_bucket, op *_operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce(&_value, &_bucket, 1, type_wrapper<T>{}, _operation->get(), _comm);
}
template <class T>
auto allreduce_impl(MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void
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
    MPI_Allreduce(_value.data(), _bucket.data(), _size, type_wrapper<T>{}, _operation->get(), _comm);
}
#pragma endregion
#pragma region alltoall
//declarations
auto alltoall_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void;
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
#pragma region broadcast
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
#pragma region gather
//declarations
auto gather_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket) -> void;
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
#pragma region receive
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
    //we need to find the proper size of the incoming data
    MPI_Probe(_source, _tag, _comm, _status);
    auto _size = int{};
    MPI_Get_count(_status, type_wrapper<T>{}, &_size);
    //we need to allocate some memory for it
    _value.resize(_size);
    //we need to receive it
    MPI_Recv(_value.data(), _size, type_wrapper<T>{}, _source, _tag, _comm, _status);
}
#pragma endregion
#pragma region reduce
//declarations
auto reduce_impl(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket, op *_operation) -> void;
//templates
template <class T>
auto reduce_impl(int _dest, MPI_Comm _comm, const T &_value, T &_bucket, op *_operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce(&_value, &_bucket, 1, type_wrapper<T>{}, _operation->get(), _dest, _comm);
}
template <class T>
auto reduce_impl(int _dest, MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void
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
    MPI_Reduce(_value.data(), _bucket.data(), _size, type_wrapper<T>{}, _operation->get(), _dest, _comm);
}
#pragma endregion
#pragma region local reduce
//declarations
auto reduce(const std::string &_value, std::string &_bucket, op *_operation) -> void;
auto reduce(const char *_value, std::string &_bucket, op *_operation) -> void;

auto reduce(const std::string &_value, op *_operation) -> std::string;
auto reduce(const char *_value, op *_operation) -> std::string;
//templates
template <class T>
auto reduce(const T &_value, T &_bucket, op *_operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce_local(&_value, &_bucket, 1, type_wrapper<T>{}, _operation->get());
}
template <class T>
auto reduce(const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //resize bucket to take all elements
    if (_value.size() != _bucket.size())
        _bucket.resize(_value.size());
    //reduce the data
    MPI_Reduce_local(_value.data(), _bucket.data(), _value.size(), type_wrapper<T>{}, _operation->get());
}
template <class T>
auto reduce(const std::vector<T> &_value, op *_operation) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    reduce(_value, _bucket, _operation);
    return _bucket;
}
template <class T>
auto reduce(const T _value, op *_operation) -> T
{
    auto _bucket = T{};
    reduce(_value, _bucket, _operation);
    return _bucket;
}

template <class T, class Op>
auto reduce(const T &_value, T &_bucket, Op _operation) -> void
{
    return reduce(_value, _bucket, make_op<T>(_operation).get());
}
template <class Op>
auto reduce(const std::string &_value, std::string &_bucket, Op _operation) -> void
{
    return reduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class Op>
auto reduce(const char *_value, std::string &_bucket, Op _operation) -> void
{
    return reduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class T, class Op>
auto reduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> void
{
    return reduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class T, class Op>
auto reduce(const std::vector<T> &_value, Op _operation) -> std::vector<T>
{
    return reduce(_value, make_op<T>(_operation).get());
}
template <class Op>
auto reduce(const std::string &_value, const Op _operation) -> std::string
{
    return reduce(_value, make_op<std::string>(_operation).get());
}
template <class T, class Op>
auto reduce(const T _value, Op _operation) -> T
{
    return reduce(_value, make_op<std::string>(_operation).get());
}
template <class Op>
auto reduce(const char *_value, Op _operation) -> std::string
{
    return reduce(_value, make_op<std::string>(_operation).get());
}
#pragma endregion
#pragma region scatter
//declarations
auto scatter_impl(int _source, MPI_Comm _comm, const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void;
//templates
template <class T>
auto scatter_impl(int _source, MPI_Comm _comm, const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //resize bucket to take all elements
    if (_chunk_size != _bucket.size())
        _bucket.resize(_chunk_size);
    //scatter the data
    MPI_Scatter(_value.data(), _chunk_size, type_wrapper<T>{}, _bucket.data(), _chunk_size, type_wrapper<T>{}, _source, _comm);
}
#pragma endregion
#pragma region send
//declarations
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
//templates
template <class T>
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const T &_value) -> void
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
#pragma region synchronized send
//declarations
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
//templates
template <class T>
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const T &_value) -> void
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
#pragma region ready mode send
//declarations
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void;
//templates
template <class T>
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const T &_value) -> void
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

#pragma region nonblocking allgather
//declarations
auto iallgather_impl(MPI_Comm _comm, MPI_Request *_request, const std::string &_value, std::unique_ptr<char[]> &_bucket) -> void;
//templates
template <class T>
auto iallgather_impl(MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value, std::vector<T> &_bucket) -> void
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
    MPI_Iallgather(_value.data(), _chunk_size, type_wrapper<T>{}, _bucket.data(), _chunk_size, type_wrapper<T>{}, _comm, _request);
}
#pragma endregion
#pragma region nonblocking allreduce
//declarations
auto iallreduce_impl(MPI_Comm _comm, MPI_Request *_request, const std::string &_value, std::unique_ptr<char[]> &_bucket, op *_operation) -> void;
//templates
template <class T>
auto iallreduce_impl(MPI_Comm _comm, MPI_Request *_request, const T &_value, T &_bucket, op *_operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Reduce(&_value, &_bucket, 1, type_wrapper<T>{}, _operation->get(), _comm, _request);
}
template <class T>
auto iallreduce_impl(MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void
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
    MPI_Iallreduce(_value.data(), _bucket.data(), _size, type_wrapper<T>{}, _operation->get(), _comm, _request);
}
#pragma endregion
#pragma region nonblocking alltoall
//declarations
auto ialltoall_impl(MPI_Comm _comm, MPI_Request *_request, const std::string &_value, std::unique_ptr<char[]> &_bucket, const size_t _chunk_size) -> void;
//templates
template <class T>
auto ialltoall_impl(MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void
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
    MPI_Ialltoall(_value.data(), _chunk_size, type_wrapper<T>{}, _bucket.data(), _chunk_size, type_wrapper<T>{}, _comm, _request);
}
#pragma endregion
#pragma region nonblocking broadcast
//declarations
auto ibcast_impl(int _source, MPI_Comm _comm, MPI_Request *_request, std::string &_value) -> void;
//templates
template <class T>
auto ibcast_impl(int _source, MPI_Comm _comm, MPI_Request *_request, T &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Ibcast(&_value, 1, type_wrapper<T>{}, _source, _comm, _request);
}
template <class T>
auto ibcast_impl(int _source, MPI_Comm _comm, MPI_Request *_request, std::vector<T> &_value) -> void
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
    MPI_Ibcast(_value.data(), _size, type_wrapper<T>{}, _source, _comm, _request);
}
#pragma endregion
#pragma region nonblocking gather
//declarations
auto igather_impl(int _dest, MPI_Comm _comm, MPI_Request *_request, const std::string &_value, std::unique_ptr<char[]> &_bucket) -> void;
//templates
template <class T>
auto igather_impl(int _dest, MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value, std::vector<T> &_bucket) -> void
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
    MPI_Igather(_value.data(), _chunk_size, type_wrapper<T>{}, _bucket.data(), _chunk_size, type_wrapper<T>{}, _dest, _comm, _request);
}
#pragma endregion
#pragma region nonblocking receive
//declarations
auto irecv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, MPI_Request *_request, std::unique_ptr<char[]> &_value) -> void;
//templates
template <class T>
auto irecv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, MPI_Request *_request, T &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Irecv(&_value, 1, type_wrapper<T>{}, _source, _tag, _comm, _request);
}
template <class T>
auto irecv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, MPI_Request *_request, std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //we need to find the proper size of the incoming data
    MPI_Probe(_source, _tag, _comm, _status);
    auto _size = int{};
    MPI_Get_count(_status, type_wrapper<T>{}, &_size);
    //we need to allocate some memory for it
    _value.resize(_size);
    //we need to receive it
    MPI_Irecv(_value.data(), _size, type_wrapper<T>{}, _source, _tag, _comm, _request);
}
#pragma endregion
#pragma region nonblocking reduce
//declarations
auto ireduce_impl(int _dest, MPI_Comm _comm, MPI_Request *_request, const std::string &_value, std::unique_ptr<char[]> &_bucket, op *_operation) -> void;
//templates
template <class T>
auto ireduce_impl(int _dest, MPI_Comm _comm, MPI_Request *_request, const T &_value, T &_bucket, op *_operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //reduce the data
    MPI_Ireduce(&_value, &_bucket, 1, type_wrapper<T>{}, _operation->get(), _dest, _comm, _request);
}
template <class T>
auto ireduce_impl(int _dest, MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void
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
    MPI_Ireduce(_value.data(), _bucket.data(), _size, type_wrapper<T>{}, _operation->get(), _dest, _comm, _request);
}
#pragma endregion
#pragma region nonblocking scatter
//declarations
auto iscatter_impl(int _source, MPI_Comm _comm, MPI_Request *_request, const std::string &_value, std::unique_ptr<char[]> &_bucket, const size_t _chunk_size) -> void;
//templates
template <class T>
auto iscatter_impl(int _source, MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //resize bucket to take all elements
    if (_chunk_size != _bucket.size())
        _bucket.resize(_chunk_size);
    //scatter the data
    MPI_Iscatter(_value.data(), _chunk_size, type_wrapper<T>{}, _bucket.data(), _chunk_size, type_wrapper<T>{}, _source, _comm, _request);
}
#pragma endregion
#pragma region nonblocking send
//declarations
auto isend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const std::string &_value) -> void;
//templates
template <class T>
auto isend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const T &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Isend(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm, _request);
}
template <class T>
auto isend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Isend(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm, _request);
}
#pragma endregion
#pragma region nonblocking synchronized send
//declarations
auto issend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const std::string &_value) -> void;
//templates
template <class T>
auto issend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const T &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Issend(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm, _request);
}
template <class T>
auto issend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Issend(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm, _request);
}
#pragma endregion
#pragma region nonblocking ready mode send
//declarations
auto irsend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const std::string &_value) -> void;
//templates
template <class T>
auto irsend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const T &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Irsend(&_value, 1, type_wrapper<T>{}, _dest, _tag, _comm, _request);
}
template <class T>
auto irsend_impl(int _dest, int _tag, MPI_Comm _comm, MPI_Request *_request, const std::vector<T> &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Irsend(_value.data(), _value.size(), type_wrapper<T>{}, _dest, _tag, _comm, _request);
}
#pragma endregion

#pragma region communicator
template <class T>
auto communicator::allgather(const T &_value, std::vector<T> &_bucket) -> void
{
    return allgather(std::vector<T>{_value}, _bucket);
}
template <class T>
auto communicator::allgather(const std::vector<T> &_value, std::vector<T> &_bucket) -> void
{
    return allgather_impl(_comm, _value, _bucket);
}
template <class T>
auto communicator::allgather(const T &_value) -> std::vector<T>
{
    return allgather(std::vector<T>{_value});
}
template <class T>
auto communicator::allgather(const std::vector<T> &_value) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    allgather(_value, _bucket);
    return _bucket;
}
template <class T>
auto communicator::iallgather(const T &_value, std::vector<T> &_bucket) -> std::unique_ptr<iallgather_request<std::vector<T>>>
{
    return iallgather(std::vector<T>{_value}, _bucket);
}
template <class T>
auto communicator::iallgather(const std::vector<T> &_value, std::vector<T> &_bucket) -> std::unique_ptr<iallgather_request<std::vector<T>>>
{
    return std::make_unique<iallgather_request<std::vector<T>>>(_comm, _value, _bucket);
}
template <class T>
auto communicator::iallgather(const T &_value) -> std::unique_ptr<iallgather_reply<std::vector<T>>>
{
    return iallgather(std::vector<T>{_value});
}
template <class T>
auto communicator::iallgather(const std::vector<T> &_value) -> std::unique_ptr<iallgather_reply<std::vector<T>>>
{
    return std::make_unique<iallgather_reply<std::vector<T>>>(_comm, _value);
}

template <class T>
auto communicator::alltoall(const T &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void
{
    return alltoall(std::vector<T>{_value}, _bucket, _chunk_size);
}
template <class T>
auto communicator::alltoall(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void
{
    return alltoall_impl(_comm, _value, _bucket, _chunk_size);
}
template <class T>
auto communicator::alltoall(const T &_value, const size_t _chunk_size) -> std::vector<T>
{
    return alltoall(std::vector<T>{_value}, _chunk_size);
}
template <class T>
auto communicator::alltoall(const std::vector<T> &_value, const size_t _chunk_size) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    alltoall(_value, _bucket, _chunk_size);
    return _bucket;
}
template <class T>
auto communicator::ialltoall(const T &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> std::unique_ptr<ialltoall_request<std::vector<T>>>
{
    return ialltoall(std::vector<T>{_value}, _bucket, _chunk_size);
}
template <class T>
auto communicator::ialltoall(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> std::unique_ptr<ialltoall_request<std::vector<T>>>
{
    return std::make_unique<ialltoall_request<std::vector<T>>>(_comm, _value, _bucket, _chunk_size);
}
template <class T>
auto communicator::ialltoall(const T &_value, const size_t _chunk_size) -> std::unique_ptr<ialltoall_reply<std::vector<T>>>
{
    return ialltoall(std::vector<T>{_value}, _chunk_size);
}
template <class T>
auto communicator::ialltoall(const std::vector<T> &_value, const size_t _chunk_size) -> std::unique_ptr<ialltoall_reply<std::vector<T>>>
{
    return std::make_unique<ialltoall_reply<std::vector<T>>>(_comm, _value, _chunk_size);
}

template <class T>
auto communicator::allreduce(const T &_value, T &_bucket, op *_operation) -> void
{
    return allreduce_impl(_comm, _value, _bucket, _operation);
}
template <class T>
auto communicator::allreduce(const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void
{
    return allreduce_impl(_comm, _value, _bucket, _operation);
}
template <class T>
auto communicator::allreduce(const T &_value, op *_operation) -> T
{
    auto _bucket = T{};
    allreduce(_value, _bucket, _operation);
    return _bucket;
}
template <class T>
auto communicator::allreduce(const std::vector<T> &_value, op *_operation) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    allreduce(_value, _bucket, _operation);
    return _bucket;
}

template <class T, class Op>
auto communicator::allreduce(const T &_value, T &_bucket, Op _operation) -> void
{
    return allreduce(_value, _bucket, make_op<T>(_operation).get());
}
template <class T, class Op>
auto communicator::allreduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> void
{
    return allreduce(_value, _bucket, make_op<T>(_operation).get());
}
template <class Op>
auto communicator::allreduce(const char _value, std::string &_bucket, Op _operation) -> void
{
    return allreduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class Op>
auto communicator::allreduce(const char *_value, std::string &_bucket, Op _operation) -> void
{
    return allreduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class Op>
auto communicator::allreduce(const std::string &_value, std::string &_bucket, Op _operation) -> void
{
    return allreduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class T, class Op>
auto communicator::allreduce(const T &_value, Op _operation) -> T
{
    return allreduce(_value, make_op<T>(_operation).get());
}
template <class T, class Op>
auto communicator::allreduce(const std::vector<T> &_value, Op _operation) -> std::vector<T>
{
    return allreduce(_value, make_op<T>(_operation).get());
}
template <class Op>
auto communicator::allreduce(const char _value, Op _operation) -> std::string
{
    return allreduce(_value, make_op<std::string>(_operation).get());
}
template <class Op>
auto communicator::allreduce(const char *_value, Op _operation) -> std::string
{
    return allreduce(_value, make_op<std::string>(_operation).get());
}
template <class Op>
auto communicator::allreduce(const std::string &_value, Op _operation) -> std::string
{
    return allreduce(_value, make_op<std::string>(_operation).get());
}

template <class T>
auto communicator::iallreduce(const T &_value, T &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_request<T>>
{
    return std::make_unique<iallreduce_request<T>>(_comm, _value, _bucket, _operation);
}
template <class T>
auto communicator::iallreduce(const std::vector<T> &_value, std::vector<T> &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_request<std::vector<T>>>
{
    return std::make_unique<iallreduce_request<std::vector<T>>>(_comm, _value, _bucket, _operation);
}
template <class T>
auto communicator::iallreduce(const T &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_reply<T>>
{
    return std::make_unique<iallreduce_reply<T>>(_comm, _value, _operation);
}
template <class T>
auto communicator::iallreduce(const std::vector<T> &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_reply<std::vector<T>>>
{
    return std::make_unique<iallreduce_reply<std::vector<T>>>(_comm, _value, _operation);
}

template <class T, class Op>
auto communicator::iallreduce(const T &_value, T &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<T>>
{
    return iallreduce(_value, _bucket, make_op<T>(_operation));
}
template <class T, class Op>
auto communicator::iallreduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<std::vector<T>>>
{
    return iallreduce(_value, _bucket, make_op<T>(_operation));
}
template <class Op>
auto communicator::iallreduce(const char _value, std::string &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<std::string>>
{
    return iallreduce(_value, _bucket, make_op<std::string>(_operation));
}
template <class Op>
auto communicator::iallreduce(const char *_value, std::string &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<std::string>>
{
    return iallreduce(_value, _bucket, make_op<std::string>(_operation));
}
template <class Op>
auto communicator::iallreduce(const std::string &_value, std::string &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<std::string>>
{
    return iallreduce(_value, _bucket, make_op<std::string>(_operation));
}
template <class T, class Op>
auto communicator::iallreduce(const T &_value, Op _operation) -> std::unique_ptr<iallreduce_reply<T>>
{
    return iallreduce(_value, make_op<T>(_operation));
}
template <class T, class Op>
auto communicator::iallreduce(const std::vector<T> &_value, Op _operation) -> std::unique_ptr<iallreduce_reply<std::vector<T>>>
{
    return iallreduce(_value, make_op<T>(_operation));
}
template <class Op>
auto communicator::iallreduce(const char _value, Op _operation) -> std::unique_ptr<iallreduce_reply<std::string>>
{
    return iallreduce(_value, make_op<std::string>(_operation));
}
template <class Op>
auto communicator::iallreduce(const char *_value, Op _operation) -> std::unique_ptr<iallreduce_reply<std::string>>
{
    return iallreduce(_value, make_op<std::string>(_operation));
}
template <class Op>
auto communicator::iallreduce(const std::string &_value, Op _operation) -> std::unique_ptr<iallreduce_reply<std::string>>
{
    return iallreduce(_value, make_op<std::string>(_operation));
}
#pragma endregion
#pragma region operation wrapper
template <class T, class Op>
auto op_proxy<T, Op>::wrapper(void *void_a, void *void_b, int *len, MPI_Datatype *) -> void
{
    auto a = static_cast<T *>(void_a);
    auto b = static_cast<T *>(void_b);
    auto op = impl::lambda_hack_impl<Op>{}.get();
    std::transform(a, a + *len, b, b, op);
}
template <class T, class Op>
op_proxy<T, Op>::op_proxy(const bool _commute) : op(_commute)
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
auto make_op(Op _func, const bool _commute) -> std::shared_ptr<op>
{
    return std::make_shared<op_proxy<T, Op>>(_commute);
}
#pragma endregion
#pragma region request implementations
template <class T>
isend_request<T>::isend_request(int _dest, int _tag, MPI_Comm _comm, const T &_value) : request(_comm), _dest(_dest), _tag(_tag), _value(_value)
{
    isend_impl(this->_dest, this->_tag, this->_comm, &this->_request, this->_value);
}
template <class T>
issend_request<T>::issend_request(int _dest, int _tag, MPI_Comm _comm, const T &_value) : request(_comm), _dest(_dest), _tag(_tag), _value(_value)
{
    issend_impl(this->_dest, this->_tag, this->_comm, &this->_request, this->_value);
}
template <class T>
irsend_request<T>::irsend_request(int _dest, int _tag, MPI_Comm _comm, const T &_value) : request(_comm), _dest(_dest), _tag(_tag), _value(_value)
{
    irsend_impl(this->_dest, this->_tag, this->_comm, &this->_request, _value);
}
template <class T>
irecv_request<T>::irecv_request(int _source, int _tag, MPI_Comm _comm, T &_value) : request(_comm), _source(_source), _tag(_tag)
{
    irecv_impl(this->_source, this->_tag, this->_comm, &this->_status, &this->_request, _value);
}
template <class T>
irecv_reply<T>::irecv_reply(int _source, int _tag, MPI_Comm _comm) : request(_comm), _source(_source), _tag(_tag), _bucket(T{})
{
    irecv_impl(this->_source, this->_tag, this->_comm, &this->_status, &this->_request, this->_bucket);
}
template <class T>
auto irecv_reply<T>::get() -> T
{
    this->wait();
    return _bucket;
}
template <class T>
ibcast_request<T>::ibcast_request(int _source, MPI_Comm _comm, T &_value) : request(_comm), _source(_source)
{
    ibcast_impl(this->_source, this->_comm, &this->_request, _value);
}
template <class T>
ibcast_reply<T>::ibcast_reply(int _source, MPI_Comm _comm, const T &_value) : request(_comm), _source(_source), _bucket(_value)
{
    ibcast_impl(this->_source, this->_comm, &this->_request, this->_bucket);
}
template <class T>
auto ibcast_reply<T>::get() -> T
{
    this->wait();
    return _bucket;
}

template <class T>
iscatter_request<T>::iscatter_request(int _source, MPI_Comm _comm, const T &_value, T &_bucket, const size_t _chunk_size) : request(_comm), _source(_source), _chunk_size(_chunk_size), _value(_value), _bucket(_bucket)
{
    iscatter_impl(this->_source, this->_comm, &this->_request, this->_value, this->_bucket, this->_chunk_size);
}
template <class T>
iscatter_reply<T>::iscatter_reply(int _source, MPI_Comm _comm, const T &_value, const size_t _chunk_size) : request(_comm), _source(_source), _chunk_size(_chunk_size), _value(_value), _bucket(T{})
{
    iscatter_impl(this->_source, this->_comm, &this->_request, this->_value, this->_bucket, this->_chunk_size);
}
template <class T>
auto iscatter_reply<T>::get() -> T
{
    this->wait();
    return _bucket;
}

template <class T>
igather_request<T>::igather_request(int _dest, MPI_Comm _comm, const T &_value, T &_bucket) : request(_comm), _dest(_dest), _value(_value), _bucket(_bucket)
{
    igather_impl(this->_dest, this->_comm, &this->_request, this->_value, this->_bucket);
}
template <class T>
igather_reply<T>::igather_reply(int _dest, MPI_Comm _comm, const T &_value) : request(_comm), _dest(_dest), _value(_value), _bucket(T{})
{
    igather_impl(this->_dest, this->_comm, &this->_request, this->_value, this->_bucket);
}
template <class T>
auto igather_reply<T>::get() -> T
{
    this->wait();
    return _bucket;
}

template <class T>
iallgather_request<T>::iallgather_request(MPI_Comm _comm, const T &_value, T &_bucket) : request(_comm), _value(_value), _bucket(_bucket)
{
    iallgather_impl(this->_comm, &this->_request, this->_value, this->_bucket);
}
template <class T>
iallgather_reply<T>::iallgather_reply(MPI_Comm _comm, const T &_value) : request(_comm), _value(_value), _bucket(T{})
{
    iallgather_impl(this->_comm, &this->_request, this->_value, this->_bucket);
}
template <class T>
auto iallgather_reply<T>::get() -> T
{
    this->wait();
    return _bucket;
}

template <class T>
ialltoall_request<T>::ialltoall_request(MPI_Comm _comm, const T &_value, T &_bucket, const size_t _chunk_size) : request(_comm), _chunk_size(_chunk_size), _value(_value), _bucket(_bucket)
{
    ialltoall_impl(this->_comm, &this->_request, this->_value, this->_bucket, this->_chunk_size);
}
template <class T>
ialltoall_reply<T>::ialltoall_reply(MPI_Comm _comm, const T &_value, const size_t _chunk_size) : request(_comm), _chunk_size(_chunk_size), _value(_value), _bucket(T{})
{
    ialltoall_impl(this->_comm, &this->_request, this->_value, this->_bucket, this->_chunk_size);
}
template <class T>
auto ialltoall_reply<T>::get() -> T
{
    this->wait();
    return _bucket;
}

template <class T>
ireduce_request<T>::ireduce_request(int _dest, MPI_Comm _comm, const T &_value, T &_bucket, std::shared_ptr<op> _operation) : request(_comm), _operation(_operation), _dest(_dest), _value(_value), _bucket(_bucket)
{
    ireduce_impl(this->_dest, this->_comm, &this->_request, this->_value, this->_bucket, this->_operation.get());
}
template <class T>
ireduce_reply<T>::ireduce_reply(int _dest, MPI_Comm _comm, const T &_value, std::shared_ptr<op> _operation) : request(_comm), _operation(_operation), _dest(_dest), _value(_value), _bucket(T{})
{
    ireduce_impl(this->_dest, this->_comm, &this->_request, this->_value, this->_bucket, this->_operation.get());
}
template <class T>
auto ireduce_reply<T>::get() -> T
{
    this->wait();
    return _bucket;
}

template <class T>
iallreduce_request<T>::iallreduce_request(MPI_Comm _comm, const T &_value, T &_bucket, std::shared_ptr<op> _operation) : request(_comm), _operation(_operation), _value(_value), _bucket(_bucket)
{
    iallreduce_impl(this->_comm, &this->_request, this->_value, this->_bucket, this->_operation.get());
}
template <class T>
iallreduce_reply<T>::iallreduce_reply(MPI_Comm _comm, const T &_value, std::shared_ptr<op> _operation) : request(_comm), _operation(_operation), _value(_value), _bucket(T{})
{
    iallreduce_impl(this->_comm, &this->_request, this->_value, this->_bucket, this->_operation.get());
}
template <class T>
auto iallreduce_reply<T>::get() -> T
{
    this->wait();
    return _bucket;
}
#pragma endregion
#pragma region test
template <class... T>
auto testall(std::unique_ptr<T> &... _values) -> bool
{
    return testall(std::vector<request *>{_values.get()...});
}
template <class... T>
auto testall(T *... _values) -> bool
{
    return testall(std::vector<request *>{_values...});
}
template <class... T>
auto testany(std::unique_ptr<T> &... _values) -> std::vector<size_t>
{
    return testany(std::vector<request *>{_values.get()...});
}
template <class... T>
auto testany(T *... _values) -> std::vector<size_t>
{
    return testany(std::vector<request *>{_values...});
}
template <class... T>
auto testsome(std::unique_ptr<T> &... _values) -> std::vector<size_t>
{
    return testsome(std::vector<request *>{_values.get()...});
}
template <class... T>
auto testsome(T *... _values) -> std::vector<size_t>
{
    return testsome(std::vector<request *>{_values...});
}
#pragma endregion
#pragma region wait
template <class... T>
auto waitall(std::unique_ptr<T> &... _values) -> void
{
    waitall(std::vector<request *>{_values.get()...});
}
template <class... T>
auto waitall(T *... _values) -> void
{
    waitall(std::vector<request *>{_values...});
}
template <class... T>
auto waitany(std::unique_ptr<T> &... _values) -> std::vector<size_t>
{
    return waitany(std::vector<request *>{_values.get()...});
}
template <class... T>
auto waitany(T *... _values) -> std::vector<size_t>
{
    return waitany(std::vector<request *>{_values...});
}
template <class... T>
auto waitsome(std::unique_ptr<T> &... _values) -> std::vector<size_t>
{
    return waitsome(std::vector<request *>{_values.get()...});
}
template <class... T>
auto waitsome(T *... _values) -> std::vector<size_t>
{
    return waitsome(std::vector<request *>{_values...});
}
#pragma endregion
#pragma region receiver
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
auto receiver::irecv() -> std::unique_ptr<irecv_reply<T>>
{
    return std::make_unique<irecv_reply<T>>(_source, _tag, _comm);
}
template <class T>
auto receiver::irecv(T &_value) -> std::unique_ptr<irecv_request<T>>
{
    return std::make_unique<irecv_request<T>>(_source, _tag, _comm, _value);
}

template <class T>
auto receiver::bcast(T &_value) -> void
{
    bcast_impl(_source, _comm, _value);
}
template <class R, class T>
auto receiver::bcast(const T &_value) -> std::enable_if_t<std::is_same<R, T>::value, T>
{
    auto _bucket = _value;
    bcast(_bucket);
    return _bucket;
}
template <class R, class T>
auto receiver::bcast(const std::vector<T> &_value) -> std::enable_if_t<std::is_same<R, std::vector<T>>::value, std::vector<T>>
{
    auto _bucket = _value;
    bcast(_bucket);
    return _bucket;
}
template <class R>
auto receiver::bcast(const char _value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::string>
{
    return bcast<R>(std::string{_value});
}
template <class R>
auto receiver::bcast(const char *_value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::string>
{
    return bcast<R>(std::string{_value});
}
template <class R>
auto receiver::bcast(const std::string &_value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::string>
{
    auto _bucket = _value;
    bcast(_bucket);
    return _bucket;
}
template <class T>
auto receiver::ibcast(T &_value) -> std::unique_ptr<ibcast_request<T>>
{
    return std::make_unique<ibcast_request<T>>(_source, _comm, _value);
}
template <class R, class T>
auto receiver::ibcast(const T &_value) -> std::enable_if_t<std::is_same<R, T>::value, std::unique_ptr<ibcast_reply<T>>>
{
    return std::make_unique<ibcast_reply<T>>(_source, _comm, _value);
}
template <class R, class T>
auto receiver::ibcast(const std::vector<T> &_value) -> std::enable_if_t<std::is_same<R, std::vector<T>>::value, std::unique_ptr<ibcast_reply<std::vector<T>>>>
{
    return std::make_unique<ibcast_reply<std::vector<T>>>(_source, _comm, _value);
}
template <class R>
auto receiver::ibcast(const char _value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::unique_ptr<ibcast_reply<std::string>>>
{
    return std::make_unique<ibcast_reply<std::string>>(_source, _comm, std::string{_value});
}
template <class R>
auto receiver::ibcast(const char *_value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::unique_ptr<ibcast_reply<std::string>>>
{
    return std::make_unique<ibcast_reply<std::string>>(_source, _comm, std::string{_value});
}
template <class R>
auto receiver::ibcast(const std::string &_value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::unique_ptr<ibcast_reply<std::string>>>
{
    return std::make_unique<ibcast_reply<std::string>>(_source, _comm, _value);
}

template <class T>
auto receiver::scatter(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void
{
    return scatter_impl(_source, _comm, _value, _bucket, _chunk_size);
}
template <class T>
auto receiver::scatter(const std::vector<T> &_value, const size_t _chunk_size) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    scatter(_value, _bucket, _chunk_size);
    return _bucket;
}
template <class T>
auto receiver::iscatter(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> std::unique_ptr<iscatter_request<std::vector<T>>>
{
    return std::make_unique<iscatter_request<std::vector<T>>>(_source, _comm, _value, _bucket, _chunk_size);
}
template <class T>
auto receiver::iscatter(const std::vector<T> &_value, const size_t _chunk_size) -> std::unique_ptr<iscatter_reply<std::vector<T>>>
{
    return std::make_unique<iscatter_reply<std::vector<T>>>(_source, _comm, _value, _chunk_size);
}
#pragma endregion
#pragma region sender
template <class T>
auto sender::send(const T &_value) -> void
{
    return send_impl(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::send(const std::vector<T> &_value) -> void
{
    return send_impl(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::ssend(const T &_value) -> void
{
    return ssend_impl(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::ssend(const std::vector<T> &_value) -> void
{
    return ssend_impl(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::rsend(const T &_value) -> void
{
    return rsend_impl(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::rsend(const std::vector<T> &_value) -> void
{
    return rsend_impl(_dest, _tag, _comm, _value);
}

template <class T>
auto sender::isend(const T &_value) -> std::unique_ptr<isend_request<T>>
{
    return std::make_unique<isend_request<T>>(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::isend(const std::vector<T> &_value) -> std::unique_ptr<isend_request<std::vector<T>>>
{
    return std::make_unique<isend_request<std::vector<T>>>(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::issend(const T &_value) -> std::unique_ptr<issend_request<T>>
{
    return std::make_unique<issend_request<T>>(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::issend(const std::vector<T> &_value) -> std::unique_ptr<issend_request<std::vector<T>>>
{
    return std::make_unique<issend_request<std::vector<T>>>(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::irsend(const T &_value) -> std::unique_ptr<irsend_request<T>>
{
    return std::make_unique<irsend_request<T>>(_dest, _tag, _comm, _value);
}
template <class T>
auto sender::irsend(const std::vector<T> &_value) -> std::unique_ptr<irsend_request<std::vector<T>>>
{
    return std::make_unique<irsend_request<std::vector<T>>>(_dest, _tag, _comm, _value);
}

template <class T>
auto sender::gather(const T &_value, std::vector<T> &_bucket) -> void
{
    return gather(std::vector<T>{_value}, _bucket);
}
template <class T>
auto sender::gather(const std::vector<T> &_value, std::vector<T> &_bucket) -> void
{
    return gather_impl(_dest, _comm, _value, _bucket);
}
template <class T>
auto sender::gather(const T &_value) -> std::vector<T>
{
    return gather(std::vector<T>{_value});
}
template <class T>
auto sender::gather(const std::vector<T> &_value) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    gather(_value, _bucket);
    return _bucket;
}
template <class T>
auto sender::igather(const T &_value, std::vector<T> &_bucket) -> std::unique_ptr<igather_request<std::vector<T>>>
{
    return igather(std::vector<T>{_value}, _bucket);
}
template <class T>
auto sender::igather(const std::vector<T> &_value, std::vector<T> &_bucket) -> std::unique_ptr<igather_request<std::vector<T>>>
{
    return std::make_unique<igather_request<std::vector<T>>>(_dest, _comm, _value, _bucket);
}
template <class T>
auto sender::igather(const T &_value) -> std::unique_ptr<igather_reply<std::vector<T>>>
{
    return igather(std::vector<T>{_value});
}
template <class T>
auto sender::igather(const std::vector<T> &_value) -> std::unique_ptr<igather_reply<std::vector<T>>>
{
    return std::make_unique<igather_reply<std::vector<T>>>(_dest, _comm, _value);
}

template <class T>
auto sender::reduce(const T &_value, T &_bucket, op *_operation) -> void
{
    return reduce_impl(_dest, _comm, _value, _bucket, _operation);
}
template <class T>
auto sender::reduce(const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void
{
    return reduce_impl(_dest, _comm, _value, _bucket, _operation);
}
template <class T>
auto sender::reduce(const T &_value, op *_operation) -> T
{
    auto _bucket = T{};
    reduce(_value, _bucket, _operation);
    return _bucket;
}
template <class T>
auto sender::reduce(const std::vector<T> &_value, op *_operation) -> std::vector<T>
{
    auto _bucket = std::vector<T>{};
    reduce(_value, _bucket, _operation);
    return _bucket;
}

template <class T, class Op>
auto sender::reduce(const T &_value, T &_bucket, Op _operation) -> void
{
    return reduce_impl(_dest, _comm, _value, _bucket, make_op<T>(_operation).get());
}
template <class T, class Op>
auto sender::reduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> void
{
    return reduce_impl(_dest, _comm, _value, _bucket, make_op<T>(_operation).get());
}
template <class Op>
auto sender::reduce(const char _value, std::string &_bucket, Op _operation) -> void
{
    return reduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class Op>
auto sender::reduce(const char *_value, std::string &_bucket, Op _operation) -> void
{
    return reduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class Op>
auto sender::reduce(const std::string &_value, std::string &_bucket, Op _operation) -> void
{
    return reduce(_value, _bucket, make_op<std::string>(_operation).get());
}
template <class T, class Op>
auto sender::reduce(const T &_value, Op _operation) -> T
{
    return reduce(_value, make_op<T>(_operation).get());
}
template <class T, class Op>
auto sender::reduce(const std::vector<T> &_value, Op _operation) -> std::vector<T>
{
    return reduce(_value, make_op<T>(_operation).get());
}
template <class Op>
auto sender::reduce(const char _value, Op _operation) -> std::string
{
    return reduce(_value, make_op<std::string>(_operation).get());
}
template <class Op>
auto sender::reduce(const char *_value, Op _operation) -> std::string
{
    return reduce(_value, make_op<std::string>(_operation).get());
}
template <class Op>
auto sender::reduce(const std::string &_value, Op _operation) -> std::string
{
    return reduce(_value, make_op<std::string>(_operation).get());
}

template <class T>
auto sender::ireduce(const T &_value, T &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_request<T>>
{
    return std::make_unique<ireduce_request<T>>(_dest, _comm, _value, _bucket, _operation);
}
template <class T>
auto sender::ireduce(const std::vector<T> &_value, std::vector<T> &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_request<std::vector<T>>>
{
    return std::make_unique<ireduce_request<std::vector<T>>>(_dest, _comm, _value, _bucket, _operation);
}
template <class T>
auto sender::ireduce(const T &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_reply<T>>
{
    return std::make_unique<ireduce_reply<T>>(_dest, _comm, _value, _operation);
}
template <class T>
auto sender::ireduce(const std::vector<T> &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_reply<std::vector<T>>>
{
    return std::make_unique<ireduce_reply<std::vector<T>>>(_dest, _comm, _value, _operation);
}

template <class T, class Op>
auto sender::ireduce(const T &_value, T &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<T>>
{
    return ireduce(_value, _bucket, make_op<T>(_operation));
}
template <class T, class Op>
auto sender::ireduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<std::vector<T>>>
{
    return ireduce(_value, _bucket, make_op<T>(_operation));
}
template <class Op>
auto sender::ireduce(const char _value, std::string &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<std::string>>
{
    return ireduce(_value, _bucket, make_op<std::string>(_operation));
}
template <class Op>
auto sender::ireduce(const char *_value, std::string &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<std::string>>
{
    return ireduce(_value, _bucket, make_op<std::string>(_operation));
}
template <class Op>
auto sender::ireduce(const std::string &_value, std::string &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<std::string>>
{
    return ireduce(_value, _bucket, make_op<std::string>(_operation));
}
template <class T, class Op>
auto sender::ireduce(const T &_value, Op _operation) -> std::unique_ptr<ireduce_reply<T>>
{
    return ireduce(_value, make_op<T>(_operation));
}
template <class T, class Op>
auto sender::ireduce(const std::vector<T> &_value, Op _operation) -> std::unique_ptr<ireduce_reply<std::vector<T>>>
{
    return ireduce(_value, make_op<T>(_operation));
}
template <class Op>
auto sender::ireduce(const char _value, Op _operation) -> std::unique_ptr<ireduce_reply<std::string>>
{
    return ireduce(_value, make_op<std::string>(_operation));
}
template <class Op>
auto sender::ireduce(const char *_value, Op _operation) -> std::unique_ptr<ireduce_reply<std::string>>
{
    return ireduce(_value, make_op<std::string>(_operation));
}
template <class Op>
auto sender::ireduce(const std::string &_value, Op _operation) -> std::unique_ptr<ireduce_reply<std::string>>
{
    return ireduce(_value, make_op<std::string>(_operation));
}
#pragma endregion
} // namespace mpi