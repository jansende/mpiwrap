#include <mpiwrap/mpi.h>

namespace mpi
{
#pragma region free functions
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
#pragma endregion
#pragma region init
mpi::mpi(int argc, char **argv)
{
    paranoidly_assert((!initialized()));
    paranoidly_assert((!finalized()));
    //add error checking
    MPI_Init(&argc, &argv);
}
mpi::~mpi()
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //add error checking
    MPI_Finalize();
}
#pragma endregion
#pragma region version information
version_info::version_info()
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));

    MPI_Get_version(&_version, &_subversion);
}
auto version_info::version() -> int
{
    return _version;
}
auto version_info::subversion() -> int
{
    return _subversion;
}
auto version() -> version_info
{
    return version_info{};
}
#pragma endregion

#pragma region allgather
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
#pragma endregion
#pragma region allreduce
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
#pragma endregion
#pragma region alltoall
auto alltoall_impl(MPI_Comm _comm, const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void
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
    //we need to allocate some memory for the result
    auto _c_str = std::make_unique<char[]>(_size + 1);
    //gather the data
    MPI_Alltoall(_value.c_str(), _chunk_size, MPI_CHAR, _c_str.get(), _chunk_size, MPI_CHAR, _comm);
    //we need to write the value back
    _bucket = std::string{_c_str.get()};
}
auto alltoall_impl(MPI_Comm _comm, const char *_value, std::string &_bucket, const size_t _chunk_size) -> void
{
    alltoall_impl(_comm, std::string{_value}, _bucket, _chunk_size);
}
#pragma endregion
#pragma region broadcast
auto bcast_impl(int _source, MPI_Comm _comm, std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //get current rank
    auto _rank = comm(_comm)->rank();
    //broadcast the size before the string
    auto _size = (_rank == _source) ? static_cast<int>(_value.size()) : int{};
    comm(_comm)->source(_source)->bcast(_size);
    //we need to allocate some memory for it
    auto _c_str = std::make_unique<char[]>(_size + 1);
    //we need to copy the data into the right place
    if (_rank == _source)
        std::strcpy(_c_str.get(), _value.c_str());
    //broadcast the data
    MPI_Bcast(_c_str.get(), _size, MPI_CHAR, _source, _comm);
    //at we need to write the value back
    _value = std::string{_c_str.get()};
}
#pragma endregion
#pragma region gather
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
#pragma endregion
#pragma region receive
auto recv_impl(int _source, int _tag, MPI_Comm _comm, MPI_Status *_status, std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //we need to finde the proper size of the incoming string
    MPI_Probe(_source, _tag, _comm, _status);
    auto _size = int{};
    MPI_Get_count(_status, MPI_CHAR, &_size);
    //we need to allocate some memory for it
    auto _c_str = std::make_unique<char[]>(_size + 1);
    //we need to receive it
    MPI_Recv(_c_str.get(), _size, MPI_CHAR, _source, _tag, _comm, _status);
    //we need to write the value back
    _value = std::string{_c_str.get()};
}
#pragma endregion
#pragma region reduce
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
#pragma endregion
#pragma region local reduce
auto reduce(const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    //we need to allocate some memory for the result
    auto _c_str = std::make_unique<char[]>(_value.size() + 1);
    //reduce the data
    MPI_Reduce_local(_value.c_str(), _c_str.get(), _value.size(), MPI_CHAR, _operation);
    //we need to write the value back
    _bucket = std::string{_c_str.get()};
}
auto reduce(const char *_value, std::string &_bucket, MPI_Op _operation) -> void
{
    reduce(std::string{_value}, _bucket, _operation);
}

auto reduce(const std::string &_value, MPI_Op _operation) -> std::string
{
    auto _bucket = std::string{};
    reduce(_value, _bucket, _operation);
    return _bucket;
}
auto reduce(const char *_value, MPI_Op _operation) -> std::string
{
    return reduce(std::string{_value}, _operation);
}
#pragma endregion
#pragma region scatter
auto scatter_impl(int _source, MPI_Comm _comm, const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void
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
    MPI_Scatter(_c_str.get(), _chunk_size, MPI_CHAR, _chunk.get(), _chunk_size, MPI_CHAR, _source, _comm);
    //at we need to write the value back
    _bucket = std::string{_chunk.get()};
}
#pragma endregion
#pragma region send
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Send(_value.c_str(), _value.size(), MPI_CHAR, _dest, _tag, _comm);
}
auto send_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    send_impl(_dest, _tag, _comm, std::string{_value});
}
#pragma endregion
#pragma region synchronized send
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Ssend(_value.c_str(), _value.size(), MPI_CHAR, _dest, _tag, _comm);
}
auto ssend_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void
{
    ssend_impl(_dest, _tag, _comm, std::string{_value});
}
#pragma endregion
#pragma region ready mode send
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const std::string &_value) -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Rsend(_value.c_str(), _value.size(), MPI_CHAR, _dest, _tag, _comm);
}
auto rsend_impl(int _dest, int _tag, MPI_Comm _comm, const char *_value) -> void
{
    rsend_impl(_dest, _tag, _comm, std::string{_value});
}
#pragma endregion

#pragma region communicator
communicator::communicator(MPI_Comm _comm) : _comm(_comm)
{
}
auto communicator::size() -> int
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    auto _size = int{};
    //add error checking
    MPI_Comm_size(_comm, &_size);
    return _size;
}
auto communicator::rank() -> int
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    auto _rank = int{};
    //add error checking
    MPI_Comm_rank(_comm, &_rank);
    return _rank;
}
auto communicator::name() -> std::string
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    auto _name = std::make_unique<char[]>(MPI_MAX_OBJECT_NAME);
    auto _size = MPI_MAX_OBJECT_NAME;
    //add error checking
    MPI_Comm_get_name(_comm, _name.get(), &_size);
    return std::string{_name.get()};
}

auto communicator::dest(int _dest) -> std::unique_ptr<sender>
{
    auto _tag = 0;
    return std::make_unique<sender>(_dest, _tag, _comm);
}
auto communicator::source(int _source) -> std::unique_ptr<receiver>
{
    auto _tag = 0;
    return std::make_unique<receiver>(_source, _tag, _comm);
}

auto communicator::allgather(const char _value, std::string &_bucket) -> void
{
    return allgather(std::string{_value}, _bucket);
}
auto communicator::allgather(const char *_value, std::string &_bucket) -> void
{
    return allgather(std::string{_value}, _bucket);
}
auto communicator::allgather(const std::string &_value, std::string &_bucket) -> void
{
    return allgather_impl(_comm, _value, _bucket);
}
auto communicator::allgather(const char _value) -> std::string
{
    return allgather(std::string{_value});
}
auto communicator::allgather(const char *_value) -> std::string
{
    return allgather(std::string{_value});
}
auto communicator::allgather(const std::string &_value) -> std::string
{
    auto _bucket = std::string{};
    allgather(_value, _bucket);
    return _bucket;
}

auto communicator::alltoall(const char _value, std::string &_bucket, const size_t _chunk_size) -> void
{
    return alltoall(std::string{_value}, _bucket, _chunk_size);
}
auto communicator::alltoall(const char *_value, std::string &_bucket, const size_t _chunk_size) -> void
{
    return alltoall(std::string{_value}, _bucket, _chunk_size);
}
auto communicator::alltoall(const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void
{
    return alltoall_impl(_comm, _value, _bucket, _chunk_size);
}
auto communicator::alltoall(const char _value, const size_t _chunk_size) -> std::string
{
    return alltoall(std::string{_value}, _chunk_size);
}
auto communicator::alltoall(const char *_value, const size_t _chunk_size) -> std::string
{
    return alltoall(std::string{_value}, _chunk_size);
}
auto communicator::alltoall(const std::string &_value, const size_t _chunk_size) -> std::string
{
    auto _bucket = std::string{};
    alltoall(_value, _bucket, _chunk_size);
    return _bucket;
}

auto communicator::barrier() -> void
{
    paranoidly_assert((initialized()));
    paranoidly_assert((!finalized()));
    MPI_Barrier(_comm);
}

auto communicator::allreduce(const char _value, std::string &_bucket, MPI_Op _operation) -> void
{
    return allreduce(std::string{_value}, _bucket, _operation);
}
auto communicator::allreduce(const char *_value, std::string &_bucket, MPI_Op _operation) -> void
{
    return allreduce(std::string{_value}, _bucket, _operation);
}
auto communicator::allreduce(const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void
{
    return allreduce_impl(_comm, _value, _bucket, _operation);
}
auto communicator::allreduce(const char _value, MPI_Op _operation) -> std::string
{
    return allreduce(std::string{_value}, _operation);
}
auto communicator::allreduce(const char *_value, MPI_Op _operation) -> std::string
{
    return allreduce(std::string{_value}, _operation);
}
auto communicator::allreduce(const std::string &_value, MPI_Op _operation) -> std::string
{
    auto _bucket = std::string{};
    allreduce(_value, _bucket, _operation);
    return _bucket;
}
#pragma endregion
#pragma region comm
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
#pragma endregion
#pragma region compare
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
#pragma endregion
#pragma region receiver
receiver::receiver(int _source, int _tag, MPI_Comm _comm) : _source(_source), _tag(_tag), _comm(_comm)
{
}

auto receiver::operator==(const receiver &rhs) -> bool
{
    return _source == rhs._source && _tag == rhs._tag && _comm == rhs._comm;
}
auto receiver::operator!=(const receiver &rhs) -> bool
{
    return !(*this == rhs);
}

auto receiver::scatter(const char *_value, std::string &_bucket, const size_t _chunk_size) -> void
{
    return scatter(std::string{_value}, _bucket, _chunk_size);
}
auto receiver::scatter(const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void
{
    return scatter_impl(_source, _comm, _value, _bucket, _chunk_size);
}
auto receiver::scatter(const char *_value, const size_t _chunk_size) -> std::string
{
    return scatter(std::string{_value}, _chunk_size);
}
auto receiver::scatter(const std::string &_value, const size_t _chunk_size) -> std::string
{
    auto _bucket = std::string{};
    scatter(_value, _bucket, _chunk_size);
    return _bucket;
}
#pragma endregion
#pragma region sender
sender::sender(int _dest, int _tag, MPI_Comm _comm) : _dest(_dest), _tag(_tag), _comm(_comm)
{
}

auto sender::operator==(const sender &rhs) -> bool
{
    return _dest == rhs._dest && _tag == rhs._tag && _comm == rhs._comm;
}
auto sender::operator!=(const sender &rhs) -> bool
{
    return !(*this == rhs);
}

auto sender::send(const char _value) -> void
{
    return send(std::string{_value});
}
auto sender::send(const char *_value) -> void
{
    return send(std::string{_value});
}
auto sender::send(const std::string &_value) -> void
{
    return send_impl(_dest, _tag, _comm, _value);
}
auto sender::ssend(const char _value) -> void
{
    return ssend(std::string{_value});
}
auto sender::ssend(const char *_value) -> void
{
    return ssend(std::string{_value});
}
auto sender::ssend(const std::string &_value) -> void
{
    return ssend_impl(_dest, _tag, _comm, _value);
}
auto sender::rsend(const char _value) -> void
{
    return rsend(std::string{_value});
}
auto sender::rsend(const char *_value) -> void
{
    return rsend(std::string{_value});
}
auto sender::rsend(const std::string &_value) -> void
{
    return rsend_impl(_dest, _tag, _comm, _value);
}

auto sender::gather(const char _value, std::string &_bucket) -> void
{
    return gather(std::string{_value}, _bucket);
}
auto sender::gather(const char *_value, std::string &_bucket) -> void
{
    return gather(std::string{_value}, _bucket);
}
auto sender::gather(const std::string &_value, std::string &_bucket) -> void
{
    return gather_impl(_dest, _comm, _value, _bucket);
}
auto sender::gather(const char _value) -> std::string
{
    return gather(std::string{_value});
}
auto sender::gather(const char *_value) -> std::string
{
    return gather(std::string{_value});
}
auto sender::gather(const std::string &_value) -> std::string
{
    auto _bucket = std::string{};
    gather(_value, _bucket);
    return _bucket;
}

auto sender::reduce(const char _value, std::string &_bucket, MPI_Op _operation) -> void
{
    return reduce(std::string{_value}, _bucket, _operation);
}
auto sender::reduce(const char *_value, std::string &_bucket, MPI_Op _operation) -> void
{
    return reduce(std::string{_value}, _bucket, _operation);
}
auto sender::reduce(const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void
{
    return reduce_impl(_dest, _comm, _value, _bucket, _operation);
}
auto sender::reduce(const char _value, MPI_Op _operation) -> std::string
{
    return reduce(std::string{_value}, _operation);
}
auto sender::reduce(const char *_value, MPI_Op _operation) -> std::string
{
    return reduce(std::string{_value}, _operation);
}
auto sender::reduce(const std::string &_value, MPI_Op _operation) -> std::string
{
    auto _bucket = std::string{};
    reduce(_value, _bucket, _operation);
    return _bucket;
}
#pragma endregion
} // namespace mpi