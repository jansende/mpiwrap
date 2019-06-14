#pragma once
#include <mpi.h>
#include <memory>
#include <string>
#include <vector>

//helper macro
#ifdef BE_PARANOID
#define paranoidly_assert(condition) assert(condition)
#else
#define paranoidly_assert(condition) ((void)0)
#endif

namespace mpi
{
#pragma region type wrapper
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
#pragma endregion

#pragma region free functions
auto initialized() -> bool;
auto finalized() -> bool;
auto processor_name() -> std::string;
#pragma endregion
#pragma region init
class mpi
{
public:
    mpi(const mpi &) = delete;
    mpi(mpi &&) = delete;
    mpi &operator=(const mpi &) = delete;
    mpi(int argc, char **argv);
    ~mpi();
};
#pragma endregion
#pragma region version information
class version_info
{
private:
    int _version;
    int _subversion;

public:
    version_info();
    auto version() -> int;
    auto subversion() -> int;
};
auto version() -> version_info;
#pragma endregion

#pragma region communicator
//declarations
class sender;
class receiver;
template <class T, class Op>
class op_proxy;

class communicator
{
protected:
    MPI_Comm _comm;

public:
    communicator(MPI_Comm _comm);

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

    auto size() -> int;
    auto rank() -> int;
    auto name() -> std::string;

    auto dest(int _dest) -> std::unique_ptr<sender>;
    auto source(int _source) -> std::unique_ptr<receiver>;

    template <class T>
    auto allgather(const T &_value, std::vector<T> &_bucket) -> void;
    template <class T>
    auto allgather(const std::vector<T> &_value, std::vector<T> &_bucket) -> void;
    auto allgather(const char _value, std::string &_bucket) -> void;
    auto allgather(const char *_value, std::string &_bucket) -> void;
    auto allgather(const std::string &_value, std::string &_bucket) -> void;
    template <class T>
    auto allgather(const T &_value) -> std::vector<T>;
    template <class T>
    auto allgather(const std::vector<T> &_value) -> std::vector<T>;
    auto allgather(const char _value) -> std::string;
    auto allgather(const char *_value) -> std::string;
    auto allgather(const std::string &_value) -> std::string;

    template <class T>
    auto alltoall(const T &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void;
    template <class T>
    auto alltoall(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void;
    auto alltoall(const char _value, std::string &_bucket, const size_t _chunk_size) -> void;
    auto alltoall(const char *_value, std::string &_bucket, const size_t _chunk_size) -> void;
    auto alltoall(const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void;
    template <class T>
    auto alltoall(const T &_value, const size_t _chunk_size) -> std::vector<T>;
    template <class T>
    auto alltoall(const std::vector<T> &_value, const size_t _chunk_size) -> std::vector<T>;
    auto alltoall(const char _value, const size_t _chunk_size) -> std::string;
    auto alltoall(const char *_value, const size_t _chunk_size) -> std::string;
    auto alltoall(const std::string &_value, const size_t _chunk_size) -> std::string;

    auto barrier() -> void;

    template <class T>
    auto allreduce(const T &_value, T &_bucket, MPI_Op _operation) -> void;
    template <class T>
    auto allreduce(const std::vector<T> &_value, std::vector<T> &_bucket, MPI_Op _operation) -> void;
    auto allreduce(const char _value, std::string &_bucket, MPI_Op _operation) -> void;
    auto allreduce(const char *_value, std::string &_bucket, MPI_Op _operation) -> void;
    auto allreduce(const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void;
    template <class T>
    auto allreduce(const T &_value, MPI_Op _operation) -> T;
    template <class T>
    auto allreduce(const std::vector<T> &_value, MPI_Op _operation) -> std::vector<T>;
    auto allreduce(const char _value, MPI_Op _operation) -> std::string;
    auto allreduce(const char *_value, MPI_Op _operation) -> std::string;
    auto allreduce(const std::string &_value, MPI_Op _operation) -> std::string;

    template <class T, class Op>
    auto allreduce(const T &_value, T &_bucket, const op_proxy<T, Op> *_operation) -> void;
    template <class T, class Op>
    auto allreduce(const std::vector<T> &_value, std::vector<T> &_bucket, const op_proxy<T, Op> *_operation) -> void;
    template <class Op>
    auto allreduce(const char _value, std::string &_bucket, const op_proxy<std::string, Op> *_operation) -> void;
    template <class Op>
    auto allreduce(const char *_value, std::string &_bucket, const op_proxy<std::string, Op> *_operation) -> void;
    template <class Op>
    auto allreduce(const std::string &_value, std::string &_bucket, const op_proxy<std::string, Op> *_operation) -> void;
    template <class T, class Op>
    auto allreduce(const T &_value, const op_proxy<T, Op> *_operation) -> T;
    template <class T, class Op>
    auto allreduce(const std::vector<T> &_value, const op_proxy<T, Op> *_operation) -> std::vector<T>;
    template <class Op>
    auto allreduce(const char _value, const op_proxy<std::string, Op> *_operation) -> std::string;
    template <class Op>
    auto allreduce(const char *_value, const op_proxy<std::string, Op> *_operation) -> std::string;
    template <class Op>
    auto allreduce(const std::string &_value, const op_proxy<std::string, Op> *_operation) -> std::string;
};
#pragma endregion
#pragma region comm
auto comm(MPI_Comm _comm) -> std::unique_ptr<communicator>;
auto comm(const std::string &_name) -> std::unique_ptr<communicator>;
#pragma endregion
#pragma region compare
auto compare(const MPI_Comm &lhs, const MPI_Comm &rhs) -> communicator::comp;
auto compare(const communicator &lhs, const MPI_Comm &rhs) -> communicator::comp;
auto compare(const MPI_Comm &lhs, const communicator &rhs) -> communicator::comp;
auto compare(const communicator &lhs, const communicator &rhs) -> communicator::comp;
#pragma endregion
#pragma region operation wrapper
template <class T, class Op>
class op_proxy
{
private:
    MPI_Op _operation;
    const bool _commute;

    static auto wrapper(void *void_a, void *void_b, int *len, MPI_Datatype *) -> void;

public:
    op_proxy(const op_proxy &) = delete;
    op_proxy(op_proxy &&) = delete;
    op_proxy &operator=(const op_proxy &) = delete;

    op_proxy(Op, const bool _commute);
    ~op_proxy();

    auto op() const -> const MPI_Op &;
    auto commutes() const -> bool;
};

template <class T, class Op>
auto make_op(Op _func, const bool _commute = false) -> std::unique_ptr<op_proxy<T, Op>>;
#pragma endregion
#pragma region receiver
class receiver
{
private:
    int _source;
    int _tag;
    MPI_Comm _comm;
    MPI_Status _status;

public:
    receiver(int _source, int _tag, MPI_Comm _comm);

    auto operator==(const receiver &rhs) -> bool;
    auto operator!=(const receiver &rhs) -> bool;

    template <class T>
    auto recv(T &_value) -> void;
    template <class T>
    auto recv() -> T;

    template <class T>
    auto bcast(T &_value) -> void;
    template <class R, class T>
    auto bcast(const T &_value) -> std::enable_if_t<std::is_same<R, T>::value, T>;
    template <class R, class T>
    auto bcast(const std::vector<T> &_value) -> std::enable_if_t<std::is_same<R, std::vector<T>>::value, std::vector<T>>;
    template <class R>
    auto bcast(const char _value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::string>;
    template <class R>
    auto bcast(const char *_value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::string>;
    template <class R>
    auto bcast(const std::string &_value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::string>;

    template <class T>
    auto scatter(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void;
    auto scatter(const char *_value, std::string &_bucket, const size_t _chunk_size) -> void;
    auto scatter(const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void;
    template <class T>
    auto scatter(const std::vector<T> &_value, const size_t _chunk_size) -> std::vector<T>;
    auto scatter(const char *_value, const size_t _chunk_size) -> std::string;
    auto scatter(const std::string &_value, const size_t _chunk_size) -> std::string;
};
#pragma endregion
#pragma region sender
class sender
{
private:
    int _dest;
    int _tag;
    MPI_Comm _comm;

public:
    sender(int _dest, int _tag, MPI_Comm _comm);

    auto operator==(const sender &rhs) -> bool;
    auto operator!=(const sender &rhs) -> bool;

    template <class T>
    auto send(const T &_value) -> void;
    template <class T>
    auto send(const std::vector<T> &_value) -> void;
    auto send(const char _value) -> void;
    auto send(const char *_value) -> void;
    auto send(const std::string &_value) -> void;
    template <class T>
    auto ssend(const T &_value) -> void;
    template <class T>
    auto ssend(const std::vector<T> &_value) -> void;
    auto ssend(const char _value) -> void;
    auto ssend(const char *_value) -> void;
    auto ssend(const std::string &_value) -> void;
    template <class T>
    auto rsend(const T &_value) -> void;
    template <class T>
    auto rsend(const std::vector<T> &_value) -> void;
    auto rsend(const char _value) -> void;
    auto rsend(const char *_value) -> void;
    auto rsend(const std::string &_value) -> void;

    template <class T>
    auto gather(const T &_value, std::vector<T> &_bucket) -> void;
    template <class T>
    auto gather(const std::vector<T> &_value, std::vector<T> &_bucket) -> void;
    auto gather(const char _value, std::string &_bucket) -> void;
    auto gather(const char *_value, std::string &_bucket) -> void;
    auto gather(const std::string &_value, std::string &_bucket) -> void;
    template <class T>
    auto gather(const T &_value) -> std::vector<T>;
    template <class T>
    auto gather(const std::vector<T> &_value) -> std::vector<T>;
    auto gather(const char _value) -> std::string;
    auto gather(const char *_value) -> std::string;
    auto gather(const std::string &_value) -> std::string;

    template <class T>
    auto reduce(const T &_value, T &_bucket, MPI_Op _operation) -> void;
    template <class T>
    auto reduce(const std::vector<T> &_value, std::vector<T> &_bucket, MPI_Op _operation) -> void;
    auto reduce(const char _value, std::string &_bucket, MPI_Op _operation) -> void;
    auto reduce(const char *_value, std::string &_bucket, MPI_Op _operation) -> void;
    auto reduce(const std::string &_value, std::string &_bucket, MPI_Op _operation) -> void;
    template <class T>
    auto reduce(const T &_value, MPI_Op _operation) -> T;
    template <class T>
    auto reduce(const std::vector<T> &_value, MPI_Op _operation) -> std::vector<T>;
    auto reduce(const char _value, MPI_Op _operation) -> std::string;
    auto reduce(const char *_value, MPI_Op _operation) -> std::string;
    auto reduce(const std::string &_value, MPI_Op _operation) -> std::string;

    template <class T, class Op>
    auto reduce(const T &_value, T &_bucket, const op_proxy<T, Op> *_operation) -> void;
    template <class T, class Op>
    auto reduce(const std::vector<T> &_value, std::vector<T> &_bucket, const op_proxy<T, Op> *_operation) -> void;
    template <class Op>
    auto reduce(const char _value, std::string &_bucket, const op_proxy<std::string, Op> *_operation) -> void;
    template <class Op>
    auto reduce(const char *_value, std::string &_bucket, const op_proxy<std::string, Op> *_operation) -> void;
    template <class Op>
    auto reduce(const std::string &_value, std::string &_bucket, const op_proxy<std::string, Op> *_operation) -> void;
    template <class T, class Op>
    auto reduce(const T &_value, const op_proxy<T, Op> *_operation) -> T;
    template <class T, class Op>
    auto reduce(const std::vector<T> &_value, const op_proxy<T, Op> *_operation) -> std::vector<T>;
    template <class Op>
    auto reduce(const char _value, const op_proxy<std::string, Op> *_operation) -> std::string;
    template <class Op>
    auto reduce(const char *_value, const op_proxy<std::string, Op> *_operation) -> std::string;
    template <class Op>
    auto reduce(const std::string &_value, const op_proxy<std::string, Op> *_operation) -> std::string;
};
#pragma endregion
} // namespace mpi

//finally include the definitions
#include <mpiwrap/mpi.tpp>