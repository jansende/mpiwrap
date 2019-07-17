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
class op;
class ibarrier_request;
template <class T>
class iallgather_request;
template <class T>
class iallgather_reply;
template <class T>
class ialltoall_request;
template <class T>
class ialltoall_reply;
template <class T>
class iallreduce_request;
template <class T>
class iallreduce_reply;

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

#pragma region allgather
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
#pragma endregion
#pragma region iallgather
    template <class T>
    auto iallgather(const T &_value, std::vector<T> &_bucket) -> std::unique_ptr<iallgather_request<std::vector<T>>>;
    template <class T>
    auto iallgather(const std::vector<T> &_value, std::vector<T> &_bucket) -> std::unique_ptr<iallgather_request<std::vector<T>>>;
    auto iallgather(const char _value, std::string &_bucket) -> std::unique_ptr<iallgather_request<std::string>>;
    auto iallgather(const char *_value, std::string &_bucket) -> std::unique_ptr<iallgather_request<std::string>>;
    auto iallgather(const std::string &_value, std::string &_bucket) -> std::unique_ptr<iallgather_request<std::string>>;
    template <class T>
    auto iallgather(const T &_value) -> std::unique_ptr<iallgather_reply<std::vector<T>>>;
    template <class T>
    auto iallgather(const std::vector<T> &_value) -> std::unique_ptr<iallgather_reply<std::vector<T>>>;
    auto iallgather(const char _value) -> std::unique_ptr<iallgather_reply<std::string>>;
    auto iallgather(const char *_value) -> std::unique_ptr<iallgather_reply<std::string>>;
    auto iallgather(const std::string &_value) -> std::unique_ptr<iallgather_reply<std::string>>;
#pragma endregion
#pragma region alltoall
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
#pragma endregion
#pragma region ialltoall
    template <class T>
    auto ialltoall(const T &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> std::unique_ptr<ialltoall_request<std::vector<T>>>;
    template <class T>
    auto ialltoall(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> std::unique_ptr<ialltoall_request<std::vector<T>>>;
    auto ialltoall(const char _value, std::string &_bucket, const size_t _chunk_size) -> std::unique_ptr<ialltoall_request<std::string>>;
    auto ialltoall(const char *_value, std::string &_bucket, const size_t _chunk_size) -> std::unique_ptr<ialltoall_request<std::string>>;
    auto ialltoall(const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> std::unique_ptr<ialltoall_request<std::string>>;
    template <class T>
    auto ialltoall(const T &_value, const size_t _chunk_size) -> std::unique_ptr<ialltoall_reply<std::vector<T>>>;
    template <class T>
    auto ialltoall(const std::vector<T> &_value, const size_t _chunk_size) -> std::unique_ptr<ialltoall_reply<std::vector<T>>>;
    auto ialltoall(const char _value, const size_t _chunk_size) -> std::unique_ptr<ialltoall_reply<std::string>>;
    auto ialltoall(const char *_value, const size_t _chunk_size) -> std::unique_ptr<ialltoall_reply<std::string>>;
    auto ialltoall(const std::string &_value, const size_t _chunk_size) -> std::unique_ptr<ialltoall_reply<std::string>>;
#pragma endregion
#pragma region barrier
    auto barrier() -> void;
#pragma endregion
#pragma region ibarrier
    auto ibarrier() -> std::unique_ptr<ibarrier_request>;
#pragma endregion
#pragma region allreduce
    template <class T>
    auto allreduce(const T &_value, T &_bucket, op *_operation) -> void;
    template <class T>
    auto allreduce(const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void;
    auto allreduce(const char _value, std::string &_bucket, op *_operation) -> void;
    auto allreduce(const char *_value, std::string &_bucket, op *_operation) -> void;
    auto allreduce(const std::string &_value, std::string &_bucket, op *_operation) -> void;
    template <class T>
    auto allreduce(const T &_value, op *_operation) -> T;
    template <class T>
    auto allreduce(const std::vector<T> &_value, op *_operation) -> std::vector<T>;
    auto allreduce(const char _value, op *_operation) -> std::string;
    auto allreduce(const char *_value, op *_operation) -> std::string;
    auto allreduce(const std::string &_value, op *_operation) -> std::string;

    template <class T, class Op>
    auto allreduce(const T &_value, T &_bucket, Op _operation) -> void;
    template <class T, class Op>
    auto allreduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> void;
    template <class Op>
    auto allreduce(const char _value, std::string &_bucket, Op _operation) -> void;
    template <class Op>
    auto allreduce(const char *_value, std::string &_bucket, Op _operation) -> void;
    template <class Op>
    auto allreduce(const std::string &_value, std::string &_bucket, Op _operation) -> void;
    template <class T, class Op>
    auto allreduce(const T &_value, Op _operation) -> T;
    template <class T, class Op>
    auto allreduce(const std::vector<T> &_value, Op _operation) -> std::vector<T>;
    template <class Op>
    auto allreduce(const char _value, Op _operation) -> std::string;
    template <class Op>
    auto allreduce(const char *_value, Op _operation) -> std::string;
    template <class Op>
    auto allreduce(const std::string &_value, Op _operation) -> std::string;
#pragma endregion
#pragma region iallreduce
    template <class T>
    auto iallreduce(const T &_value, T &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_request<T>>;
    template <class T>
    auto iallreduce(const std::vector<T> &_value, std::vector<T> &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_request<std::vector<T>>>;
    auto iallreduce(const char _value, std::string &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_request<std::string>>;
    auto iallreduce(const char *_value, std::string &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_request<std::string>>;
    auto iallreduce(const std::string &_value, std::string &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_request<std::string>>;
    template <class T>
    auto iallreduce(const T &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_reply<T>>;
    template <class T>
    auto iallreduce(const std::vector<T> &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_reply<std::vector<T>>>;
    auto iallreduce(const char _value, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_reply<std::string>>;
    auto iallreduce(const char *_value, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_reply<std::string>>;
    auto iallreduce(const std::string &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<iallreduce_reply<std::string>>;

    template <class T, class Op>
    auto iallreduce(const T &_value, T &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<T>>;
    template <class T, class Op>
    auto iallreduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<std::vector<T>>>;
    template <class Op>
    auto iallreduce(const char _value, std::string &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<std::string>>;
    template <class Op>
    auto iallreduce(const char *_value, std::string &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<std::string>>;
    template <class Op>
    auto iallreduce(const std::string &_value, std::string &_bucket, Op _operation) -> std::unique_ptr<iallreduce_request<std::string>>;
    template <class T, class Op>
    auto iallreduce(const T &_value, Op _operation) -> std::unique_ptr<iallreduce_reply<T>>;
    template <class T, class Op>
    auto iallreduce(const std::vector<T> &_value, Op _operation) -> std::unique_ptr<iallreduce_reply<std::vector<T>>>;
    template <class Op>
    auto iallreduce(const char _value, Op _operation) -> std::unique_ptr<iallreduce_reply<std::string>>;
    template <class Op>
    auto iallreduce(const char *_value, Op _operation) -> std::unique_ptr<iallreduce_reply<std::string>>;
    template <class Op>
    auto iallreduce(const std::string &_value, Op _operation) -> std::unique_ptr<iallreduce_reply<std::string>>;
#pragma endregion
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
class op
{
protected:
    MPI_Op _operation;
    const bool _commute;
    op(MPI_Op _operation, const bool _commute);
    op(const bool _commute);

public:
    op(const op &) = delete;
    op(op &&) = delete;
    op &operator=(const op &) = delete;

    virtual ~op() = default;

    auto get() const -> const MPI_Op &;
    auto commutes() const -> bool;
};

template <class T, class Op>
class op_proxy : public op
{
private:
    static auto wrapper(void *void_a, void *void_b, int *len, MPI_Datatype *) -> void;

public:
    using op::op;

    op_proxy(const bool _commute);
    virtual ~op_proxy();
};

template <class T, class Op>
auto make_op(Op _func, const bool _commute = false) -> std::shared_ptr<op>;
#pragma endregion
#pragma region request
class request
{
    friend auto testall(const std::vector<request *> &_values) -> bool;
    friend auto testany(const std::vector<request *> &_values) -> std::vector<size_t>;
    friend auto testsome(const std::vector<request *> &_values) -> std::vector<size_t>;

    friend auto waitall(const std::vector<request *> &_values) -> void;
    friend auto waitany(const std::vector<request *> &_values) -> std::vector<size_t>;
    friend auto waitsome(const std::vector<request *> &_values) -> std::vector<size_t>;

protected:
    MPI_Comm _comm;
    MPI_Request _request;
    MPI_Status _status;
    bool is_finished = false;
    bool is_canceled = false;

    request(MPI_Comm _comm);
    virtual ~request();

public:
    virtual auto cancel() -> void;
    virtual auto test() -> bool;
    virtual auto wait() -> void;
};
#pragma endregion
#pragma region request implementations
#pragma region ibarrier
class ibarrier_request : public request
{
public:
    ibarrier_request(MPI_Comm _comm);
};
#pragma endregion
#pragma region isend
template <class T>
class isend_request : public request
{
private:
    int _dest;
    int _tag;
    T _value;

public:
    isend_request(int _dest, int _tag, MPI_Comm _comm, const T &_value);
};
#pragma endregion
#pragma region issend
template <class T>
class issend_request : public request
{
private:
    int _dest;
    int _tag;
    T _value;

public:
    issend_request(int _dest, int _tag, MPI_Comm _comm, const T &_value);
};
#pragma endregion
#pragma region irsend
template <class T>
class irsend_request : public request
{
private:
    int _dest;
    int _tag;
    T _value;

public:
    irsend_request(int _dest, int _tag, MPI_Comm _comm, const T &_value);
};
#pragma endregion
#pragma region irecv
template <class T>
class irecv_request : public request
{
private:
    int _source;
    int _tag;

public:
    irecv_request(int _source, int _tag, MPI_Comm _comm, T &_value);
};
template <>
class irecv_request<std::string> : public request
{
private:
    int _source;
    int _tag;
    std::unique_ptr<char[]> _c_str;
    std::string &_bucket;

public:
    irecv_request(int _source, int _tag, MPI_Comm _comm, std::string &_value);
    virtual auto wait() -> void;
};

template <class T>
class irecv_reply : public request
{
private:
    int _source;
    int _tag;
    T _bucket;

public:
    irecv_reply(int _source, int _tag, MPI_Comm _comm);
    auto get() -> T;
};
template <>
class irecv_reply<std::string> : public request
{
private:
    int _source;
    int _tag;
    std::unique_ptr<char[]> _c_str;

public:
    irecv_reply(int _source, int _tag, MPI_Comm _comm);
    auto get() -> std::string;
};
#pragma endregion
#pragma region ibcast
template <class T>
class ibcast_request : public request
{
private:
    int _source;

public:
    ibcast_request(int _source, MPI_Comm _comm, T &_value);
};
template <>
class ibcast_request<std::string> : public request
{
private:
    int _source;
    std::unique_ptr<char[]> _c_str;
    std::string &_bucket;

public:
    ibcast_request(int _source, MPI_Comm _comm, std::string &_value);
    virtual auto wait() -> void;
};
template <class T>
class ibcast_reply : public request
{
private:
    int _source;
    T _bucket;

public:
    ibcast_reply(int _source, MPI_Comm _comm, const T &_value);
    auto get() -> T;
};
template <>
class ibcast_reply<std::string> : public request
{
private:
    int _source;
    std::unique_ptr<char[]> _c_str;

public:
    ibcast_reply(int _source, MPI_Comm _comm, const std::string &_value);
    auto get() -> std::string;
};
#pragma endregion
#pragma region iscatter
template <class T>
class iscatter_request : public request
{
private:
    int _source;
    size_t _chunk_size;
    T _value;
    T &_bucket;

public:
    iscatter_request(int _source, MPI_Comm _comm, const T &_value, T &_bucket, const size_t _chunk_size);
};
template <>
class iscatter_request<std::string> : public request
{
private:
    int _source;
    size_t _chunk_size;
    std::string _value;
    std::unique_ptr<char[]> _c_str;
    std::string &_bucket;

public:
    iscatter_request(int _source, MPI_Comm _comm, const std::string &_value, std::string &_bucket, const size_t _chunk_size);
    virtual auto wait() -> void;
};
template <class T>
class iscatter_reply : public request
{
private:
    int _source;
    size_t _chunk_size;
    T _value;
    T _bucket;

public:
    iscatter_reply(int _source, MPI_Comm _comm, const T &_value, const size_t _chunk_size);
    auto get() -> T;
};
template <>
class iscatter_reply<std::string> : public request
{
private:
    int _source;
    size_t _chunk_size;
    std::string _value;
    std::unique_ptr<char[]> _c_str;

public:
    iscatter_reply(int _source, MPI_Comm _comm, const std::string &_value, const size_t _chunk_size);
    auto get() -> std::string;
};
#pragma endregion
#pragma region igather
template <class T>
class igather_request : public request
{
private:
    int _dest;
    T _value;
    T &_bucket;

public:
    igather_request(int _dest, MPI_Comm _comm, const T &_value, T &_bucket);
};
template <>
class igather_request<std::string> : public request
{
private:
    int _dest;
    std::string _value;
    std::unique_ptr<char[]> _c_str;
    std::string &_bucket;

public:
    igather_request(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket);
    virtual auto wait() -> void;
};
template <class T>
class igather_reply : public request
{
private:
    int _dest;
    T _value;
    T _bucket;

public:
    igather_reply(int _dest, MPI_Comm _comm, const T &_value);
    auto get() -> T;
};
template <>
class igather_reply<std::string> : public request
{
private:
    int _dest;
    std::string _value;
    std::unique_ptr<char[]> _c_str;

public:
    igather_reply(int _dest, MPI_Comm _comm, const std::string &_value);
    auto get() -> std::string;
};
#pragma endregion
#pragma region iallgather
template <class T>
class iallgather_request : public request
{
private:
    T _value;
    T &_bucket;

public:
    iallgather_request(MPI_Comm _comm, const T &_value, T &_bucket);
};
template <>
class iallgather_request<std::string> : public request
{
private:
    std::string _value;
    std::unique_ptr<char[]> _c_str;
    std::string &_bucket;

public:
    iallgather_request(MPI_Comm _comm, const std::string &_value, std::string &_bucket);
    virtual auto wait() -> void;
};
template <class T>
class iallgather_reply : public request
{
private:
    T _value;
    T _bucket;

public:
    iallgather_reply(MPI_Comm _comm, const T &_value);
    auto get() -> T;
};
template <>
class iallgather_reply<std::string> : public request
{
private:
    std::string _value;
    std::unique_ptr<char[]> _c_str;

public:
    iallgather_reply(MPI_Comm _comm, const std::string &_value);
    auto get() -> std::string;
};
#pragma endregion
#pragma region ialltoall
template <class T>
class ialltoall_request : public request
{
private:
    size_t _chunk_size;
    T _value;
    T &_bucket;

public:
    ialltoall_request(MPI_Comm _comm, const T &_value, T &_bucket, const size_t _chunk_size);
};
template <>
class ialltoall_request<std::string> : public request
{
private:
    size_t _chunk_size;
    std::string _value;
    std::unique_ptr<char[]> _c_str;
    std::string &_bucket;

public:
    ialltoall_request(MPI_Comm _comm, const std::string &_value, std::string &_bucket, const size_t _chunk_size);
    virtual auto wait() -> void;
};
template <class T>
class ialltoall_reply : public request
{
private:
    size_t _chunk_size;
    T _value;
    T _bucket;

public:
    ialltoall_reply(MPI_Comm _comm, const T &_value, const size_t _chunk_size);
    auto get() -> T;
};
template <>
class ialltoall_reply<std::string> : public request
{
private:
    size_t _chunk_size;
    std::string _value;
    std::unique_ptr<char[]> _c_str;

public:
    ialltoall_reply(MPI_Comm _comm, const std::string &_value, const size_t _chunk_size);
    auto get() -> std::string;
};
#pragma endregion
#pragma region ireduce
template <class T>
class ireduce_request : public request
{
private:
    std::shared_ptr<op> _operation;
    int _dest;
    T _value;
    T &_bucket;

public:
    ireduce_request(int _dest, MPI_Comm _comm, const T &_value, T &_bucket, std::shared_ptr<op> _operation);
};
template <>
class ireduce_request<std::string> : public request
{
private:
    std::shared_ptr<op> _operation;
    int _dest;
    std::string _value;
    std::unique_ptr<char[]> _c_str;
    std::string &_bucket;

public:
    ireduce_request(int _dest, MPI_Comm _comm, const std::string &_value, std::string &_bucket, std::shared_ptr<op> _operation);
    virtual auto wait() -> void;
};
template <class T>
class ireduce_reply : public request
{
private:
    std::shared_ptr<op> _operation;
    int _dest;
    T _value;
    T _bucket;

public:
    ireduce_reply(int _dest, MPI_Comm _comm, const T &_value, std::shared_ptr<op> _operation);
    auto get() -> T;
};
template <>
class ireduce_reply<std::string> : public request
{
private:
    std::shared_ptr<op> _operation;
    int _dest;
    std::string _value;
    std::unique_ptr<char[]> _c_str;

public:
    ireduce_reply(int _dest, MPI_Comm _comm, const std::string &_value, std::shared_ptr<op> _operation);
    auto get() -> std::string;
};
#pragma endregion
#pragma region iallreduce
template <class T>
class iallreduce_request : public request
{
private:
    std::shared_ptr<op> _operation;
    T _value;
    T &_bucket;

public:
    iallreduce_request(MPI_Comm _comm, const T &_value, T &_bucket, std::shared_ptr<op> _operation);
};
template <>
class iallreduce_request<std::string> : public request
{
private:
    std::shared_ptr<op> _operation;
    std::string _value;
    std::unique_ptr<char[]> _c_str;
    std::string &_bucket;

public:
    iallreduce_request(MPI_Comm _comm, const std::string &_value, std::string &_bucket, std::shared_ptr<op> _operation);
    virtual auto wait() -> void;
};
template <class T>
class iallreduce_reply : public request
{
private:
    std::shared_ptr<op> _operation;
    T _value;
    T _bucket;

public:
    iallreduce_reply(MPI_Comm _comm, const T &_value, std::shared_ptr<op> _operation);
    auto get() -> T;
};
template <>
class iallreduce_reply<std::string> : public request
{
private:
    std::shared_ptr<op> _operation;
    std::string _value;
    std::unique_ptr<char[]> _c_str;

public:
    iallreduce_reply(MPI_Comm _comm, const std::string &_value, std::shared_ptr<op> _operation);
    auto get() -> std::string;
};
#pragma endregion
#pragma endregion
#pragma region test
auto test(request *_value) -> bool;
auto test(std::unique_ptr<request> &_value) -> bool;
auto testall(const std::vector<std::unique_ptr<request>> &_values) -> bool;
template <class... T>
auto testall(std::unique_ptr<T> &... _values) -> bool;
template <class... T>
auto testall(T *... _values) -> bool;
auto testany(const std::vector<std::unique_ptr<request>> &_values) -> std::vector<size_t>;
template <class... T>
auto testany(std::unique_ptr<T> &... _values) -> std::vector<size_t>;
template <class... T>
auto testany(T *... _values) -> std::vector<size_t>;
auto testsome(const std::vector<std::unique_ptr<request>> &_values) -> std::vector<size_t>;
template <class... T>
auto testsome(std::unique_ptr<T> &... _values) -> std::vector<size_t>;
template <class... T>
auto testsome(T *... _values) -> std::vector<size_t>;
#pragma endregion
#pragma region wait
auto wait(request *_value) -> void;
auto wait(std::unique_ptr<request> &_value) -> void;
auto waitall(const std::vector<std::unique_ptr<request>> &_values) -> void;
template <class... T>
auto waitall(std::unique_ptr<T> &... _values) -> void;
template <class... T>
auto waitall(T *... _values) -> void;
auto waitany(const std::vector<std::unique_ptr<request>> &_values) -> std::vector<size_t>;
template <class... T>
auto waitany(std::unique_ptr<T> &... _values) -> std::vector<size_t>;
template <class... T>
auto waitany(T *... _values) -> std::vector<size_t>;
auto waitsome(const std::vector<std::unique_ptr<request>> &_values) -> std::vector<size_t>;
template <class... T>
auto waitsome(std::unique_ptr<T> &... _values) -> std::vector<size_t>;
template <class... T>
auto waitsome(T *... _values) -> std::vector<size_t>;
#pragma endregion
#pragma region sender_receiver
class sender_receiver
{
private:
    int _dest;
    int _source;
    int _sendtag;
    int _recvtag;
    MPI_Comm _comm;
    MPI_Status _status;

public:
    sender_receiver(int _dest, int _source, int _sendtag, int _recvtag, MPI_Comm _comm);

    auto operator==(const sender_receiver &rhs) -> bool;
    auto operator!=(const sender_receiver &rhs) -> bool;

#pragma region sendrecv
    template <class T, class U>
    auto sendrecv(const U &_value, T &_bucket) -> void;
    template <class T, class U>
    auto sendrecv(const std::vector<U> &_value, T &_bucket) -> void;
    template <class T>
    auto sendrecv(const char _value, T &_bucket) -> void;
    template <class T>
    auto sendrecv(const char *_value, T &_bucket) -> void;
    template <class T>
    auto sendrecv(const std::string &_value, T &_bucket) -> void;

    template <class T, class U>
    auto sendrecv(const U &_value) -> T;
    template <class T, class U>
    auto sendrecv(const std::vector<U> &_value) -> T;
    template <class T>
    auto sendrecv(const char _value) -> T;
    template <class T>
    auto sendrecv(const char *_value) -> T;
    template <class T>
    auto sendrecv(const std::string &_value) -> T;
#pragma endregion
#pragma region sendrecv_replace
    template <class T>
    auto sendrecv_replace(T &_value) -> void;
    auto sendrecv_replace(std::string &_value) -> void;
#pragma endregion
};
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

    auto dest(int _dest) -> std::unique_ptr<sender_receiver>;

#pragma region recv
    template <class T>
    auto recv(T &_value) -> void;
    template <class T>
    auto recv() -> T;
#pragma endregion
#pragma region irecv
    template <class T>
    auto irecv(T &_value) -> std::unique_ptr<irecv_request<T>>;
    template <class T>
    auto irecv() -> std::unique_ptr<irecv_reply<T>>;
#pragma endregion
#pragma region bcast
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
#pragma endregion
#pragma region ibcast
    template <class T>
    auto ibcast(T &_value) -> std::unique_ptr<ibcast_request<T>>;
    template <class R, class T>
    auto ibcast(const T &_value) -> std::enable_if_t<std::is_same<R, T>::value, std::unique_ptr<ibcast_reply<T>>>;
    template <class R, class T>
    auto ibcast(const std::vector<T> &_value) -> std::enable_if_t<std::is_same<R, std::vector<T>>::value, std::unique_ptr<ibcast_reply<std::vector<T>>>>;
    template <class R>
    auto ibcast(const char _value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::unique_ptr<ibcast_reply<std::string>>>;
    template <class R>
    auto ibcast(const char *_value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::unique_ptr<ibcast_reply<std::string>>>;
    template <class R>
    auto ibcast(const std::string &_value) -> std::enable_if_t<std::is_same<R, std::string>::value, std::unique_ptr<ibcast_reply<std::string>>>;
#pragma endregion
#pragma region scatter
    template <class T>
    auto scatter(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> void;
    auto scatter(const char *_value, std::string &_bucket, const size_t _chunk_size) -> void;
    auto scatter(const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> void;
    template <class T>
    auto scatter(const std::vector<T> &_value, const size_t _chunk_size) -> std::vector<T>;
    auto scatter(const char *_value, const size_t _chunk_size) -> std::string;
    auto scatter(const std::string &_value, const size_t _chunk_size) -> std::string;
#pragma endregion
#pragma region iscatter
    template <class T>
    auto iscatter(const std::vector<T> &_value, std::vector<T> &_bucket, const size_t _chunk_size) -> std::unique_ptr<iscatter_request<std::vector<T>>>;
    auto iscatter(const char *_value, std::string &_bucket, const size_t _chunk_size) -> std::unique_ptr<iscatter_request<std::string>>;
    auto iscatter(const std::string &_value, std::string &_bucket, const size_t _chunk_size) -> std::unique_ptr<iscatter_request<std::string>>;
    template <class T>
    auto iscatter(const std::vector<T> &_value, const size_t _chunk_size) -> std::unique_ptr<iscatter_reply<std::vector<T>>>;
    auto iscatter(const char *_value, const size_t _chunk_size) -> std::unique_ptr<iscatter_reply<std::string>>;
    auto iscatter(const std::string &_value, const size_t _chunk_size) -> std::unique_ptr<iscatter_reply<std::string>>;
#pragma endregion
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

    auto source(int _source) -> std::unique_ptr<sender_receiver>;

#pragma region send
    template <class T>
    auto send(const T &_value) -> void;
    template <class T>
    auto send(const std::vector<T> &_value) -> void;
    auto send(const char _value) -> void;
    auto send(const char *_value) -> void;
    auto send(const std::string &_value) -> void;
#pragma endregion
#pragma region isend
    template <class T>
    auto isend(const T &_value) -> std::unique_ptr<isend_request<T>>;
    template <class T>
    auto isend(const std::vector<T> &_value) -> std::unique_ptr<isend_request<std::vector<T>>>;
    auto isend(const char _value) -> std::unique_ptr<isend_request<std::string>>;
    auto isend(const char *_value) -> std::unique_ptr<isend_request<std::string>>;
    auto isend(const std::string &_value) -> std::unique_ptr<isend_request<std::string>>;
    template <class T>
    auto issend(const T &_value) -> std::unique_ptr<issend_request<T>>;
#pragma endregion
#pragma region ssend
    template <class T>
    auto ssend(const T &_value) -> void;
    template <class T>
    auto ssend(const std::vector<T> &_value) -> void;
    auto ssend(const char _value) -> void;
    auto ssend(const char *_value) -> void;
    auto ssend(const std::string &_value) -> void;
#pragma endregion
#pragma region issend
    template <class T>
    auto issend(const std::vector<T> &_value) -> std::unique_ptr<issend_request<std::vector<T>>>;
    auto issend(const char _value) -> std::unique_ptr<issend_request<std::string>>;
    auto issend(const char *_value) -> std::unique_ptr<issend_request<std::string>>;
    auto issend(const std::string &_value) -> std::unique_ptr<issend_request<std::string>>;
    template <class T>
    auto irsend(const T &_value) -> std::unique_ptr<irsend_request<T>>;
#pragma endregion
#pragma region rsend
    template <class T>
    auto rsend(const T &_value) -> void;
    template <class T>
    auto rsend(const std::vector<T> &_value) -> void;
    auto rsend(const char _value) -> void;
    auto rsend(const char *_value) -> void;
    auto rsend(const std::string &_value) -> void;
#pragma endregion
#pragma region irsend
    template <class T>
    auto irsend(const std::vector<T> &_value) -> std::unique_ptr<irsend_request<std::vector<T>>>;
    auto irsend(const char _value) -> std::unique_ptr<irsend_request<std::string>>;
    auto irsend(const char *_value) -> std::unique_ptr<irsend_request<std::string>>;
    auto irsend(const std::string &_value) -> std::unique_ptr<irsend_request<std::string>>;
#pragma endregion
#pragma region gather
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
#pragma endregion
#pragma region igather
    template <class T>
    auto igather(const T &_value, std::vector<T> &_bucket) -> std::unique_ptr<igather_request<std::vector<T>>>;
    template <class T>
    auto igather(const std::vector<T> &_value, std::vector<T> &_bucket) -> std::unique_ptr<igather_request<std::vector<T>>>;
    auto igather(const char _value, std::string &_bucket) -> std::unique_ptr<igather_request<std::string>>;
    auto igather(const char *_value, std::string &_bucket) -> std::unique_ptr<igather_request<std::string>>;
    auto igather(const std::string &_value, std::string &_bucket) -> std::unique_ptr<igather_request<std::string>>;
    template <class T>
    auto igather(const T &_value) -> std::unique_ptr<igather_reply<std::vector<T>>>;
    template <class T>
    auto igather(const std::vector<T> &_value) -> std::unique_ptr<igather_reply<std::vector<T>>>;
    auto igather(const char _value) -> std::unique_ptr<igather_reply<std::string>>;
    auto igather(const char *_value) -> std::unique_ptr<igather_reply<std::string>>;
    auto igather(const std::string &_value) -> std::unique_ptr<igather_reply<std::string>>;
#pragma endregion
#pragma region reduce
    template <class T>
    auto reduce(const T &_value, T &_bucket, op *_operation) -> void;
    template <class T>
    auto reduce(const std::vector<T> &_value, std::vector<T> &_bucket, op *_operation) -> void;
    auto reduce(const char _value, std::string &_bucket, op *_operation) -> void;
    auto reduce(const char *_value, std::string &_bucket, op *_operation) -> void;
    auto reduce(const std::string &_value, std::string &_bucket, op *_operation) -> void;
    template <class T>
    auto reduce(const T &_value, op *_operation) -> T;
    template <class T>
    auto reduce(const std::vector<T> &_value, op *_operation) -> std::vector<T>;
    auto reduce(const char _value, op *_operation) -> std::string;
    auto reduce(const char *_value, op *_operation) -> std::string;
    auto reduce(const std::string &_value, op *_operation) -> std::string;

    template <class T, class Op>
    auto reduce(const T &_value, T &_bucket, Op _operation) -> void;
    template <class T, class Op>
    auto reduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> void;
    template <class Op>
    auto reduce(const char _value, std::string &_bucket, Op _operation) -> void;
    template <class Op>
    auto reduce(const char *_value, std::string &_bucket, Op _operation) -> void;
    template <class Op>
    auto reduce(const std::string &_value, std::string &_bucket, Op _operation) -> void;
    template <class T, class Op>
    auto reduce(const T &_value, Op _operation) -> T;
    template <class T, class Op>
    auto reduce(const std::vector<T> &_value, Op _operation) -> std::vector<T>;
    template <class Op>
    auto reduce(const char _value, Op _operation) -> std::string;
    template <class Op>
    auto reduce(const char *_value, Op _operation) -> std::string;
    template <class Op>
    auto reduce(const std::string &_value, Op _operation) -> std::string;
#pragma endregion
#pragma region ireduce
    template <class T>
    auto ireduce(const T &_value, T &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_request<T>>;
    template <class T>
    auto ireduce(const std::vector<T> &_value, std::vector<T> &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_request<std::vector<T>>>;
    auto ireduce(const char _value, std::string &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_request<std::string>>;
    auto ireduce(const char *_value, std::string &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_request<std::string>>;
    auto ireduce(const std::string &_value, std::string &_bucket, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_request<std::string>>;
    template <class T>
    auto ireduce(const T &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_reply<T>>;
    template <class T>
    auto ireduce(const std::vector<T> &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_reply<std::vector<T>>>;
    auto ireduce(const char _value, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_reply<std::string>>;
    auto ireduce(const char *_value, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_reply<std::string>>;
    auto ireduce(const std::string &_value, std::shared_ptr<op> _operation) -> std::unique_ptr<ireduce_reply<std::string>>;

    template <class T, class Op>
    auto ireduce(const T &_value, T &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<T>>;
    template <class T, class Op>
    auto ireduce(const std::vector<T> &_value, std::vector<T> &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<std::vector<T>>>;
    template <class Op>
    auto ireduce(const char _value, std::string &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<std::string>>;
    template <class Op>
    auto ireduce(const char *_value, std::string &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<std::string>>;
    template <class Op>
    auto ireduce(const std::string &_value, std::string &_bucket, Op _operation) -> std::unique_ptr<ireduce_request<std::string>>;
    template <class T, class Op>
    auto ireduce(const T &_value, Op _operation) -> std::unique_ptr<ireduce_reply<T>>;
    template <class T, class Op>
    auto ireduce(const std::vector<T> &_value, Op _operation) -> std::unique_ptr<ireduce_reply<std::vector<T>>>;
    template <class Op>
    auto ireduce(const char _value, Op _operation) -> std::unique_ptr<ireduce_reply<std::string>>;
    template <class Op>
    auto ireduce(const char *_value, Op _operation) -> std::unique_ptr<ireduce_reply<std::string>>;
    template <class Op>
    auto ireduce(const std::string &_value, Op _operation) -> std::unique_ptr<ireduce_reply<std::string>>;
#pragma endregion
};
#pragma endregion
} // namespace mpi

//finally include the definitions
#include <mpiwrap/mpi.tpp>
//include the operator overloads as well
#include <mpiwrap/impl/ops.h>