#pragma once
#include <functional>
#include <mpiwrap/mpi.h>

namespace mpi
{
namespace impl
{
//maximum
struct max
{
};
//minimum
struct min
{
};
//sum
struct sum
{
};
//product
struct prod
{
};
//logical and
struct land
{
};
//bitwise and
struct band
{
};
//logical or
struct lor
{
};
//bitwise or
struct bor
{
};
//logical xor
struct lxor
{
};
//bitwise xor
struct bxor
{
};
//maximum value and location
struct maxloc
{
};
//minimum value and location
struct minloc
{
};
//do nothing --> f(a,b) = a
struct no_op
{
};
//replace --> f(a,b) = b
struct replace
{
};
} // namespace impl

//shortcuts
constexpr auto max = impl::max{};
constexpr auto min = impl::min{};
constexpr auto sum = impl::sum{};
constexpr auto prod = impl::prod{};
constexpr auto land = impl::land{};
constexpr auto band = impl::band{};
constexpr auto lor = impl::lor{};
constexpr auto bor = impl::bor{};
constexpr auto lxor = impl::lxor{};
constexpr auto bxor = impl::bxor{};
constexpr auto maxloc = impl::maxloc{};
constexpr auto minloc = impl::minloc{};
constexpr auto no_op = impl::no_op{};
constexpr auto replace = impl::replace{};

//maximum
template <class T>
class op_proxy<T, impl::max> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_MAX, _commute) {}
};
//minimum
template <class T>
class op_proxy<T, impl::min> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_MIN, _commute) {}
};
//sum
template <class T>
class op_proxy<T, impl::sum> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_SUM, _commute) {}
};
//std::plus overload for sum
template <class T>
class op_proxy<T, std::plus<T>> : public op_proxy<T, impl::sum>
{
    using op_proxy<T, impl::sum>::op_proxy;
};
//product
template <class T>
class op_proxy<T, impl::prod> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_PROD, _commute) {}
};
//std::multiplies overload for product
template <class T>
class op_proxy<T, std::multiplies<T>> : public op_proxy<T, impl::prod>
{
    using op_proxy<T, impl::prod>::op_proxy;
};
//logical and
template <class T>
class op_proxy<T, impl::land> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_LAND, _commute) {}
};
//std::logical_and overload for logical and
template <class T>
class op_proxy<T, std::logical_and<T>> : public op_proxy<T, impl::land>
{
    using op_proxy<T, impl::land>::op_proxy;
};
//bitwise and
template <class T>
class op_proxy<T, impl::band> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_BAND, _commute) {}
};
//std::bit_and overload for bitwise and
template <class T>
class op_proxy<T, std::bit_and<T>> : public op_proxy<T, impl::band>
{
    using op_proxy<T, impl::band>::op_proxy;
};
//logical or
template <class T>
class op_proxy<T, impl::lor> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_LOR, _commute) {}
};
//std::logical_or overload for logical or
template <class T>
class op_proxy<T, std::logical_or<T>> : public op_proxy<T, impl::lor>
{
    using op_proxy<T, impl::lor>::op_proxy;
};
//bitwise or
template <class T>
class op_proxy<T, impl::bor> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_BOR, _commute) {}
};
//std::bit_or overload for bitwise or
template <class T>
class op_proxy<T, std::bit_or<T>> : public op_proxy<T, impl::bor>
{
    using op_proxy<T, impl::bor>::op_proxy;
};
//logical xor
template <class T>
class op_proxy<T, impl::lxor> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_LXOR, _commute) {}
};
//bitwise xor
template <class T>
class op_proxy<T, impl::bxor> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_BXOR, _commute) {}
};
//std::bit_xor overload for bitwise xor
template <class T>
class op_proxy<T, std::bit_xor<T>> : public op_proxy<T, impl::bxor>
{
    using op_proxy<T, impl::bxor>::op_proxy;
};
//maximum value and location
template <class T>
class op_proxy<T, impl::maxloc> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_MAXLOC, _commute) {}
};
//minimum value and location
template <class T>
class op_proxy<T, impl::minloc> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_MINLOC, _commute) {}
};
//do nothing
template <class T>
class op_proxy<T, impl::no_op> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_NO_OP, _commute) {}
};
//replace
template <class T>
class op_proxy<T, impl::replace> : public op
{
    using op::op;

public:
    op_proxy(const bool _commute) : op(MPI_REPLACE, _commute) {}
};
} // namespace mpi