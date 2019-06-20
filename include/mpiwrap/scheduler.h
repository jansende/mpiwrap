#pragma once
#include <mpiwrap/mpi.h>

namespace mpi
{
namespace impl
{
#pragma region custom sleep function
auto sleep_for_ms(int time) -> void;
#pragma endregion
} // namespace impl
#pragma region custom task
class task
{
public:
    virtual auto prepare(communicator *_communicator, bool _is_worker) -> void {}
    virtual auto execute_subtask(receiver *_source, sender *_dest) -> void = 0;
    virtual auto clean(communicator *_communicator, bool _is_worker) -> void {}
    virtual ~task() {}
};
#pragma endregion
#pragma region custom scheduler
template <class... Tasks>
class scheduler
{
protected:
    std::unique_ptr<communicator> _communicator;
    const size_t _rank;
    const size_t _size;
    const size_t _sleep_in_ms;

public:
    scheduler(std::unique_ptr<communicator> &&_communicator, size_t _sleep_in_ms = 500);
    ~scheduler();

    template <class Task>
    auto execute(Task _task) -> decltype(_task.get_result());
    auto run() -> void;

    auto is_manager() const -> bool;
    auto is_worker() const -> bool;
};
#pragma endregion
} // namespace mpi

//finally include the definitions
#include <mpiwrap/scheduler.tpp>