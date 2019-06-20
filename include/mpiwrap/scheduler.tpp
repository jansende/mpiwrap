#pragma once
#include <type_traits>

namespace mpi
{
namespace impl
{
#pragma region template helpers
template <class Task, size_t Pos, class... Tasks>
struct find_position_impl
{
};
template <class Task, size_t Pos>
struct find_position_impl<Task, Pos>
{
    static_assert(!std::is_same<Task, Task>::value, "ERROR");
};
template <class Task, size_t Pos, class First, class... Rest>
struct find_position_impl<Task, Pos, First, Rest...> : find_position_impl<Task, Pos + 1, Rest...>
{
};
template <class Task, size_t Pos, class... Rest>
struct find_position_impl<Task, Pos, Task, Rest...>
{
    static constexpr auto value = Pos;
};
template <class Task, class... Tasks>
constexpr auto find_position = find_position_impl<Task, 0, Tasks...>::value;

template <class Base, size_t Index, class... Tasks>
struct create_task_impl
{
};
template <class Base, size_t Index, class First, class... Rest>
struct create_task_impl<Base, Index, First, Rest...>
{
    static constexpr auto create(size_t index) -> std::unique_ptr<Base>
    {
        if (index == Index - 1)
            return std::make_unique<First>();
        else
            return create_task_impl<Base, Index - 1, Rest...>::create(index);
    }
};
template <class Base, size_t Index>
struct create_task_impl<Base, Index>
{
    static constexpr auto create(size_t index) -> std::unique_ptr<Base>
    {
        assert(false);
        return std::unique_ptr<Base>{nullptr};
    }
};

template <class Base, class... Tasks>
constexpr auto create_task(size_t index)
{
    return create_task_impl<Base, sizeof...(Tasks), Tasks...>::create(sizeof...(Tasks) - 1 - index);
}
#pragma endregion
} // namespace impl
#pragma region custom scheduler
template <class... Tasks>
scheduler<Tasks...>::scheduler(std::unique_ptr<communicator> &&_communicator, size_t _sleep_in_ms) : _communicator(std::move(_communicator)), _rank(this->_communicator->rank()), _size(this->_communicator->size()), _sleep_in_ms(_sleep_in_ms)
{
}
template <class... Tasks>
scheduler<Tasks...>::~scheduler()
{
    if (is_manager())
    {
        //shutdown workers
        for (auto _worker = 1; _worker < _size; ++_worker)
            //send shutdown signal
            _communicator->dest(_worker)->isend(true)->wait();
    }
}

template <class... Tasks>
template <class Task>
auto scheduler<Tasks...>::execute(Task _task) -> decltype(_task.get_result())
{
    //preparations
    for (auto _worker = size_t{1}; _worker < _size; ++_worker)
    {
        //send no shutdown signal
        _communicator->dest(_worker)->isend(false);
        //send task id
        _communicator->dest(_worker)->isend(impl::find_position<Task, Tasks...>);
        //prepare task on workers
        _task.prepare(_communicator.get(), is_worker());
    }

    class promise
    {
    private:
        bool _is_active = false;
        size_t _subtask_id = 0;
        size_t _worker_id = 0;
        using Callback = std::remove_cv_t<std::remove_reference_t<decltype(_task.direct_subtask(nullptr, nullptr, 0))>>;
        Callback _callback;

    public:
        promise() = default;
        promise(size_t _worker_id) : _worker_id(_worker_id) {}
        auto activate(size_t _subtask_id, Callback &&_callback)
        {
            this->_subtask_id = _subtask_id;
            this->_callback = std::move(_callback);
            this->_is_active = true;
        }
        auto deactivate() -> void
        {
            this->_is_active = false;
        }
        auto id() const
        {
            return _worker_id;
        }
        auto subtask_id() const
        {
            return _subtask_id;
        }
        auto is_active() const -> bool
        {
            return _is_active;
        }
        auto is_inactive() const -> bool
        {
            return !_is_active;
        }
        auto is_finished() const -> bool
        {
            if (is_active())
                return test();
            else
                return false;
        }
        auto test() const -> bool
        {
            if (is_active())
                return _callback->test();
            else
                return false;
        }
        auto wait() const
        {
            if (is_active())
                return _callback->wait();
        }
        auto get() const
        {
            if (is_active())
                return _callback->get();
            else
                throw;
        }
    };
    auto workers = std::vector<promise>{};
    for (auto id = size_t{1}; id < _size; ++id)
        workers.emplace_back(promise{id});

    while (!_task.is_finished())
    {
        //give task to worker if one is available
        for (auto &&worker : workers)
        {
            if (worker.is_inactive())
            {
                //send not finished signal
                _communicator->dest(worker.id())->isend(false)->wait();
                //direct_subtask task
                worker.activate(_task.subtask_id(), std::move(_task.direct_subtask(_communicator->source(worker.id()).get(), _communicator->dest(worker.id()).get(), _task.subtask_id())));
                _task.advance_to_next_subtask();
                if (_task.is_finished())
                    break;
            }
        }
        //is anyone already finished?
        for (auto &&worker : workers)
            if (worker.is_finished())
            {
                //save result
                _task.store_subtask_result(worker.subtask_id(), worker.get());
                worker.deactivate();
            }
        //sleep a bit to conserve cpu power
        impl::sleep_for_ms(_sleep_in_ms);
    }
    //wait for all workers to finish
    for (auto is_finished = false; !is_finished;)
    {
        is_finished = true;
        for (auto &&worker : workers)
            if (worker.is_finished())
            {
                //save result
                _task.store_subtask_result(worker.subtask_id(), worker.get());
                worker.deactivate();
            }
            else if (worker.is_active())
            {
                is_finished = false;
            }
        //sleep a bit to conserve cpu power
        impl::sleep_for_ms(_sleep_in_ms);
    }

    //cleanup
    for (auto _worker = 1; _worker < _size; ++_worker)
        //send finish task signal
        _communicator->dest(_worker)->isend(true);
    _task.clean(_communicator.get(), is_worker());

    //finish
    return _task.get_result();
}
template <class... Tasks>
auto scheduler<Tasks...>::run() -> void
{
    if (is_worker())
    {
        while (true)
        {
            //keep running?
            auto _shutdown = _communicator->source(0)->irecv<bool>()->get();
            if (_shutdown)
            {
                break;
            }
            //YES!
            else
            {
                //get task
                auto _task = impl::create_task<task, Tasks...>(_communicator->source(0)->irecv<size_t>()->get());
                //prepare task
                _task->prepare(_communicator.get(), is_worker());
                //execute_subtask task
                while (true)
                {
                    //keep working?
                    auto _finish_task = _communicator->source(0)->irecv<bool>()->get();
                    if (_finish_task)
                    {
                        break;
                    }
                    //YES!
                    else
                    {
                        _task->execute_subtask(_communicator->source(0).get(), _communicator->dest(0).get());
                    }
                }
                //cleanup
                _task->clean(_communicator.get(), is_worker());
            }
        }
    }
}

template <class... Tasks>
auto scheduler<Tasks...>::is_manager() const -> bool
{
    return this->_rank == 0;
}
template <class... Tasks>
auto scheduler<Tasks...>::is_worker() const -> bool
{
    return this->_rank != 0;
}
#pragma endregion
} // namespace mpi