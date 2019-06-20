#include <mpiwrap/scheduler.h>
#include <iostream>
#include <random>

#ifdef __linux__
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

void sleep_for_ms(int time)
{
#ifdef __linux__
    usleep(time * 1000); // usleep takes sleep time in us (1 millionth of a second)
#endif
#ifdef _WIN32
    Sleep(time);
#endif
}

class my_task : public mpi::task
{
private:
    std::vector<int> _tasks;
    std::vector<int> _result;
    size_t _subtask_id = 0;

public:
    my_task() = default;
    my_task(std::vector<int> _tasks) : _tasks(_tasks), _result(std::vector<int>(_tasks.size())) {}

    auto subtask_id() const
    {
        return _subtask_id;
    }
    auto store_subtask_result(size_t _subtask_id, int result) -> void
    {
        _result[_subtask_id] = result;
    }
    auto get_result() const -> std::vector<int>
    {
        return _result;
    }
    auto is_finished() const -> bool
    {
        return _subtask_id >= _tasks.size();
    }
    auto advance_to_next_subtask() -> void
    {
        ++_subtask_id;
    }

    virtual auto execute_subtask(mpi::receiver *_source, mpi::sender *_dest) -> void
    {
        //receive data from manager
        auto data = _source->irecv<int>()->get();
        //do calculations
        //we actually don't do any heavy calculations, just sleeping...
        sleep_for_ms(data);
        //return result to manager
        _dest->isend(data + 42)->wait();
    }
    auto direct_subtask(mpi::receiver *_source, mpi::sender *_dest, size_t _id) -> std::unique_ptr<mpi::irecv_reply<int>>
    {
        //send task data to worker
        _dest->isend(_tasks[_id])->wait();
        //return result expectation
        return _source->irecv<int>();
    }
};

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    //create scheduler
    mpi::scheduler<my_task> scheduler{mpi::comm("world")};
    //activate on workers
    scheduler.run();
    if (scheduler.is_manager())
    {

        //create task
        constexpr auto problem_size = 37;
        std::cout << "Generating task...";
        auto tasks = std::vector<int>(problem_size);
        {
            std::random_device rd;
            std::uniform_int_distribution<int> dist(4, 9);
            for (auto &&task : tasks)
                task = dist(rd) * 1000; //wait time in ms
        }
        std::cout << "Done\n";
        //run task
        std::cout << "Executing task..." << std::flush;
        auto result = scheduler.execute(my_task{tasks});
        //check result
        std::cout << "Done\n"
                  << "Checking results...";
        {
            auto has_erroneous_result = false;
            for (auto task_id = size_t{0}; task_id < tasks.size(); ++task_id)
                if (tasks[task_id] != result[task_id] - 42)
                {
                    // std::cout << "at: " << task_id << " got: " << result[task_id] << ", expected: " << tasks[task_id] + 42 << '\n';
                    has_erroneous_result = true;
                    break;
                }

            if (has_erroneous_result)
                std::cout << "\nError in results. Try again!\n";
            else
                std::cout << "Done\n";
        }
    }
}