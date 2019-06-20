#include <mpiwrap/scheduler.h>

#ifdef __linux__
#include <unistd.h>
#endif
#ifdef _WIN32
#include <windows.h>
#endif

namespace mpi
{
namespace impl
{
#pragma region custom sleep function
auto sleep_for_ms(int time) -> void
{
#ifdef __linux__
    usleep(time * 1000); // usleep takes sleep time in us (1 millionth of a second)
#endif
#ifdef _WIN32
    Sleep(time);
#endif
}
#pragma endregion
} // namespace impl
} // namespace mpi