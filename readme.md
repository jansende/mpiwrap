# Intro
MPI (*Message Passing Interface*) is the standard for parallel, scientific programming. For whatever reason, the MPI commitee decided to drop C++ bindings a while ago. C bindings still exist, but being C, do not fit well into modern C++ programmes. **mpiwrap** seeks to solve that problem by providing a convenient C++ wrapper around the C bindings. Thus, if you are an avid C++ programmer who is disgusted by the horrible C interface of MPI, this is the library for you!

**mpiwrap** provides overload for seamless usage with all standard C++ types, plus `std::vector` of these types. However, it is easy to provide user overloads for custom types. When useful, **mpiwrap** inserts additional checks in the form of `asserts` in order to prevent size errors when using vectors. Furthermore, **mpiwrap** tries to generalize all MPI functions, allowing the user to utilize a wider range of use-cases hassle-free.

# Quickstart
## Install using cmake
The best way to install **mpiwrap** is to clone the repository and add the `add_subdirectory` and `target_link` library command to your `CMakeLists.txt`. This will automatically include MPI to your project (you need to install it separetly though). A simple project file might look like this:
```cmake
cmake_minimum_required(VERSION 3.1)

#Define the project.
project(hello_mpi)
add_executable(hello_mpi main.cpp)

#Add the mpiwrap repository.
add_subdirectory(mpiwrap)
#Add the library to the project.
target_link_libraries(hello_mpi PRIVATE mpiwrap)
```

## Hello MPI
You can then try MPI with a little sample programme:

```c++
#include <mpiwrap/mpi.h>

int main(int argc, char **argv)
{
    mpi::mpi init{argc, argv};

    auto world_size = mpi::comm("world")->size();
    auto world_rank = mpi::comm("world")->rank();

    auto processor_name = mpi::processor_name();
    std::cout << "Hello world from processor " << processor_name << ", rank " << world_rank << " out of " << world_size << " processors.\n";

    return 0;
}
```

# Functions
This is a list of the currently implemented MPI functions, and their usage with the **mpiwrap** wrapper. Values marked with bracket mean, that you have to substitute reasonable values there. For example: `[COMM]` is a `MPI_Comm` value, `[VALUE]` and `[BUCKET]` can be either single variables or `std::vectors`, and `[OP]` is a `MPI_Op` value. `[CHUNKSIZE]` and `[RANK]` are both positive integer values.

| MPI Function                   | Implemented        | Version | Usage with the **mpiwrap** wrapper                                     |
|:-------------------------------|:------------------:|:-------:|:-----------------------------------------------------------------------|
| MPI_Abort                      | :x:                |         |                                                                        |
| MPI_Accumulate                 | :x:                |         |                                                                        |
| MPI_Add_error_class            | :x:                |         |                                                                        |
| MPI_Add_error_code             | :x:                |         |                                                                        |
| MPI_Add_error_string           | :x:                |         |                                                                        |
| MPI_Address                    | :x:                |         |                                                                        |
| MPI_Aint_add                   | :x:                |         |                                                                        |
| MPI_Aint_diff                  | :x:                |         |                                                                        |
| MPI_Allgather                  | :heavy_check_mark: |         | `mpi::comm([COMM])->allgather([VALUE], [BUCKET])`                      |
| MPI_Allgatherv                 | :x:                |         |                                                                        |
| MPI_Alloc_mem                  | :x:                |         |                                                                        |
| MPI_Allreduce                  | :heavy_check_mark: |         | `mpi::comm([COMM])->allreduce([VALUE], [BUCKET], [OP])`                |
| MPI_Alltoall                   | :heavy_check_mark: |         | `mpi::comm([COMM])->allreduce([VALUE], [BUCKET], [CHUNKSIZE])`         |
| MPI_Alltoallv                  | :x:                |         |                                                                        |
| MPI_Alltoallw                  | :x:                |         |                                                                        |
| MPI_Attr_delete                | :x:                |         |                                                                        |
| MPI_Attr_get                   | :x:                |         |                                                                        |
| MPI_Attr_put                   | :x:                |         |                                                                        |
| MPI_Barrier                    | :heavy_check_mark: |         | `mpi::comm([COMM])->barrier()`                                         |
| MPI_Bcast                      | :heavy_check_mark: |         | `mpi::comm([COMM])->source([RANK])->bcast[VALUE])`                     |
| MPI_Bsend                      | :x:                |         | Will not be implemented because raw memory management is required.     |
| MPI_Bsend_init                 | :x:                |         | Will not be implemented because raw memory management is required.     |
| MPI_Buffer_attach              | :x:                |         | Will not be implemented because raw memory management is required.     |
| MPI_Buffer_detach              | :x:                |         | Will not be implemented because raw memory management is required.     |
| MPI_Cancel                     | :heavy_check_mark: |         | `.cancel()` on the `mpi::request` object.                              |
| MPI_Cart_coords                | :x:                |         |                                                                        |
| MPI_Cart_create                | :x:                |         |                                                                        |
| MPI_Cart_get                   | :x:                |         |                                                                        |
| MPI_Cart_map                   | :x:                |         |                                                                        |
| MPI_Cart_rank                  | :x:                |         |                                                                        |
| MPI_Cart_shift                 | :x:                |         |                                                                        |
| MPI_Cart_sub                   | :x:                |         |                                                                        |
| MPI_Cartdim_get                | :x:                |         |                                                                        |
| MPI_Close_port                 | :x:                |         |                                                                        |
| MPI_Comm_accept                | :x:                |         |                                                                        |
| MPI_Comm_call_errhandler       | :x:                |         |                                                                        |
| MPI_Comm_compare               | :heavy_check_mark: |         | `mpi::compare([COMM],[COMM])`                                          |
| MPI_Comm_connect               | :x:                |         |                                                                        |
| MPI_Comm_create                | :x:                |         |                                                                        |
| MPI_Comm_create_errhandler     | :x:                |         |                                                                        |
| MPI_Comm_create_group          | :x:                |         |                                                                        |
| MPI_Comm_create_keyval         | :x:                |         |                                                                        |
| MPI_Comm_delete_attr           | :x:                |         |                                                                        |
| MPI_Comm_disconnect            | :x:                |         |                                                                        |
| MPI_Comm_dup                   | :x:                |         |                                                                        |
| MPI_Comm_dup_with_info         | :x:                |         |                                                                        |
| MPI_Comm_free                  | :x:                |         |                                                                        |
| MPI_Comm_free_keyval           | :x:                |         |                                                                        |
| MPI_Comm_get_attr              | :x:                |         |                                                                        |
| MPI_Comm_get_errhandler        | :x:                |         |                                                                        |
| MPI_Comm_get_info              | :x:                |         |                                                                        |
| MPI_Comm_get_name              | :heavy_check_mark: |         | `mpi::comm([COMM])->name()`                                            |
| MPI_Comm_get_parent            | :x:                |         |                                                                        |
| MPI_Comm_group                 | :x:                |         |                                                                        |
| MPI_Comm_idup                  | :x:                |         |                                                                        |
| MPI_Comm_join                  | :x:                |         |                                                                        |
| MPI_Comm_rank                  | :heavy_check_mark: |         | `mpi::comm([COMM])->rank()`                                            |
| MPI_Comm_remote_group          | :x:                |         |                                                                        |
| MPI_Comm_remote_size           | :x:                |         |                                                                        |
| MPI_Comm_set_attr              | :x:                |         |                                                                        |
| MPI_Comm_set_errhandler        | :x:                |         |                                                                        |
| MPI_Comm_set_info              | :x:                |         |                                                                        |
| MPI_Comm_set_name              | :x:                |         |                                                                        |
| MPI_Comm_size                  | :heavy_check_mark: |         | `mpi::comm([COMM])->size()`                                            |
| MPI_Comm_spawn                 | :x:                |         |                                                                        |
| MPI_Comm_spawn_multiple        | :x:                |         |                                                                        |
| MPI_Comm_split                 | :x:                |         |                                                                        |
| MPI_Comm_split_type            | :x:                |         |                                                                        |
| MPI_Comm_test_inter            | :x:                |         |                                                                        |
| MPI_Compare_and_swap           | :x:                |         |                                                                        |
| MPI_Dims_create                | :x:                |         |                                                                        |
| MPI_Dist_graph_create          | :x:                |         |                                                                        |
| MPI_Dist_graph_create_adjacent | :x:                |         |                                                                        |
| MPI_Dist_graph_neighbors       | :x:                |         |                                                                        |
| MPI_Dist_graph_neighbors_count | :x:                |         |                                                                        |
| MPI_Errhandler_create          | :x:                |         |                                                                        |
| MPI_Errhandler_free            | :x:                |         |                                                                        |
| MPI_Errhandler_get             | :x:                |         |                                                                        |
| MPI_Errhandler_set             | :x:                |         |                                                                        |
| MPI_Error_class                | :x:                |         |                                                                        |
| MPI_Error_string               | :x:                |         |                                                                        |
| MPI_Exscan                     | :x:                |         |                                                                        |
| MPI_Fetch_and_op               | :x:                |         |                                                                        |
| MPI_File_c2f                   | :x:                |         |                                                                        |
| MPI_File_call_errhandler       | :x:                |         |                                                                        |
| MPI_File_close                 | :x:                |         |                                                                        |
| MPI_File_create_errhandler     | :x:                |         |                                                                        |
| MPI_File_delete                | :x:                |         |                                                                        |
| MPI_File_f2c                   | :x:                |         |                                                                        |
| MPI_File_get_amode             | :x:                |         |                                                                        |
| MPI_File_get_atomicity         | :x:                |         |                                                                        |
| MPI_File_get_byte_offset       | :x:                |         |                                                                        |
| MPI_File_get_errhandler        | :x:                |         |                                                                        |
| MPI_File_get_group             | :x:                |         |                                                                        |
| MPI_File_get_info              | :x:                |         |                                                                        |
| MPI_File_get_position          | :x:                |         |                                                                        |
| MPI_File_get_position_shared   | :x:                |         |                                                                        |
| MPI_File_get_size              | :x:                |         |                                                                        |
| MPI_File_get_type_extent       | :x:                |         |                                                                        |
| MPI_File_get_view              | :x:                |         |                                                                        |
| MPI_File_iread                 | :x:                |         |                                                                        |
| MPI_File_iread_all             | :x:                |         |                                                                        |
| MPI_File_iread_at              | :x:                |         |                                                                        |
| MPI_File_iread_at_all          | :x:                |         |                                                                        |
| MPI_File_iread_shared          | :x:                |         |                                                                        |
| MPI_File_iwrite                | :x:                |         |                                                                        |
| MPI_File_iwrite_all            | :x:                |         |                                                                        |
| MPI_File_iwrite_at             | :x:                |         |                                                                        |
| MPI_File_iwrite_at_all         | :x:                |         |                                                                        |
| MPI_File_iwrite_shared         | :x:                |         |                                                                        |
| MPI_File_open                  | :x:                |         |                                                                        |
| MPI_File_preallocate           | :x:                |         |                                                                        |
| MPI_File_read                  | :x:                |         |                                                                        |
| MPI_File_read_all              | :x:                |         |                                                                        |
| MPI_File_read_all_begin        | :x:                |         |                                                                        |
| MPI_File_read_all_end          | :x:                |         |                                                                        |
| MPI_File_read_at               | :x:                |         |                                                                        |
| MPI_File_read_at_all           | :x:                |         |                                                                        |
| MPI_File_read_at_all_begin     | :x:                |         |                                                                        |
| MPI_File_read_at_all_end       | :x:                |         |                                                                        |
| MPI_File_read_ordered          | :x:                |         |                                                                        |
| MPI_File_read_ordered_begin    | :x:                |         |                                                                        |
| MPI_File_read_ordered_end      | :x:                |         |                                                                        |
| MPI_File_read_shared           | :x:                |         |                                                                        |
| MPI_File_seek                  | :x:                |         |                                                                        |
| MPI_File_seek_shared           | :x:                |         |                                                                        |
| MPI_File_set_atomicity         | :x:                |         |                                                                        |
| MPI_File_set_errhandler        | :x:                |         |                                                                        |
| MPI_File_set_info              | :x:                |         |                                                                        |
| MPI_File_set_size              | :x:                |         |                                                                        |
| MPI_File_set_view              | :x:                |         |                                                                        |
| MPI_File_sync                  | :x:                |         |                                                                        |
| MPI_File_write                 | :x:                |         |                                                                        |
| MPI_File_write_all             | :x:                |         |                                                                        |
| MPI_File_write_all_begin       | :x:                |         |                                                                        |
| MPI_File_write_all_end         | :x:                |         |                                                                        |
| MPI_File_write_at              | :x:                |         |                                                                        |
| MPI_File_write_at_all          | :x:                |         |                                                                        |
| MPI_File_write_at_all_begin    | :x:                |         |                                                                        |
| MPI_File_write_at_all_end      | :x:                |         |                                                                        |
| MPI_File_write_ordered         | :x:                |         |                                                                        |
| MPI_File_write_ordered_begin   | :x:                |         |                                                                        |
| MPI_File_write_ordered_end     | :x:                |         |                                                                        |
| MPI_File_write_shared          | :x:                |         |                                                                        |
| MPI_Finalize                   | :heavy_check_mark: |         | MPI_Finalize is automatically called when `mpi::mpi` is destroyed.     |
| MPI_Finalized                  | :heavy_check_mark: |         | `mpi::finalized()`                                                     |
| MPI_Free_mem                   | :x:                |         |                                                                        |
| MPI_Gather                     | :heavy_check_mark: |         | `mpi::comm([COMM])->dest([RANK])->gather([VALUE], [BUCKET])`           |
| MPI_Gatherv                    | :x:                |         |                                                                        |
| MPI_Get                        | :x:                |         |                                                                        |
| MPI_Get_accumulate             | :x:                |         |                                                                        |
| MPI_Get_address                | :x:                |         |                                                                        |
| MPI_Get_count                  | :x:                |         |                                                                        |
| MPI_Get_elements               | :x:                |         |                                                                        |
| MPI_Get_elements_x             | :x:                |         |                                                                        |
| MPI_Get_library_version        | :x:                |         |                                                                        |
| MPI_Get_processor_name         | :heavy_check_mark: |         | `mpi::processor_name()`                                                |
| MPI_Get_version                | :heavy_check_mark: |         | `mpi::version()` with members: `.version()` and `.subversion()`        |
| MPI_Graph_create               | :x:                |         |                                                                        |
| MPI_Graph_get                  | :x:                |         |                                                                        |
| MPI_Graph_map                  | :x:                |         |                                                                        |
| MPI_Graph_neighbors            | :x:                |         |                                                                        |
| MPI_Graph_neighbors_count      | :x:                |         |                                                                        |
| MPI_Graphdims_get              | :x:                |         |                                                                        |
| MPI_Grequest_complete          | :x:                |         |                                                                        |
| MPI_Grequest_start             | :x:                |         |                                                                        |
| MPI_Group_compare              | :x:                |         |                                                                        |
| MPI_Group_difference           | :x:                |         |                                                                        |
| MPI_Group_excl                 | :x:                |         |                                                                        |
| MPI_Group_free                 | :x:                |         |                                                                        |
| MPI_Group_incl                 | :x:                |         |                                                                        |
| MPI_Group_intersection         | :x:                |         |                                                                        |
| MPI_Group_range_excl           | :x:                |         |                                                                        |
| MPI_Group_range_incl           | :x:                |         |                                                                        |
| MPI_Group_rank                 | :x:                |         |                                                                        |
| MPI_Group_size                 | :x:                |         |                                                                        |
| MPI_Group_translate_ranks      | :x:                |         |                                                                        |
| MPI_Group_union                | :x:                |         |                                                                        |
| MPI_Iallgather                 | :x:                |         |                                                                        |
| MPI_Iallgatherv                | :x:                |         |                                                                        |
| MPI_Iallreduce                 | :x:                |         |                                                                        |
| MPI_Ialltoall                  | :x:                |         |                                                                        |
| MPI_Ialltoallv                 | :x:                |         |                                                                        |
| MPI_Ialltoallw                 | :x:                |         |                                                                        |
| MPI_Ibarrier                   | :heavy_check_mark: |         | `mpi::comm([COMM])->ibarrier()`                                        |
| MPI_Ibcast                     | :x:                |         |                                                                        |
| MPI_Ibsend                     | :x:                |         | Will not be implemented because raw memory management is required.     |
| MPI_Iexscan                    | :x:                |         |                                                                        |
| MPI_Igather                    | :x:                |         |                                                                        |
| MPI_Igatherv                   | :x:                |         |                                                                        |
| MPI_Improbe                    | :x:                |         |                                                                        |
| MPI_Imrecv                     | :x:                |         |                                                                        |
| MPI_Ineighbor_allgather        | :x:                |         |                                                                        |
| MPI_Ineighbor_allgatherv       | :x:                |         |                                                                        |
| MPI_Ineighbor_alltoall         | :x:                |         |                                                                        |
| MPI_Ineighbor_alltoallv        | :x:                |         |                                                                        |
| MPI_Ineighbor_alltoallw        | :x:                |         |                                                                        |
| MPI_Info_create                | :x:                |         |                                                                        |
| MPI_Info_delete                | :x:                |         |                                                                        |
| MPI_Info_dup                   | :x:                |         |                                                                        |
| MPI_Info_free                  | :x:                |         |                                                                        |
| MPI_Info_get                   | :x:                |         |                                                                        |
| MPI_Info_get_nkeys             | :x:                |         |                                                                        |
| MPI_Info_get_nthkey            | :x:                |         |                                                                        |
| MPI_Info_get_valuelen          | :x:                |         |                                                                        |
| MPI_Info_set                   | :x:                |         |                                                                        |
| MPI_Init                       | :heavy_check_mark: |         | `mpi::mpi init(argc, argv)`                                            |
| MPI_Init_thread                | :x:                |         |                                                                        |
| MPI_Initialized                | :heavy_check_mark: |         | `mpi::initialized()`                                                   |
| MPI_Intercomm_create           | :x:                |         |                                                                        |
| MPI_Intercomm_merge            | :x:                |         |                                                                        |
| MPI_Iprobe                     | :x:                |         |                                                                        |
| MPI_Irecv                      | :heavy_check_mark: |         | `mpi::comm([COMM])->source([RANK])->irecv([BUCKET])`                   |
| MPI_Ireduce                    | :x:                |         |                                                                        |
| MPI_Ireduce_scatter            | :x:                |         |                                                                        |
| MPI_Ireduce_scatter_block      | :x:                |         |                                                                        |
| MPI_Irsend                     | :heavy_check_mark: |         | `mpi::comm([COMM])->dest([RANK])->irsend([VALUE])`                     |
| MPI_Is_thread_main             | :x:                |         |                                                                        |
| MPI_Iscan                      | :x:                |         |                                                                        |
| MPI_Iscatter                   | :x:                |         |                                                                        |
| MPI_Iscatterv                  | :x:                |         |                                                                        |
| MPI_Isend                      | :heavy_check_mark: |         | `mpi::comm([COMM])->dest([RANK])->isend([VALUE])`                      |
| MPI_Issend                     | :heavy_check_mark: |         | `mpi::comm([COMM])->dest([RANK])->issend([VALUE])`                     |
| MPI_Keyval_create              | :x:                |         |                                                                        |
| MPI_Keyval_free                | :x:                |         |                                                                        |
| MPI_Lookup_name                | :x:                |         |                                                                        |
| MPI_Mprobe                     | :x:                |         |                                                                        |
| MPI_Mrecv                      | :x:                |         |                                                                        |
| MPI_Neighbor_allgather         | :x:                |         |                                                                        |
| MPI_Neighbor_allgatherv        | :x:                |         |                                                                        |
| MPI_Neighbor_alltoall          | :x:                |         |                                                                        |
| MPI_Neighbor_alltoallv         | :x:                |         |                                                                        |
| MPI_Neighbor_alltoallw         | :x:                |         |                                                                        |
| MPI_Op_commute                 | :heavy_check_mark: |         | `.commutes()` on the `mpi::op_proxy` object.                           |
| MPI_Op_create                  | :heavy_check_mark: |         | `mpi::make_op<T>([LAMBDA])` or `mpi::make_op<T>(mpi::wrap<T,[FUNC]>)`¹ |
| MPI_Op_free                    | :heavy_check_mark: |         | Automatically called by the `mpi::make_op` object (`mpi::op_proxy`).   |
| MPI_Open_port                  | :x:                |         |                                                                        |
| MPI_Pack                       | :x:                |         |                                                                        |
| MPI_Pack_external              | :x:                |         |                                                                        |
| MPI_Pack_external_size         | :x:                |         |                                                                        |
| MPI_Pack_size                  | :x:                |         |                                                                        |
| MPI_Pcontrol                   | :x:                |         |                                                                        |
| MPI_Probe                      | :x:                |         |                                                                        |
| MPI_Publish_name               | :x:                |         |                                                                        |
| MPI_Put                        | :x:                |         |                                                                        |
| MPI_Query_thread               | :x:                |         |                                                                        |
| MPI_Raccumulate                | :x:                |         |                                                                        |
| MPI_Recv                       | :heavy_check_mark: |         | `mpi::comm([COMM])->source([RANK])->recv([BUCKET])`                    |
| MPI_Recv_init                  | :x:                |         |                                                                        |
| MPI_Reduce                     | :heavy_check_mark: |         | `mpi::comm([COMM])->dest([RANK])->reduce([VALUE], [BUCKET], [OP])`     |
| MPI_Reduce_local               | :heavy_check_mark: |         | `mpi::reduce([VALUE], [BUCKET], [OP])`                                 |
| MPI_Reduce_scatter             | :x:                |         |                                                                        |
| MPI_Reduce_scatter_block       | :x:                |         |                                                                        |
| MPI_Register_datarep           | :x:                |         |                                                                        |
| MPI_Request_free               | :x:                |         |                                                                        |
| MPI_Request_get_status         | :x:                |         |                                                                        |
| MPI_Rget                       | :x:                |         |                                                                        |
| MPI_Rget_accumulate            | :x:                |         |                                                                        |
| MPI_Rput                       | :x:                |         |                                                                        |
| MPI_Rsend                      | :heavy_check_mark: |         | `mpi::comm([COMM])->dest([RANK])->rsend([VALUE])`                      |
| MPI_Rsend_init                 | :x:                |         |                                                                        |
| MPI_Scan                       | :x:                |         |                                                                        |
| MPI_Scatter                    | :heavy_check_mark: |         | `mpi::comm([COMM])->source([RANK])->scatter([VALUE], [CHUNKSIZE])`     |
| MPI_Scatterv                   | :x:                |         |                                                                        |
| MPI_Send                       | :heavy_check_mark: |         | `mpi::comm([COMM])->dest([RANK])->send([VALUE])`                       |
| MPI_Send_init                  | :x:                |         |                                                                        |
| MPI_Sendrecv                   | :x:                |         |                                                                        |
| MPI_Sendrecv_replace           | :x:                |         |                                                                        |
| MPI_Ssend                      | :heavy_check_mark: |         | `mpi::comm([COMM])->dest([RANK])->ssend([VALUE])`                      |
| MPI_Ssend_init                 | :x:                |         |                                                                        |
| MPI_Start                      | :x:                |         |                                                                        |
| MPI_Startall                   | :x:                |         |                                                                        |
| MPI_Status_set_cancelled       | :x:                |         |                                                                        |
| MPI_Status_set_elements        | :x:                |         |                                                                        |
| MPI_Status_set_elements_x      | :x:                |         |                                                                        |
| MPI_T_category_changed         | :x:                |         |                                                                        |
| MPI_T_category_get_categories  | :x:                |         |                                                                        |
| MPI_T_category_get_cvars       | :x:                |         |                                                                        |
| MPI_T_category_get_info        | :x:                |         |                                                                        |
| MPI_T_category_get_num         | :x:                |         |                                                                        |
| MPI_T_category_get_pvars       | :x:                |         |                                                                        |
| MPI_T_cvar_get_info            | :x:                |         |                                                                        |
| MPI_T_cvar_get_num             | :x:                |         |                                                                        |
| MPI_T_cvar_handle_alloc        | :x:                |         |                                                                        |
| MPI_T_cvar_handle_free         | :x:                |         |                                                                        |
| MPI_T_cvar_read                | :x:                |         |                                                                        |
| MPI_T_cvar_write               | :x:                |         |                                                                        |
| MPI_T_enum_get_info            | :x:                |         |                                                                        |
| MPI_T_enum_get_item            | :x:                |         |                                                                        |
| MPI_T_finalize                 | :x:                |         |                                                                        |
| MPI_T_init_thread              | :x:                |         |                                                                        |
| MPI_T_pvar_get_info            | :x:                |         |                                                                        |
| MPI_T_pvar_get_num             | :x:                |         |                                                                        |
| MPI_T_pvar_handle_alloc        | :x:                |         |                                                                        |
| MPI_T_pvar_handle_free         | :x:                |         |                                                                        |
| MPI_T_pvar_read                | :x:                |         |                                                                        |
| MPI_T_pvar_readreset           | :x:                |         |                                                                        |
| MPI_T_pvar_reset               | :x:                |         |                                                                        |
| MPI_T_pvar_session_create      | :x:                |         |                                                                        |
| MPI_T_pvar_session_free        | :x:                |         |                                                                        |
| MPI_T_pvar_start               | :x:                |         |                                                                        |
| MPI_T_pvar_stop                | :x:                |         |                                                                        |
| MPI_T_pvar_write               | :x:                |         |                                                                        |
| MPI_Test                       | :heavy_check_mark: |         | `.test()` on the `mpi::request` object, or `mpi::test([REQUEST])`.     |
| MPI_Test_cancelled             | :x:                |         |                                                                        |
| MPI_Testall                    | :heavy_check_mark: |         | `mpi::testall([REQUEST], ...)`, or `mpi::testall([REQUEST_VECTOR])`    |
| MPI_Testany                    | :heavy_check_mark: |         | `mpi::testany([REQUEST], ...)`, or `mpi::testany([REQUEST_VECTOR])`    |
| MPI_Testsome                   | :heavy_check_mark: |         | `mpi::testsome([REQUEST], ...)`, or `mpi::testsome([REQUEST_VECTOR])`  |
| MPI_Topo_test                  | :x:                |         |                                                                        |
| MPI_Type_commit                | :x:                |         |                                                                        |
| MPI_Type_contiguous            | :x:                |         |                                                                        |
| MPI_Type_create_darray         | :x:                |         |                                                                        |
| MPI_Type_create_hindexed       | :x:                |         |                                                                        |
| MPI_Type_create_hindexed_block | :x:                |         |                                                                        |
| MPI_Type_create_hvector        | :x:                |         |                                                                        |
| MPI_Type_create_indexed_block  | :x:                |         |                                                                        |
| MPI_Type_create_keyval         | :x:                |         |                                                                        |
| MPI_Type_create_resized        | :x:                |         |                                                                        |
| MPI_Type_create_struct         | :x:                |         |                                                                        |
| MPI_Type_create_subarray       | :x:                |         |                                                                        |
| MPI_Type_delete_attr           | :x:                |         |                                                                        |
| MPI_Type_dup                   | :x:                |         |                                                                        |
| MPI_Type_extent                | :x:                |         |                                                                        |
| MPI_Type_free                  | :x:                |         |                                                                        |
| MPI_Type_free_keyval           | :x:                |         |                                                                        |
| MPI_Type_get_attr              | :x:                |         |                                                                        |
| MPI_Type_get_contents          | :x:                |         |                                                                        |
| MPI_Type_get_envelope          | :x:                |         |                                                                        |
| MPI_Type_get_extent            | :x:                |         |                                                                        |
| MPI_Type_get_extent_x          | :x:                |         |                                                                        |
| MPI_Type_get_name              | :x:                |         |                                                                        |
| MPI_Type_get_true_extent       | :x:                |         |                                                                        |
| MPI_Type_get_true_extent_x     | :x:                |         |                                                                        |
| MPI_Type_hindexed              | :x:                |         |                                                                        |
| MPI_Type_hvector               | :x:                |         |                                                                        |
| MPI_Type_indexed               | :x:                |         |                                                                        |
| MPI_Type_lb                    | :x:                |         |                                                                        |
| MPI_Type_match_size            | :x:                |         |                                                                        |
| MPI_Type_set_attr              | :x:                |         |                                                                        |
| MPI_Type_set_name              | :x:                |         |                                                                        |
| MPI_Type_size                  | :x:                |         |                                                                        |
| MPI_Type_size_x                | :x:                |         |                                                                        |
| MPI_Type_struct                | :x:                |         |                                                                        |
| MPI_Type_ub                    | :x:                |         |                                                                        |
| MPI_Type_vector                | :x:                |         |                                                                        |
| MPI_Unpack                     | :x:                |         |                                                                        |
| MPI_Unpack_external            | :x:                |         |                                                                        |
| MPI_Unpublish_name             | :x:                |         |                                                                        |
| MPI_Wait                       | :heavy_check_mark: |         | `.wait()` on the `mpi::request` object, or `mpi::wait([REQUEST])`.     |
| MPI_Waitall                    | :heavy_check_mark: |         | `mpi::waitall([REQUEST], ...)`, or `mpi::waitall([REQUEST_VECTOR])`    |
| MPI_Waitany                    | :heavy_check_mark: |         | `mpi::waitany([REQUEST], ...)`, or `mpi::waitany([REQUEST_VECTOR])`    |
| MPI_Waitsome                   | :heavy_check_mark: |         | `mpi::waitsome([REQUEST], ...)`, or `mpi::waitsome([REQUEST_VECTOR])`  |
| MPI_Win_allocate               | :x:                |         |                                                                        |
| MPI_Win_allocate_shared        | :x:                |         |                                                                        |
| MPI_Win_attach                 | :x:                |         |                                                                        |
| MPI_Win_call_errhandler        | :x:                |         |                                                                        |
| MPI_Win_complete               | :x:                |         |                                                                        |
| MPI_Win_create                 | :x:                |         |                                                                        |
| MPI_Win_create_dynamic         | :x:                |         |                                                                        |
| MPI_Win_create_errhandler      | :x:                |         |                                                                        |
| MPI_Win_create_keyval          | :x:                |         |                                                                        |
| MPI_Win_delete_attr            | :x:                |         |                                                                        |
| MPI_Win_detach                 | :x:                |         |                                                                        |
| MPI_Win_fence                  | :x:                |         |                                                                        |
| MPI_Win_flush                  | :x:                |         |                                                                        |
| MPI_Win_flush_all              | :x:                |         |                                                                        |
| MPI_Win_flush_local            | :x:                |         |                                                                        |
| MPI_Win_flush_local_all        | :x:                |         |                                                                        |
| MPI_Win_free                   | :x:                |         |                                                                        |
| MPI_Win_free_keyval            | :x:                |         |                                                                        |
| MPI_Win_get_attr               | :x:                |         |                                                                        |
| MPI_Win_get_errhandler         | :x:                |         |                                                                        |
| MPI_Win_get_group              | :x:                |         |                                                                        |
| MPI_Win_get_info               | :x:                |         |                                                                        |
| MPI_Win_get_name               | :x:                |         |                                                                        |
| MPI_Win_lock                   | :x:                |         |                                                                        |
| MPI_Win_lock_all               | :x:                |         |                                                                        |
| MPI_Win_post                   | :x:                |         |                                                                        |
| MPI_Win_set_attr               | :x:                |         |                                                                        |
| MPI_Win_set_errhandler         | :x:                |         |                                                                        |
| MPI_Win_set_info               | :x:                |         |                                                                        |
| MPI_Win_set_name               | :x:                |         |                                                                        |
| MPI_Win_shared_query           | :x:                |         |                                                                        |
| MPI_Win_start                  | :x:                |         |                                                                        |
| MPI_Win_sync                   | :x:                |         |                                                                        |
| MPI_Win_test                   | :x:                |         |                                                                        |
| MPI_Win_unlock                 | :x:                |         |                                                                        |
| MPI_Win_unlock_all             | :x:                |         |                                                                        |
| MPI_Win_wait                   | :x:                |         |                                                                        |
| MPI_Wtick                      | :x:                |         |                                                                        |
| MPI_Wtime                      | :x:                |         |                                                                        |

¹ MPI takes a special function signature for its operations, which is annoying to create. **mpiwrap** thus provides a proxy object (`mpi::op_proxy`) for generating this signature from a binary operation. This proxy is created by calling `mpi::make_op` with either a pure lambda or a wrapped C++ function pointer. Unfortunately due to the way C++ function pointers interact with C function pointers, we are limited to these two options. As for the MPI version, `mpi::make_op` can be provided a `commute` setting, which has a standard value of `false`.