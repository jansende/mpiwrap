#this project is valid for cmake 3.1 to 3.11
cmake_minimum_required(VERSION 3.1...3.11)
#fix the above syntax for cmake <3.12
if(${CMAKE_VERSION} VERSION_LESS 3.12)
    cmake_policy(VERSION ${CMAKE_VERSION})
endif()

#create the mpiwrap project
project(mpiwrap VERSION 0.1.0
                DESCRIPTION "mpiwrap is a c++ wrapper for the horrible message passing interface."
                LANGUAGES CXX)
                
#add executable name
add_executable(mpiwrap include/mpiwrap/test.cpp)

find_package(MPI REQUIRED)
#fix for cmake <3.9 --> https://cliutils.gitlab.io/modern-cmake/chapters/packages/MPI.html
if(NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX IMPORTED INTERFACE)
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${MPI_CXX_COMPILE_FLAGS})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH})
    set_property(TARGET MPI::MPI_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${MPI_CXX_LINK_FLAGS} ${MPI_CXX_LIBRARIES})
endif()
target_link_libraries(mpiwrap PRIVATE MPI::MPI_CXX)



# #add header files
# target_include_directories(benri INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


# #add source files
# add_library(lace src/benri/casts.cpp
#                  src/benri/constants.cpp
#                  src/benri/math.cpp
#                  src/benri/types.cpp
#                  src/benri/types/part1.cpp
#                  src/benri/types/part2.cpp
#                  src/benri/types/part3.cpp
#                  src/benri/types/part4.cpp
#                  src/benri/types/part5.cpp
#                  src/benri/types/part6.cpp
#                  src/benri/types/part7.cpp
#                  src/card/config/config.cpp
#                  src/card/phase_space/functions.cpp
#                  src/card/phase_space/io.cpp
#                  src/card/phase_space/moments.cpp
#                  src/card/phase_space/phase_space.cpp
#                  src/card/physics/collisions.cpp
#                  src/card/physics/drag.cpp
#                  src/card/physics/initial.cpp
#                  src/card/physics/integrators.cpp
#                  src/card/physics/misc.cpp
#                  src/card/physics/orbits.cpp
#                  src/cmd/cmd.cpp
#                  src/helper.cpp
#                  src/phscut/phscut.cpp
#                  src/spdlog/spdlog.cpp
#                  src/std_math.cpp
#                  src/util/column_reader.cpp
#                  src/util/grids.cpp
#                  src/util/math.cpp
#                  src/util/md_writer.cpp
#                  src/util/stopwatch.cpp
#                  src/util/system.cpp
#                  src/util/tensors.cpp
# )
# #add include directory
# target_include_directories(lace PUBLIC include)
# #add libraries
# target_link_libraries(lace PRIVATE benri)
# target_link_libraries(lace PRIVATE card)
# target_link_libraries(lace PRIVATE chaiscript)
# target_link_libraries(lace PRIVATE cmd)
# target_link_libraries(lace PRIVATE spdlog)
# target_link_libraries(lace PRIVATE util)

# #add openmp support
# if(ACE_OPENMP_ENABLED)
#     find_package(OpenMP REQUIRED)
#     #fix for cmake <3.9 --> https://cliutils.gitlab.io/modern-cmake/chapters/packages/OpenMP.html
#     if(NOT TARGET OpenMP::OpenMP_CXX)
#         find_package(Threads REQUIRED)
#         add_library(OpenMP::OpenMP_CXX IMPORTED INTERFACE)      
#         set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_COMPILE_OPTIONS ${OpenMP_CXX_FLAGS})
    
#         #only works if the same flag is passed to the linker.
#         #use cmake 3.9+ otherwise (Intel, AppleClang)
#         set_property(TARGET OpenMP::OpenMP_CXX PROPERTY INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS} Threads::Threads)
#     endif()
#     target_link_libraries(lace PRIVATE OpenMP::OpenMP_CXX)

#     target_compile_definitions(lace PRIVATE USE_OPENMP)
# endif()

# #add compile features
# if (ACE_CPP17_ENABLED)
#     target_compile_features(lace PRIVATE cxx_std_17)
# else()
#     target_compile_features(lace PRIVATE cxx_std_14)
# endif()
# #set compiler flags
# if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
#     #we want as many warnings as possible
#     target_compile_options(lace PRIVATE -Weverything)
#     #disable annoying warnings
#     target_compile_options(lace PRIVATE -Wno-c++98-compat
#                                         -Wno-c++98-compat-pedantic
#                                         -Wno-newline-eof
#                                         -Wno-nonportable-system-include-path
#                                         -Wno-shadow-field-in-constructor
#                                         -Wno-shadow-uncaptured-local
#                                         -Wno-covered-switch-default
#                                         -Wno-extra-semi
#     )
#     #disable class/struct padding warning
#     target_compile_options(lace PRIVATE -Wno-padded)
#     #disable warnings caused by fmt/spdlog
#     target_compile_options(lace PRIVATE -Wno-deprecated-declarations
#                                         -Wno-documentation-unknown-command
#                                         -Wno-inconsistent-missing-destructor-override
#                                         -Wno-unused-member-function
#                                         -Wno-language-extension-token
#                                         -Wno-exit-time-destructors
#                                         -Wno-switch-enum
#                                         -Wno-old-style-cast
#     )
#     #disable warnings caused by benri
#     target_compile_options(lace PRIVATE -Wno-float-equal)
#     #prevent clang on windows from importing min/max from windows headers
#     target_compile_definitions(lace PRIVATE NOMINMAX)
#     #do not warn about openmp pragmas, when we are not using openmp
#     if(NOT ACE_OPENMP_ENABLED)
#         target_compile_options(lace PRIVATE -Wno-source-uses-openmp)
#     endif()
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#     #we want as many warnings as possible
#     target_compile_options(lace PRIVATE -Wall -Wextra -Wpedantic)
# elseif (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
#     #we want as many warnings as possible
#     target_compile_options(lace PRIVATE /W4)
#     #disable unknown pragma warnings
#     target_compile_options(lace PRIVATE /wd4068)
#     #fix for x64 chaiscript
#     target_compile_options(lace PRIVATE /bigobj)
#     #prevent msvc from importing min/max from windows headers
#     target_compile_definitions(lace PRIVATE NOMINMAX)
#     #apply a fix for the new two-phase name lookup (which is incompatible with openmp at the moment)
#     target_compile_options(lace PRIVATE /Zc:twoPhase-)
# endif()