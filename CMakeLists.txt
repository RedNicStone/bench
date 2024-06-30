
cmake_minimum_required(VERSION 3.28)
project(bench)
include(CTest)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -O3")

function(bench_add_plight target_name)

    add_executable(blight_${target_name} ${CMAKE_CURRENT_SOURCE_DIR}/bench.cpp ${ARGN})
    target_link_libraries(blight_${target_name}
        PRIVATE
            nanobench::nanobench
            doctest
    )
    target_compile_definitions(blight_${target_name}
        PRIVATE
            DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
    )
    add_test(NAME bench_${target_name}
            COMMAND blight_${target_name}
    )

endfunction()

add_subdirectory(lib)
add_subdirectory(plights)
