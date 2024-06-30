
cmake_minimum_required(VERSION 3.28)
project(bench)

set(CMAKE_CXX_STANDARD 23)

function(bench_add_plight target_name)

    add_executable(${target_name} ${CMAKE_CURRENT_SOURCE_DIR}/bench_${target_name}.cpp)
    target_link_libraries(${target_name}
        PRIVATE
            nanobench::nanobench
    )

endfunction()

add_subdirectory(lib)
add_subdirectory(plights)
