cmake_minimum_required(VERSION 3.25)
project(graph_khw)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(graph_khw main.cpp)

target_compile_options(graph_khw PRIVATE
        -fuse-ld=gold
        -fsanitize=address,undefined
        -fno-sanitize-recover=all
        -O0
        -Wall
        -Werror
        -Wsign-compare
        -g
        )

set(CMAKE_EXE_LINKER_FLAGS "-fsanitize=address,undefined -fno-sanitize-recover=all")

add_custom_target(run
        COMMAND graph_khw
        DEPENDS graph_khw
        WORKING_DIRECTORY ${CMAKE_PROJECT_DIR}
        )
