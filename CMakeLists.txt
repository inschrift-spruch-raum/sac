cmake_minimum_required(VERSION 3.14)

project(sac LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(MSYS2_Clang "Compile using Clang in MSYS2" OFF)
option(ENABLE_AVX2 "AVX2/FMA" OFF)
option(ENABLE_AVX512 "AVX512" OFF)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS src/*.cpp)

add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/utf-8>)

add_executable(sac ${SOURCES})

if(ENABLE_AVX512)
    target_compile_options(sac PRIVATE
        -mavx512f
        -mavx512dq
        -mavx512bw
        -mavx512vl
        -mfma
    )
    target_compile_definitions(sac PRIVATE USE_AVX512=1)
elseif(ENABLE_AVX2)
    target_compile_options(sac PRIVATE
        -mavx2
        -mfma
    )
    target_compile_definitions(sac PRIVATE USE_AVX256=1)
endif()

if(MSYS2_Clang)
    target_link_libraries(C++23-Module-Example PRIVATE 
        -static
    )
endif()
