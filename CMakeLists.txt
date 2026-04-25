cmake_minimum_required(VERSION 3.10)
project(PlatonicSolids)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include_directories(
    ${CMAKE_SOURCE_DIR}/includes
)

link_directories(${CMAKE_SOURCE_DIR}/lib)

set(LIBS glfw3 opengl32 assimp)

add_library(GLAD src/glad.c)

add_executable(PlatonicSolids
    src/main.cpp
)

target_link_libraries(PlatonicSolids
    ${LIBS}
    GLAD
)

if(MSVC)
    target_compile_options(PlatonicSolids PRIVATE /std:c++17 /MP)
endif()