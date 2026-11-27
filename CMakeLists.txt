cmake_minimum_required(VERSION 3.16)
project(libkittyops VERSION 0.1.0 LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

option(KITTYOPS_BUILD_SHARED "Build shared libkittyops" ON)
option(KITTYOPS_BUILD_STATIC "Build static libkittyops" ON)
option(KITTYOPS_BUILD_TESTS "Build C unit tests" ON)
option(KITTYOPS_BUILD_BENCH "Build 1080p benchmark binary" ON)
option(KITTYOPS_USE_LIBDEFLATE "Use libdeflate if found" ON)
option(KITTYOPS_USE_ZLIB_NG "Prefer zlib-ng if found" ON)
option(KITTYOPS_USE_ISAL "Use Intel ISA-L if found" ON)
option(KITTYOPS_PREFER_STATIC_DEPS "Prefer static archives for optional compression deps (relocatable builds)" OFF)
option(KITTYOPS_PYTHON_WHEEL "Build single unversioned shared lib installed into kittyops/ Python package" OFF)

include(GNUInstallDirs)
find_package(Threads REQUIRED)

# Compression backend detection:
# 1. Streaming zlib backend (for parallel multi-slice deflate):
#    prefer zlib-ng if available, else system zlib.
set(KOPS_ZLIB_LIBS "")
set(KOPS_DEFS "")
set(KOPS_INC_DIRS "")

if(KITTYOPS_USE_ZLIB_NG)
  find_path(ZLIB_NG_INCLUDE_DIR zlib-ng.h PATHS /opt/homebrew/include /usr/local/include /usr/include)
  find_library(ZLIB_NG_LIBRARY NAMES z-ng zlib-ng PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
  if(ZLIB_NG_INCLUDE_DIR AND ZLIB_NG_LIBRARY)
    message(STATUS "libkittyops: using zlib-ng (${ZLIB_NG_LIBRARY})")
    list(APPEND KOPS_DEFS KOPS_HAVE_ZLIB_NG=1)
    list(APPEND KOPS_INC_DIRS "${ZLIB_NG_INCLUDE_DIR}")
    list(APPEND KOPS_ZLIB_LIBS "${ZLIB_NG_LIBRARY}")
  endif()
endif()

if(NOT KOPS_ZLIB_LIBS)
  find_package(ZLIB REQUIRED)
  message(STATUS "libkittyops: using system zlib (${ZLIB_LIBRARIES})")
  list(APPEND KOPS_ZLIB_LIBS ZLIB::ZLIB)
endif()

# 2. Fast single-block libdeflate (for small/medium rects < 224 KB and multi-rect parallel workers)
if(KITTYOPS_USE_LIBDEFLATE)
  find_path(LIBDEFLATE_INCLUDE_DIR libdeflate.h PATHS /opt/homebrew/include /usr/local/include /usr/include)
  if(KITTYOPS_PREFER_STATIC_DEPS)
    find_library(LIBDEFLATE_LIBRARY NAMES libdeflate.a deflate libdeflate PATHS /opt/homebrew/lib /usr/local/lib /usr/lib /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu)
  else()
    find_library(LIBDEFLATE_LIBRARY NAMES deflate libdeflate PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
  endif()
  if(LIBDEFLATE_INCLUDE_DIR AND LIBDEFLATE_LIBRARY)
    message(STATUS "libkittyops: using libdeflate (${LIBDEFLATE_LIBRARY})")
    list(APPEND KOPS_DEFS KOPS_HAVE_LIBDEFLATE=1)
    list(APPEND KOPS_INC_DIRS "${LIBDEFLATE_INCLUDE_DIR}")
    list(APPEND KOPS_ZLIB_LIBS "${LIBDEFLATE_LIBRARY}")
  endif()
endif()

# 3. Optional Intel ISA-L (ultra-fast level 0-3 deflate with streaming flush)
if(KITTYOPS_USE_ISAL)
  find_path(ISAL_INCLUDE_DIR isa-l/igzip_lib.h PATHS /opt/homebrew/include /usr/local/include /usr/include)
  find_library(ISAL_LIBRARY NAMES isal PATHS /opt/homebrew/lib /usr/local/lib /usr/lib)
  if(ISAL_INCLUDE_DIR AND ISAL_LIBRARY)
    message(STATUS "libkittyops: using Intel ISA-L (${ISAL_LIBRARY})")
    list(APPEND KOPS_DEFS KOPS_HAVE_ISAL=1)
    list(APPEND KOPS_INC_DIRS "${ISAL_INCLUDE_DIR}")
    list(APPEND KOPS_ZLIB_LIBS "${ISAL_LIBRARY}")
  endif()
endif()

set(KOPS_SOURCES
  src/parallel.c
  src/diff.c
  src/scroll.c
  src/compress.c
  src/apc.c
  src/encoder.c
)

function(kops_configure_target tgt)
  target_include_directories(${tgt}
    PUBLIC
      $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
      $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/src
      ${KOPS_INC_DIRS}
  )
  target_compile_definitions(${tgt} PRIVATE ${KOPS_DEFS})
  target_link_libraries(${tgt} PRIVATE ${KOPS_ZLIB_LIBS} Threads::Threads)
  if(NOT MSVC)
    target_compile_options(${tgt} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
  if(UNIX AND NOT APPLE)
    target_link_libraries(${tgt} PRIVATE rt)
  endif()
endfunction()

if(KITTYOPS_BUILD_STATIC)
  add_library(kittyops_static STATIC ${KOPS_SOURCES})
  set_target_properties(kittyops_static PROPERTIES OUTPUT_NAME kittyops POSITION_INDEPENDENT_CODE ON)
  kops_configure_target(kittyops_static)
  add_library(kittyops::static ALIAS kittyops_static)
endif()

if(KITTYOPS_BUILD_SHARED)
  add_library(kittyops SHARED ${KOPS_SOURCES})
  if(KITTYOPS_PYTHON_WHEEL)
    set_target_properties(kittyops PROPERTIES
      C_VISIBILITY_PRESET default
      INSTALL_NAME_DIR "@loader_path"
    )
  else()
    set_target_properties(kittyops PROPERTIES
      VERSION ${PROJECT_VERSION}
      SOVERSION 0
      C_VISIBILITY_PRESET default
    )
  endif()
  kops_configure_target(kittyops)
  add_library(kittyops::shared ALIAS kittyops)
endif()

if(KITTYOPS_BUILD_TESTS)
  enable_testing()
  add_executable(test_kittyops tests/test_kittyops.c)
  if(TARGET kittyops_static)
    target_link_libraries(test_kittyops PRIVATE kittyops_static ${KOPS_ZLIB_LIBS} Threads::Threads)
  else()
    target_link_libraries(test_kittyops PRIVATE kittyops ${KOPS_ZLIB_LIBS} Threads::Threads)
  endif()
  target_include_directories(test_kittyops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include ${KOPS_INC_DIRS})
  target_compile_definitions(test_kittyops PRIVATE ${KOPS_DEFS})
  add_test(NAME kittyops_c_tests COMMAND test_kittyops)
endif()

if(KITTYOPS_BUILD_BENCH)
  add_executable(bench_kittyops bench/bench_kittyops.c)
  if(TARGET kittyops_static)
    target_link_libraries(bench_kittyops PRIVATE kittyops_static ${KOPS_ZLIB_LIBS} Threads::Threads)
  else()
    target_link_libraries(bench_kittyops PRIVATE kittyops ${KOPS_ZLIB_LIBS} Threads::Threads)
  endif()
  target_include_directories(bench_kittyops PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

if(KITTYOPS_PYTHON_WHEEL)
  if(TARGET kittyops)
    install(TARGETS kittyops LIBRARY DESTINATION kittyops RUNTIME DESTINATION kittyops)
  endif()
else()
  install(FILES include/kittyops.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
  if(TARGET kittyops_static)
    install(TARGETS kittyops_static ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
  endif()
  if(TARGET kittyops)
    install(TARGETS kittyops LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR} RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
  endif()
endif()
