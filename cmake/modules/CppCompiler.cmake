# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

if(USE_CPP_COMPILER)
  # Raise priority for all CPP legalizations
  add_definitions(-DTVM_LEGALIZE_CPP_LEVEL=20)
  message(STATUS "CppCompiler : Enabled")
else()
  add_definitions(-DTVM_LEGALIZE_CPP_LEVEL=5)
  message(STATUS "CppCompiler : Disabled")
endif(USE_CPP_COMPILER)

if(USE_CPP_COMPILER_TESTS)
  if(EXISTS ${USE_CPP_COMPILER_TESTS})
    include(FetchContent)
    FetchContent_Declare(googletest SOURCE_DIR "${USE_CPP_COMPILER_TESTS}")
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    install(TARGETS gtest EXPORT ${PROJECT_NAME}Targets DESTINATION lib${LIB_SUFFIX})

    message(STATUS "Found Cpp Compiler gtest at ${USE_CPP_COMPILER_TESTS}")
    set(Build_CppCompiler_GTests ON)
  elseif (ANDROID_ABI AND DEFINED ENV{ANDROID_NDK_HOME})
    set(GOOGLETEST_ROOT $ENV{ANDROID_NDK_HOME}/sources/third_party/googletest)
    if(NOT TARGET gtest_main)
      add_library(gtest_main STATIC ${GOOGLETEST_ROOT}/src/gtest_main.cc ${GOOGLETEST_ROOT}/src/gtest-all.cc)
      target_include_directories(gtest_main PRIVATE ${GOOGLETEST_ROOT})
      target_include_directories(gtest_main PUBLIC ${GOOGLETEST_ROOT}/include)
    endif()
    message(STATUS "Using gtest from Android NDK")
    set(Build_CppCompiler_GTests ON)
  endif()

  if(Build_CppCompiler_GTests)
    message(STATUS "Building CppCompiler-Gtests")
    tvm_file_glob(GLOB_RECURSE CPP_COMPILER_TEST_SRCS
      "tests/cpp-compiler/*.cc"
    )
    add_executable(cpp-compiler-test ${CPP_COMPILER_TEST_SRCS})
    target_link_libraries(cpp-compiler-test PRIVATE gtest_main tvm ${CppCompiler_LIBRARIES})
  else()
    message(STATUS "Couldn't build CppCompiler-Gtests")
  endif()
endif(USE_CPP_COMPILER_TESTS)
