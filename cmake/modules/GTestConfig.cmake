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

set(Build_GTests OFF)
if(NOT TARGET gtest)
  unset(USE_GTEST)
  # Check for OpenCL GTest
  if(DEFINED USE_OPENCL_GTEST)
    if(EXISTS ${USE_OPENCL_GTEST})
      set(USE_GTEST ${USE_OPENCL_GTEST})
      message(STATUS "USE_OPENCL_GTEST is defined and exists: ${USE_GTEST}")
    else()
      message(STATUS "USE_OPENCL_GTEST is defined but path does not exist: ${USE_OPENCL_GTEST}")
    endif()
  endif()

  # Check for Vulkan GTest if USE_GTEST is still not set
  if(NOT DEFINED USE_GTEST AND DEFINED USE_VULKAN_GTEST)
    if(EXISTS ${USE_VULKAN_GTEST})
      set(USE_GTEST ${USE_VULKAN_GTEST})
      message(STATUS "USE_VULKAN_GTEST is defined and exists: ${USE_GTEST}")
    else()
      message(STATUS "USE_VULKAN_GTEST is defined but path does not exist: ${USE_VULKAN_GTEST}")
    endif()
  endif()

  # If USE_GTEST is set, use it
  if(DEFINED USE_GTEST)
    include(FetchContent)
    FetchContent_Declare(
      googletest
      SOURCE_DIR "${USE_GTEST}"
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    message(STATUS "Found gtest at ${USE_GTEST}")
    set(Build_GTests ON)
  elseif(ANDROID_ABI AND DEFINED ENV{ANDROID_NDK_HOME})
    set(GOOGLETEST_ROOT $ENV{ANDROID_NDK_HOME}/sources/third_party/googletest)
    if(EXISTS ${GOOGLETEST_ROOT}/src/gtest-all.cc)
      add_library(gtest_main STATIC ${GOOGLETEST_ROOT}/src/gtest_main.cc ${GOOGLETEST_ROOT}/src/gtest-all.cc)
      target_include_directories(gtest_main PRIVATE ${GOOGLETEST_ROOT})
      target_include_directories(gtest_main PUBLIC ${GOOGLETEST_ROOT}/include)
      message(STATUS "Using gtest from Android NDK at ${GOOGLETEST_ROOT}")
      set(Build_GTests ON)
    else()
      message(WARNING "gtest not found in Android NDK at ${GOOGLETEST_ROOT}")
    endif()
  else()
    message(WARNING "Cannot find gtest; GTests will not be built")
  endif()
else()
  set(Build_GTests ON)
endif()
