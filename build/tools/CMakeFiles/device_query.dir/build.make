# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/e419/caffe

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/e419/caffe/build

# Include any dependencies generated for this target.
include tools/CMakeFiles/device_query.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/device_query.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/device_query.dir/flags.make

tools/CMakeFiles/device_query.dir/device_query.cpp.o: tools/CMakeFiles/device_query.dir/flags.make
tools/CMakeFiles/device_query.dir/device_query.cpp.o: ../tools/device_query.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/e419/caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/device_query.dir/device_query.cpp.o"
	cd /home/e419/caffe/build/tools && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/device_query.dir/device_query.cpp.o -c /home/e419/caffe/tools/device_query.cpp

tools/CMakeFiles/device_query.dir/device_query.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/device_query.dir/device_query.cpp.i"
	cd /home/e419/caffe/build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/e419/caffe/tools/device_query.cpp > CMakeFiles/device_query.dir/device_query.cpp.i

tools/CMakeFiles/device_query.dir/device_query.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/device_query.dir/device_query.cpp.s"
	cd /home/e419/caffe/build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/e419/caffe/tools/device_query.cpp -o CMakeFiles/device_query.dir/device_query.cpp.s

tools/CMakeFiles/device_query.dir/device_query.cpp.o.requires:

.PHONY : tools/CMakeFiles/device_query.dir/device_query.cpp.o.requires

tools/CMakeFiles/device_query.dir/device_query.cpp.o.provides: tools/CMakeFiles/device_query.dir/device_query.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/device_query.dir/build.make tools/CMakeFiles/device_query.dir/device_query.cpp.o.provides.build
.PHONY : tools/CMakeFiles/device_query.dir/device_query.cpp.o.provides

tools/CMakeFiles/device_query.dir/device_query.cpp.o.provides.build: tools/CMakeFiles/device_query.dir/device_query.cpp.o


# Object files for target device_query
device_query_OBJECTS = \
"CMakeFiles/device_query.dir/device_query.cpp.o"

# External object files for target device_query
device_query_EXTERNAL_OBJECTS =

tools/device_query: tools/CMakeFiles/device_query.dir/device_query.cpp.o
tools/device_query: tools/CMakeFiles/device_query.dir/build.make
tools/device_query: lib/libcaffe.so.1.0.0-rc3
tools/device_query: lib/libproto.a
tools/device_query: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libglog.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/device_query: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
tools/device_query: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libsz.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libz.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libdl.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libm.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libglog.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/device_query: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
tools/device_query: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libsz.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libz.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libdl.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libm.so
tools/device_query: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libsnappy.so
tools/device_query: /usr/local/cuda/lib64/libcudart.so
tools/device_query: /usr/local/cuda/lib64/libcurand.so
tools/device_query: /usr/local/cuda/lib64/libcublas.so
tools/device_query: /usr/local/cuda/lib64/libcudnn.so
tools/device_query: /usr/local/lib/libopencv_highgui.so.2.4.13
tools/device_query: /usr/local/lib/libopencv_imgproc.so.2.4.13
tools/device_query: /usr/local/lib/libopencv_core.so.2.4.13
tools/device_query: /usr/local/cuda/lib64/libcudart.so
tools/device_query: /usr/local/cuda/lib64/libnppc.so
tools/device_query: /usr/local/cuda/lib64/libnppi.so
tools/device_query: /usr/local/cuda/lib64/libnpps.so
tools/device_query: /usr/lib/liblapack.so
tools/device_query: /usr/lib/libcblas.so
tools/device_query: /usr/lib/libatlas.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libpython2.7.so
tools/device_query: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/device_query: tools/CMakeFiles/device_query.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/e419/caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable device_query"
	cd /home/e419/caffe/build/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/device_query.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/device_query.dir/build: tools/device_query

.PHONY : tools/CMakeFiles/device_query.dir/build

tools/CMakeFiles/device_query.dir/requires: tools/CMakeFiles/device_query.dir/device_query.cpp.o.requires

.PHONY : tools/CMakeFiles/device_query.dir/requires

tools/CMakeFiles/device_query.dir/clean:
	cd /home/e419/caffe/build/tools && $(CMAKE_COMMAND) -P CMakeFiles/device_query.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/device_query.dir/clean

tools/CMakeFiles/device_query.dir/depend:
	cd /home/e419/caffe/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/e419/caffe /home/e419/caffe/tools /home/e419/caffe/build /home/e419/caffe/build/tools /home/e419/caffe/build/tools/CMakeFiles/device_query.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/device_query.dir/depend

