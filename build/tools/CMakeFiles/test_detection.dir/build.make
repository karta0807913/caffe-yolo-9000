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
include tools/CMakeFiles/test_detection.dir/depend.make

# Include the progress variables for this target.
include tools/CMakeFiles/test_detection.dir/progress.make

# Include the compile flags for this target's objects.
include tools/CMakeFiles/test_detection.dir/flags.make

tools/CMakeFiles/test_detection.dir/test_detection.cpp.o: tools/CMakeFiles/test_detection.dir/flags.make
tools/CMakeFiles/test_detection.dir/test_detection.cpp.o: ../tools/test_detection.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/e419/caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tools/CMakeFiles/test_detection.dir/test_detection.cpp.o"
	cd /home/e419/caffe/build/tools && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/test_detection.dir/test_detection.cpp.o -c /home/e419/caffe/tools/test_detection.cpp

tools/CMakeFiles/test_detection.dir/test_detection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/test_detection.dir/test_detection.cpp.i"
	cd /home/e419/caffe/build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/e419/caffe/tools/test_detection.cpp > CMakeFiles/test_detection.dir/test_detection.cpp.i

tools/CMakeFiles/test_detection.dir/test_detection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/test_detection.dir/test_detection.cpp.s"
	cd /home/e419/caffe/build/tools && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/e419/caffe/tools/test_detection.cpp -o CMakeFiles/test_detection.dir/test_detection.cpp.s

tools/CMakeFiles/test_detection.dir/test_detection.cpp.o.requires:

.PHONY : tools/CMakeFiles/test_detection.dir/test_detection.cpp.o.requires

tools/CMakeFiles/test_detection.dir/test_detection.cpp.o.provides: tools/CMakeFiles/test_detection.dir/test_detection.cpp.o.requires
	$(MAKE) -f tools/CMakeFiles/test_detection.dir/build.make tools/CMakeFiles/test_detection.dir/test_detection.cpp.o.provides.build
.PHONY : tools/CMakeFiles/test_detection.dir/test_detection.cpp.o.provides

tools/CMakeFiles/test_detection.dir/test_detection.cpp.o.provides.build: tools/CMakeFiles/test_detection.dir/test_detection.cpp.o


# Object files for target test_detection
test_detection_OBJECTS = \
"CMakeFiles/test_detection.dir/test_detection.cpp.o"

# External object files for target test_detection
test_detection_EXTERNAL_OBJECTS =

tools/test_detection: tools/CMakeFiles/test_detection.dir/test_detection.cpp.o
tools/test_detection: tools/CMakeFiles/test_detection.dir/build.make
tools/test_detection: lib/libcaffe.so.1.0.0-rc3
tools/test_detection: lib/libproto.a
tools/test_detection: /usr/lib/x86_64-linux-gnu/libboost_system.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libboost_thread.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libboost_atomic.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libglog.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libsz.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libz.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libdl.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libm.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libpthread.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libglog.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libgflags.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libprotobuf.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5_hl.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/hdf5/serial/lib/libhdf5.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libsz.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libz.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libdl.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libm.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/liblmdb.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libleveldb.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libsnappy.so
tools/test_detection: /usr/local/cuda/lib64/libcudart.so
tools/test_detection: /usr/local/cuda/lib64/libcurand.so
tools/test_detection: /usr/local/cuda/lib64/libcublas.so
tools/test_detection: /usr/local/cuda/lib64/libcudnn.so
tools/test_detection: /usr/local/lib/libopencv_highgui.so.2.4.13
tools/test_detection: /usr/local/lib/libopencv_imgproc.so.2.4.13
tools/test_detection: /usr/local/lib/libopencv_core.so.2.4.13
tools/test_detection: /usr/local/cuda/lib64/libcudart.so
tools/test_detection: /usr/local/cuda/lib64/libnppc.so
tools/test_detection: /usr/local/cuda/lib64/libnppi.so
tools/test_detection: /usr/local/cuda/lib64/libnpps.so
tools/test_detection: /usr/lib/liblapack.so
tools/test_detection: /usr/lib/libcblas.so
tools/test_detection: /usr/lib/libatlas.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libpython2.7.so
tools/test_detection: /usr/lib/x86_64-linux-gnu/libboost_python.so
tools/test_detection: tools/CMakeFiles/test_detection.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/e419/caffe/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable test_detection"
	cd /home/e419/caffe/build/tools && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/test_detection.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tools/CMakeFiles/test_detection.dir/build: tools/test_detection

.PHONY : tools/CMakeFiles/test_detection.dir/build

tools/CMakeFiles/test_detection.dir/requires: tools/CMakeFiles/test_detection.dir/test_detection.cpp.o.requires

.PHONY : tools/CMakeFiles/test_detection.dir/requires

tools/CMakeFiles/test_detection.dir/clean:
	cd /home/e419/caffe/build/tools && $(CMAKE_COMMAND) -P CMakeFiles/test_detection.dir/cmake_clean.cmake
.PHONY : tools/CMakeFiles/test_detection.dir/clean

tools/CMakeFiles/test_detection.dir/depend:
	cd /home/e419/caffe/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/e419/caffe /home/e419/caffe/tools /home/e419/caffe/build /home/e419/caffe/build/tools /home/e419/caffe/build/tools/CMakeFiles/test_detection.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tools/CMakeFiles/test_detection.dir/depend

