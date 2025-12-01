# Makefile for HIP N-Body Simulation (hw5)

CXX = hipcc
CXXFLAGS = -O3 -std=c++17 --offload-arch=gfx908:sramecc+:xnack- -munsafe-fp-atomics -ffast-math
CXXFLAGS += -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -Rpass-analysis=kernel-resource-usage
TARGET = hw5

# Source file
SRC = hw5.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

# Build optimized HIP version
hip: hw5_hip.cpp
	$(CXX) $(CXXFLAGS) -o hw5_hip hw5_hip.cpp

# Build Pb1-only optimized version
pb1: hw5_pb1.cpp
	$(CXX) $(CXXFLAGS) -o hw5_pb1 hw5_pb1.cpp

# Build Pb1 v2 (cooperative kernel)
pb1v2: hw5_pb1_v2.cpp
	$(CXX) $(CXXFLAGS) -o hw5_pb1_v2 hw5_pb1_v2.cpp

# Build Pb1 v3 (cooperative kernel, multi-body per block)
pb1v3: hw5_pb1_v3.cpp
	$(CXX) $(CXXFLAGS) -o hw5_pb1_v3 hw5_pb1_v3.cpp

# Build Pb1 hybrid (cooperative for small N, separate kernels for large N)
pb1h: hw5_pb1_hybrid.cpp
	$(CXX) $(CXXFLAGS) -o hw5_pb1_hybrid hw5_pb1_hybrid.cpp

# Build full version (Pb1 + Pb2 + Pb3, dual GPU)
full: hw5_full.cpp
	$(CXX) $(CXXFLAGS) -o hw5_full hw5_full.cpp

# Build full version v2 (Pb2 uses per-step kernel)
fullv2: hw5_full_v2.cpp
	$(CXX) $(CXXFLAGS) -o hw5_full_v2 hw5_full_v2.cpp

# Build full version v3 (Pb2+Pb3 both use per-step kernel)
fullv3: hw5_full_v3.cpp
	$(CXX) $(CXXFLAGS) -o hw5_full_v3 hw5_full_v3.cpp

# Build full version v4 (Pb2 first, then Pb3)
fullv4: hw5_full_v4.cpp
	$(CXX) $(CXXFLAGS) -o hw5_full_v4 hw5_full_v4.cpp

# Test Pb2 with cooperative kernel (like Pb1)
test_pb2_coop: test_pb2_coop.cpp
	$(CXX) $(CXXFLAGS) -o test_pb2_coop test_pb2_coop.cpp

# Build full version v5 (Pb2 cooperative kernel)
fullv5: hw5_full_v5.cpp
	$(CXX) $(CXXFLAGS) -o hw5_full_v5 hw5_full_v5.cpp

clean:
	rm -f $(TARGET) hw5_hip hw5_pb1 hw5_pb1_v2 hw5_pb1_v3 hw5_pb1_hybrid hw5_full hw5_full_v2 hw5_full_v3 hw5_full_v4 hw5_full_v5 test_pb2_coop

test: $(TARGET)
	./$(TARGET) testcases/b20.in output.txt
	cat output.txt

test-hip: hip
	./hw5_hip testcases/b20.in output.txt
	cat output.txt

test-pb1: pb1
	./hw5_pb1 testcases/b20.in output_pb1.txt
	cat output_pb1.txt

.PHONY: all clean test test-hip hip pb1 test-pb1
