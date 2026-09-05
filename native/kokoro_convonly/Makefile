CXX ?= g++
CXXFLAGS ?= -O3 -DNDEBUG -std=c++17 -Wall -Wextra -Wpedantic -ffp-contract=off -fPIC

RKNN_INC ?= /usr/include
RKNN_LIB ?= /usr/lib

all: libkokoro_convonly_native.so

libkokoro_convonly_native.so: kokoro_convonly_native.cpp kokoro_convonly_native.h kokoro_rk3588_bridge.cpp kokoro_rk3588_bridge.h
	$(CXX) $(CXXFLAGS) -I$(RKNN_INC) -shared kokoro_convonly_native.cpp kokoro_rk3588_bridge.cpp -L$(RKNN_LIB) -lrknnrt -lpthread -Wl,-soname,libkokoro_convonly_native.so -o $@

clean:
	rm -f libkokoro_convonly_native.so
