TARGET = concurrencycpp-test
OBJS = main.o

BUILD_PRX=1

INCDIR = 
CFLAGS = -O2 -Wall -std=c++20
CXXFLAGS = $(CFLAGS) 
ASFLAGS = $(CFLAGS)

LIBDIR =
LIBS = -lstdc++
LDFLAGS =

PSPSDK=$(shell psp-config --pspsdk-path)
include $(PSPSDK)/lib/build.mak
