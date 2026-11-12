# Makefile
JULIA ?= julia

ROOT_DIR:=$(shell pwd)
TARGET="CAMNASCompiled"

.PHONY: all, default, clean, new

default: all

all: camnasjl.so wrapper.o plugin.so

new: clean all

clean:
	rm -rf $(ROOT_DIR)/$(TARGET)
	rm -f $(ROOT_DIR)/camnasjl.so
	rm -f $(ROOT_DIR)/dpsim_wrapper.o

camnasjl.so: build/build.jl src/CAMNAS.jl src/mna_solver.jl src/config.jl build/precompile_statements.jl
	$(JULIA) --project=. --threads=auto --startup-file=no build/build.jl $(TARGET)

wrapper.o: camnasjl.so
	gcc -c -fPIC -O2 -I$(ROOT_DIR)/CAMNASCompiled/include -I$(ROOT_DIR)/../../../include build/dpsim_wrapper.c

plugin.so: wrapper.o
	gcc -shared -o camnasjl.so dpsim_wrapper.o  -L$(ROOT_DIR)/CAMNASCompiled/lib -lcamnasjl
