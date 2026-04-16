# Makefile for DeepSeek MoE CUDA + NCCL implementation

NVCC      := nvcc
NCCL_INC  ?= /usr/include
NCCL_LIB  ?= /usr/lib/x86_64-linux-gnu
CUDA_INC  ?= /usr/local/cuda/include

NVCC_FLAGS := -O3 -arch=sm_70 \
              -I$(NCCL_INC) -I$(CUDA_INC) \
              -lineinfo \
              --expt-relaxed-constexpr

LDFLAGS   := -L$(NCCL_LIB) -lnccl -lcudart -lm

TARGET    := deepseek_moe

all: $(TARGET)

$(TARGET): deepseek_moe.cu
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(LDFLAGS)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean run
