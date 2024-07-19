QT += widgets
CONFIG += c++11

SOURCES += main.cpp

# CUDA settings
CUDA_DIR = /usr/local/cuda  # Adjust this path if necessary
INCLUDEPATH += $$CUDA_DIR/include
LIBS += -L$$CUDA_DIR/lib64 -lcudart

# CUDA sources
CUDA_SOURCES += main.cu

# Custom CUDA compiler configuration
CUDA_LIBS = -lcudart -lcuda

# NVCC flags
NVCCFLAGS = --compiler-options -fno-strict-aliasing -use_fast_math --ptxas-options=-v

# Add here your specific CUDA architecture
CUDA_ARCH = sm_50  # Modify this to match your GPU architecture

# CUDA compile
cuda.input = CUDA_SOURCES
cuda.output = ${QMAKE_FILE_BASE}_cuda.o
cuda.commands = nvcc -c $$NVCCFLAGS -arch=$$CUDA_ARCH -o ${QMAKE_FILE_OUT} ${QMAKE_FILE_NAME}
cuda.dependency_type = TYPE_C
QMAKE_EXTRA_COMPILERS += cuda