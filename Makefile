#CXX=clang++
CXXFLAGS=-O3 -ffast-math -march=native -mtune=native -ftree-vectorize -ffast-math -Rpass=loop-vectorize -cl-fast-relaxed-math -fvectorize
#CXXFLAGS=-O3 -march=native -mtune=native -ftree-vectorize -ffast-math -Rpass=loop-vectorize -cl-fast-relaxed-math -fvectorize \
	--gcc-toolchain=$(GCC_ROOT) \
	-fsycl \
	-fsycl-unnamed-lambda \
	-fsycl-targets=nvptx64-nvidia-cuda-sycldevice \
	-nocudalib \
	-I$(HOME)/opt/local/intel/oneapi/dpct/latest/include
CXX=dpcpp
LDLIBS=-lm

SOURCES=\
	EasyWave.cpp \
	cOgrd.cpp \
	cOkadaEarthquake.cpp \
	cOkadaFault.cpp \
	cSphere.cpp \
	ewCudaKernels.cpp \
	ewGpuNode.cpp \
	ewGrid.cpp \
	ewOut2D.cpp \
	ewParam.cpp \
	ewPOIs.cpp \
	ewSource.cpp \
	ewStep.cpp \
	okada.cpp \
	utilits.cpp

OBJECTS=$(patsubst %.cpp,%.o,$(SOURCES))

easywave: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

.PHONY: rename

rename:
	for f in $$(find . -name '*cpp.dp.cpp'); do mv -v $$f $$(sed 's:\.dp\.cpp::' <<< "$$f"); done
	for f in $$(find . -name '*.dp.[ch]pp'); do mv -v $$f $$(sed 's:\.dp::' <<< "$$f"); done
	find . -name '*.[ch]*' | xargs -I%% sed -i -e 's:\.dp\.:\.:' %%

.PHONY: clean

clean:
	rm -f *.o easywave
