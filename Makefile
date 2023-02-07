include make.inc

BASE_SOURCES=\
	EasyWave.cpp \
	ewGpuNode.sycl.cpp \
	cOgrd.cpp \
	cOkadaEarthquake.cpp \
	cOkadaFault.cpp \
	cSphere.cpp \
	ewGrid.cpp \
	ewOut2D.cpp \
	ewParam.cpp \
	ewPOIs.cpp \
	ewSource.cpp \
	ewStep.cpp \
	okada.cpp \
	utilits.cpp

ifeq ($(strip $(INLINE_KERNELS)),)
EXTRA_SOURCES=ewKernels.sycl.cpp
endif

SOURCES=$(EXTRA_SOURCES) $(BASE_SOURCES)
OBJECTS=$(patsubst %.cpp,%.o,$(SOURCES))

easywave-sycl: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean

clean:
	rm -f *.o easywave-sycl
