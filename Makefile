include make.inc

BASE_SOURCES=\
	EasyWave.cpp \
	cOgrd.cpp \
	cOkadaEarthquake.cpp \
	cOkadaFault.cpp \
	cSphere.cpp \
	ewGpuNode.cpp \
	ewGrid.cpp \
	ewOut2D.cpp \
	ewParam.cpp \
	ewPOIs.cpp \
	ewSource.cpp \
	ewStep.cpp \
	okada.cpp \
	utilits.cpp

ifeq ($(strip $(INLINE_KERNELS)),)
EXTRA_SOURCES=ewCudaKernels.cpp
endif

SOURCES=$(EXTRA_SOURCES) $(BASE_SOURCES)
OBJECTS=$(patsubst %.cpp,%.o,$(SOURCES))

easywave: $(OBJECTS)
	$(CXX) $(CXXFLAGS) $^ $(LDLIBS) -o $@

.PHONY: clean

clean:
	rm -f *.o easywave
