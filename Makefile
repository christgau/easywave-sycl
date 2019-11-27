CXX=dpcpp
CXXFLAGS=-O0 -g
LDLIBS=-lm

SOURCES=\
	EasyWave.cpp.dp.cpp \
	ewGpuNode.dp.cpp \
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
	utilits.cpp \
#	ewCudaKernels.dp.cpp


OBJECTS=$(patsubst %.cpp,%.o,$(SOURCES))

easywave: $(OBJECTS)
	$(CXX) $^ $(LDLIBS) -o $@

.PHONY: clean

clean:
	rm -f *.o easywave
