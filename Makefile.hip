CXX=hipcc
CXXFLAGS=-O3 -std=c++17 -Wall -Wextra

SOURCES=\
	EasyWave.cpp \
	ewKernels.hip.cpp \
	ewGpuNode.hip.cpp \
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

OBJECTS=$(patsubst %.cu,%.o,$(patsubst %.cpp,%.o,$(SOURCES)))

easywave-hip: $(OBJECTS)
	$(CXX) $^ -o $@

.PHONY: clean

clean:
	rm -f *.o easywave-hip
