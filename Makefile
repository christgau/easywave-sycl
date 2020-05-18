CXX=dpcpp
CXXFLAGS=-O0 -g
#-march=native -mtune=native -Rpass=loop-vectorize
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
