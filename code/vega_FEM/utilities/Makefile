R ?= ../
L=$(R)/libraries
LIBRARIES_DIR=$(L)
# argument $(1) is a list of library names (e.g. "integrator minivector")
GET_LIB_MAKEFILES=$(addsuffix /Makefile, $(addprefix $(R)/libraries/, $(1)))
GET_LIB_FILENAMES=$(join $(addprefix $(R)/libraries/, $(1)), $(addsuffix .a, $(addprefix /lib, $(notdir $(1)) )))

CXX = g++
#CXXFLAGS= -g -std=c++11 -fsanitize=address -fsanitize=undefined
CXXFLAGS= -g -O3 -std=c++11 -DGL_SILENCE_DEPRECATION -Wno-deprecated-declarations -Wno-deprecated

VOLUMETRICMESHUTILITIES_LIBS= volumetricMesh matrixIO objMesh mesh basicAlgorithms getopts

VOLUMETRICMESHUTILITIES_HEADERS=

VOLUMETRICMESHUTILITIES_LINK=$(addprefix -l, $(VOLUMETRICMESHUTILITIES_LIBS)) $(CGAL_LIB) $(PARDISO_LIB) $(IMAGE_LIBS) $(STANDARD_LIBS)

VOLUMETRICMESHUTILITIES_LIB_MAKEFILES=$(call GET_LIB_MAKEFILES, $(VOLUMETRICMESHUTILITIES_LIBS))
VOLUMETRICMESHUTILITIES_LIB_FILENAMES=$(call GET_LIB_FILENAMES, $(VOLUMETRICMESHUTILITIES_LIBS))

include $(VOLUMETRICMESHUTILITIES_LIB_MAKEFILES)

ALL= interpolateData textMatrix2Matrix



all: $(ALL)

$(ALL): % : %.o
	$(CXXLD) $(LDFLAGS) $^ $(VOLUMETRICMESHUTILITIES_LINK) -o $@; cp $@ $(R)/utilities/bin/

%.o: %.cpp $(VOLUMETRICMESHUTILITIES_LIB_FILENAMES) $(VOLUMETRICMESHUTILITIES_HEADERS)
	$(CXX) $(CXXFLAGS) -c $(GLUI_INCLUDE) $(INCLUDE) $< -o $@

clean:
	rm -f *.o $(ALL)
