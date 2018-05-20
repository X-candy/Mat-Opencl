EXE         = umat2mat
CXX_FLAGS   = -O3
CL6X_FLAGS  = -O3
CLOCL_FLAGS = 

CXX_FLAGS	+= `pkg-config --cflags --libs opencv`

include make.inc

$(EXE): umat2mat_main.o stereoBM.o
		@$(CXX) $(CXX_FLAGS) umat2mat_main.o  stereoBM.o $(LD_FLAGS) $(LIBS) -o $@
