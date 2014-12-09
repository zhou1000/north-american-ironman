CXX = g++
CXXFLAGS = -Wall -Wconversion -O2 -fPIC -std=c++0x -march=native -fopenmp

SRCDIR = src

FILES = DataWriter.cpp DataReader.cpp GBDT.cpp
SRCS = $(FILES:%.cpp=$(SRCDIR)/%.cpp)
HEADERS = $(FILES:%.cpp=$(SRCDIR)/%.h)
OBJS = $(FILES:%.cpp=$(SRCDIR)/%.o)
MAIN = main

all: $(MAIN)

$(SRCDIR)/%.o: $(SRCDIR)/%.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

main: $(SRCDIR)/main.cpp $(OBJS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(OBJS)

clean:
	rm -f $(OBJS)

# -o $@ says to put the output of the compilation in the file named on the left side of the :
# $^ the left and right sides of the :
# $< is the first item in the dependencies list
# % (rule character) is a wild card, matching zero or more characters
