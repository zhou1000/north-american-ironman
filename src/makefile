CFLAG=-std=c++11 -c
all: main

main: main.o FTRL.o
	g++ main.o FTRL.o -o main

main.o: main.cpp FTRL.h
	g++ $(CFLAG) main.cpp

FTRL.o: FTRL.cpp FTRL.h
	g++ $(CFLAG) FTRL.cpp

clean:
	rm -rf *o
