CC = g++
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result
LFLAGS = -lgsl -lm -lgslcblas
INCLUDES = -I/usr/local/include -I../ple/eigen-3.2.5
LIBS = -L/usr/local/lib

pte line: ransampl.o linelib.o pte.o line.o
	$(CC) $(CFLAGS) -o pte ransampl.o linelib.o pte.o $(INCLUDES) $(LIBS) $(LFLAGS)
	$(CC) $(CFLAGS) -o line ransampl.o linelib.o line.o $(INCLUDES) $(LIBS) $(LFLAGS)

ransampl.o : ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS) $(LFLAGS)

linelib.o : linelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c linelib.cpp $(INCLUDES) $(LIBS) $(LFLAGS)

pte.o line.o: pte.cpp line.cpp linelib.o
	$(CC) $(CFLAGS) -c pte.cpp $(INCLUDES) $(LIBS) $(LFLAGS)
	$(CC) $(CFLAGS) -c line.cpp $(INCLUDES) $(LIBS) $(LFLAGS)

clean :
	rm -rf *.o pte line
