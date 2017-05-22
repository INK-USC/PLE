CC = g++
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result
LFLAGS = -lgsl -lm -lgslcblas
INCLUDES = -I/usr/local/include -I../eigen-3.2.5
LIBS = -L/usr/local/lib

hple: ransampl.o hplelib.o hple.o
	$(CC) $(CFLAGS) -o hple ransampl.o hplelib.o hple.o $(INCLUDES) $(LIBS) $(LFLAGS)

ransampl.o: ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS) $(LFLAGS)

hplelib.o: hplelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c hplelib.cpp $(INCLUDES) $(LIBS) $(LFLAGS)

hple.o: hple.cpp hplelib.o
	$(CC) $(CFLAGS) -c hple.cpp $(INCLUDES) $(LIBS) $(LFLAGS)

clean:
	rm -rf *.o hple