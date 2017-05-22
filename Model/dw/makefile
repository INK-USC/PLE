CC = g++
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result
LFLAGS = -lgsl -lm -lgslcblas
LIBS = -L/usr/local/lib

deepwalk-hete: deepwalk-hete.o
	$(CC) $(CFLAGS) -o deepwalk-hete deepwalk-hete.o $(INCLUDES) $(LIBS) $(LFLAGS)

deepwalk-hete.o: deepwalk-hete.cpp
	$(CC) $(CFLAGS) -c deepwalk-hete.cpp $(INCLUDES) $(LIBS) $(LFLAGS)

clean:
	rm -rf *.o deepwalk-hete