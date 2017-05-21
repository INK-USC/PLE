CC = g++
CFLAGS = -lm -pthread -O2 -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
INCLUDES = -I/usr/local/include -I./eigen-3.2.5
LIBS = -L/usr/local/lib


hple-bs hple-corrKB hple-corrH hple-noCorr: ransampl.o hplelib.o hple-bs.o hple-corrKB.o hple-corrH.o hple-noCorr.o
	$(CC) $(CFLAGS) -o hple-bs ransampl.o hplelib.o hple-bs.o $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -o hple-corrKB ransampl.o hplelib.o hple-corrKB.o $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -o hple-corrH ransampl.o hplelib.o hple-corrH.o $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -o hple-noCorr ransampl.o hplelib.o hple-noCorr.o $(INCLUDES) $(LIBS)

ransampl.o: ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS)

hplelib.o: hplelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c hplelib.cpp $(INCLUDES) $(LIBS)

hple-bs.o hple-corrKB.o hple-corrH.o hple-noCorr.o: hple-bs.cpp hple-corrKB.cpp hple-corrH.cpp hple-noCorr.cpp hplelib.o
	$(CC) $(CFLAGS) -c hple-bs.cpp $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -c hple-corrKB.cpp $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -c hple-corrH.cpp $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -c hple-noCorr.cpp $(INCLUDES) $(LIBS)

clean:
	rm -rf *.o hple-bs hple-corrKB hple-corrH hple-noCorr
