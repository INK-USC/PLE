CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
INCLUDES = -I/usr/local/include -I./eigen-3.2.5
LIBS = -L/usr/local/lib


pte-hete-pretrain pte-hete pte-bipa: ransampl.o linelib.o pte-hete.o pte-bipa.o pte-hete-pretrain.o
	$(CC) $(CFLAGS) -o pte-hete ransampl.o linelib.o pte-hete.o $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -o pte-hete-pretrain ransampl.o linelib.o pte-hete-pretrain.o $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -o pte-bipa ransampl.o linelib.o pte-bipa.o $(INCLUDES) $(LIBS)

ransampl.o : ransampl.c
	$(CC) $(CFLAGS) -c ransampl.c $(INCLUDES) $(LIBS)

linelib.o : linelib.cpp ransampl.h
	$(CC) $(CFLAGS) -c linelib.cpp $(INCLUDES) $(LIBS)

pte-hete-pretrain.o pte-hete.o pte-bipa.o: pte-hete-pretrain.cpp pte-hete.cpp pte-bipa.cpp linelib.o
	$(CC) $(CFLAGS) -c pte-hete-pretrain.cpp $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -c pte-hete.cpp $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -c pte-bipa.cpp $(INCLUDES) $(LIBS)

clean :
	rm -rf *.o pte-hete-pretrain pte-hete pte-bipa
