CC = g++
CFLAGS = -lm -pthread -Ofast -march=native -Wall -funroll-loops -Wno-unused-result -lgsl -lm -lgslcblas
LIBS = -L/usr/local/lib

deepwalk-hete deepwalk-bipa: deepwalk-hete.o deepwalk-bipa.o
	$(CC) $(CFLAGS) -o deepwalk-hete deepwalk-hete.o $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -o deepwalk-bipa deepwalk-bipa.o $(INCLUDES) $(LIBS)

deepwalk-hete.o deepwalk-bipa.o: deepwalk-hete.cpp deepwalk-bipa.cpp
	$(CC) $(CFLAGS) -c deepwalk-hete.cpp $(INCLUDES) $(LIBS)
	$(CC) $(CFLAGS) -c deepwalk-bipa.cpp $(INCLUDES) $(LIBS)

clean:
	rm -rf *.o deepwalk-hete deepwalk-bipa