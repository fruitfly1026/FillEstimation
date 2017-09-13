CC = gcc-7
CFLAGS = -std=c99 -lm -march=native -O3 -g3 `pkg-config --cflags glib-2.0`
LDFLAGS = -I/usr/local/include -L/usr/local/lib 
LDFLAGS2 = -lgsl -lgslcblas `pkg-config --libs glib-2.0` -lm

all: reference asx oski
clean:
	rm -rf reference asx oski *.o *.dSYM *.trace *.pyc

reference: run.c test.o reference.o util.o
			$(CC) $(LDFLAGS) run.c test.o reference.o util.o -o reference $(LDFLAGS2)

asx: run.c test.o asx.o util.o
			$(CC) $(LDFLAGS) run.c test.o asx.o util.o -o asx $(LDFLAGS2)


oski: run.c test.o oski.o util.o
			$(CC) $(LDFLAGS) run.c test.o oski.o util.o -o oski $(LDFLAGS2)
