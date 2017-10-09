CC = gcc-7
SPLATT_PATH = /home/jli/Work/FillEstimation/splatt-install
CFLAGS = -std=c99 -lm -march=native -O3 -g3 `pkg-config --cflags glib-2.0` 
LDFLAGS = -I/usr/local/include -L/usr/local/lib -I$(SPLATT_PATH)/include -L$(SPLATT_PATH)/lib/
LDFLAGS2 = -lgsl -lgslcblas `pkg-config --libs glib-2.0` -lm $(SPLATT_PATH)/lib/libsplatt.a -fopenmp

all: csf_load
clean:
	rm -rf csf_load *.o *.dSYM *.trace *.pyc

csf_load: csf_load.o
			@$(CC) $(LDFLAGS) $^ -o $@ $(LDFLAGS2)
			@echo "\033[0;32m[66%]\033[0m Compiling asx"