CC = gcc
CFLAGS = -g
LDFLAGS =
LIBS = -lm

PROGS = create_xdata_best2 trainclassify_best2 classify_best2 classify_best2fast classify_mxcode
PROGS_SRCS = $(PROGS:=.c)

ROUTINES_SRCS = routines_taxonomy.c routines_sequence.c routines_model.c
ROUTINES_OBJS = $(ROUTINES_SRCS:.c=.o)

DEFS = defs.h

all: $(PROGS)

objs: $(PROGS_SRCS:.c=.o) $(ROUTINES_OBJS)

%.o: %.c $(DEFS)
	$(CC) $(CFLAGS) -c $< -o $@

$(PROGS) : %: %.o $(ROUTINES_OBJS) $(DEFS)
	$(CC) $(CFLAGS) $(LDFLAGS) $< -o $@ $(ROUTINES_OBJS) $(LIBS)

clean: 
	rm *.o
	rm $(PROGS)
