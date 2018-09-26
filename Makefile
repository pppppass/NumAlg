DIRS = ptmpls W01Exercise

.PHONY: all
all: recursive

.PHONY: recursive
recursive:
	for DIR in $(DIRS);\
	do\
		$(MAKE) -C $${DIR};\
	done
