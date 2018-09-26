DIRS = ptmpls W01Exercise W02Exercise

.PHONY: all
all: recursive environment.yml

environment.yml:
	conda env export > environment.yml

.PHONY: recursive
recursive:
	for DIR in $(DIRS);\
	do\
		$(MAKE) -C $${DIR};\
	done

.PHONY: environment
environment: environment.yml
	conda env create -f envorinment.yml
