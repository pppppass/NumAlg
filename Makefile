DIRS = $(shell ls -d */ | grep -v ptmpls)

.PHONY: all
all: hardware.txt environment.yml recursive

hardware.txt:
	echo 'lscpu:' > hardware.txt
	lscpu >> hardware.txt
	echo 'lsmem:' >> hardware.txt
	lsmem >> hardware.txt
	echo 'uname -a:' >> hardware.txt
	uname -a | awk '$$2="********"' >> hardware.txt

environment.yml:
	conda env export | grep -v prefix > environment.yml

.PHONY: recursive
recursive: template
	for DIR in $(DIRS);\
	do\
		$(MAKE) -C $${DIR};\
	done

.PHONY: template
template:
	$(MAKE) -C ptmpls

.PHONY: environment
environment: environment.yml
	conda env create -f environment.yml
