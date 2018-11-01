TBLOBJS = Table11.tbl Table12.tbl Table21.tbl Table22.tbl Table31.tbl Table32.tbl
RUNOBJS = Problem.run

.PHONY: all
all: Answer.pdf

Answer.pdf: Answer.tex $(TBLOBJS) | Answer.aux
	lualatex Answer.tex

Answer.aux: Answer.tex
	lualatex Answer.tex

$(TBLOBJS): $(wildcard $(RUNOBJS))
	touch $@

.PRECIOUS: %.py
%.py: %.ipynb
	jupyter nbconvert --to script $*

%.run: %.py utils/lu.py utils/ortho.py
	python $<
	touch $@

.PHONY: report
report: Answer.pdf

.PHONY: run
run: $(RUNOBJS)
