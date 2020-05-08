dataset = b
log_dir = logs
log_prefix = gcn
score_file = $(log_dir)/scores_$(log_prefix).txt
log_file = $(log_dir)/logs_$(log_prefix).txt

datasets = a b c d e
dirs:=$(shell ls data/ )
gpuid:=$(shell gpustat | awk '{if (NR>1) print NR-2, $$11}' | sort -k 2 -n | awk '{print $$1}' | sed -n '1p')

run:
	python run_local_test.py --dataset_dir=data/$(dataset) --code_dir=code_submission
all:
	echo $(shell date "+%Y-%m-%d %H:%M:%S") > $(score_file) 
	echo $(datasets) >> $(score_file)
	for d in `echo $(datasets)`; \
	do \
		make run dataset=$$d; \
		cat scoring_output/scores.txt >> $(score_file); \
	done
zip:
	cd code_submission/ && zip submit.zip *.py
log:
	awk '{if (NR>2) {print $$2} }' $(score_file) | sed -n '{N;s/\n/\t/p}'
total_all:
	echo $(shell date "+%Y-%m-%d %H:%M:%S") > $(score_file) 
	echo $(dirs) >> $(score_file)
	for d in `echo $(dirs)`; \
	do \
		make run dataset=$$d; \
		cat scoring_output/scores.txt >> $(score_file); \
	done
docker:
	docker run --gpus=0 -it --rm -v "$$(pwd):/app/autograph" -w /app/autograph -e NVIDIA_VISIBLE_DEVICES=$(gpuid) nehzux/kddcup2020:v2
clean:
	rm */.*.swp*
