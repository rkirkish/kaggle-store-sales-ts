TAG=kaggle-forecast-trainer
HISTORY_FILE=.image_bash_history
ROOT_DIR := $(abspath $(dir $(abspath $(dir $$PWD)/../../)))

run:
	touch $(PWD)/$(HISTORY_FILE)
	docker run -it \
                   --rm \
                   --entrypoint=bash \
                   -e HISTFILE=/root/$(HISTORY_FILE) \
                   -e GEN_USER \
                   -v $(PWD)/$(HISTORY_FILE):/root/$(HISTORY_FILE) \
                   -v /var/run/docker.sock:/var/run/docker.sock \
                   -v $(HOME)/.aws/:/root/.aws/:rw \
                   -v $(PWD)/:/app/:rw \
                   -p 8888:8888 \
                   $(TAG)
build:
	docker build -t $(TAG) .
