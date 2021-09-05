IMAGE=senet
run:
	docker build -t ${IMAGE} .
in:
	NV_GPU=0 nvidia-docker run --runtime=nvidia -v /home/hagi:/home/app --rm -it ${IMAGE}:latest bash
