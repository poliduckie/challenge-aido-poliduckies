ARG DOCKER_REGISTRY=docker.io
ARG ARCH=amd64
ARG MAJOR=daffy
ARG BASE_TAG=${MAJOR}-${ARCH}

FROM ${DOCKER_REGISTRY}/duckietown/dt-machine-learning-base-environment:${BASE_TAG}

ARG PIP_INDEX_URL="https://pypi.org/simple"
ENV PIP_INDEX_URL=${PIP_INDEX_URL}
RUN echo PIP_INDEX_URL=${PIP_INDEX_URL}

COPY requirements.* ./
RUN cat requirements.* > .requirements.txt
RUN python3 -m pip install -r .requirements.txt

# let's copy all our solution files to our workspace
WORKDIR /submission
COPY solution.py /submission
COPY models /submission/models
COPY wrappers.py /submission

CMD ["python3", "solution.py"]
