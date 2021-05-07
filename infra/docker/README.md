
Commands to build and push to GCR repository:

1. `docker build -t speech-emotion-modelling:latest -f infra/docker/Dockerfile .`

2. `docker tag speech-emotion-modelling:latest eu.gcr.io/wewyse-centralesupelec-ftv/ser-modelling:latest`

3. `docker push eu.gcr.io/wewyse-centralesupelec-ftv/ser-modelling:latest`
