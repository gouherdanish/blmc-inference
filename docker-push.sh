#!/bin/bash
cd /Users/gouher/Documents/personal/codes/blmc/inference
docker build -t blmc:latest .
docker tag blmc:latest gouherdanishiitkgp/blmc-docker-repo:latest
docker push gouherdanishiitkgp/blmc-docker-repo:latest