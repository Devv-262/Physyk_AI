#!/bin/bash
docker run --entrypoint bash --rm nvcr.io/nvidia/isaac-sim:6.0.1 -c "ls -la /isaac-sim/*.sh"
