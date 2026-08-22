#!/bin/bash
docker run --entrypoint bash --rm nvcr.io/nvidia/isaac-sim:6.0.1 -c "cat /isaac-sim/isaac-sim.streaming.sh"
