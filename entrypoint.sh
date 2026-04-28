#!/bin/bash

# Ensure output directory exists
mkdir -p /workspace/docker_out/Results

# Now run inference
exec "$@"

