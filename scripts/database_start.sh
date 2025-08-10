#!/bin/bash

docker pull qdrant/qdrant

HOST_PORT=7001
QDRANT_PORT=6333
SECOND_PORT=6334
LOCAL_STORAGE_PATH=$(pwd)/collections
CONTAINER_STORAGE_PATH=/qdrant/storage


mkdir -p "$LOCAL_STORAGE_PATH"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
  echo "Error: Docker is not running. Please start Docker and try again."
  exit 1
fi

docker run -d \
    -p $HOST_PORT:$QDRANT_PORT \
    -p $SECOND_PORT:$SECOND_PORT \
    -v $LOCAL_STORAGE_PATH:$CONTAINER_STORAGE_PATH:z \
    qdrant/qdrant

echo "Qdrant container started on port $HOST_PORT"