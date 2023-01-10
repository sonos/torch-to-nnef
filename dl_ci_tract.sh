#!/bin/bash
TRACT_VERSION=$1
mkdir -p .tract && \
cd .tract && \
wget --quiet "https://github.com/sonos/tract/releases/download/${TRACT_VERSION}/tract-x86_64-unknown-linux-musl-${TRACT_VERSION}.tgz" && \
tar -xvzf tract-x86_64-unknown-linux-musl-${TRACT_VERSION}.tgz
