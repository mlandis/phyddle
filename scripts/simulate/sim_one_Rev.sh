#!/usr/bin/env sh

TMP_FN=$1
ARGS="tmp_fn=\"${TMP_FN}\""
SRC_CMD="source(\"sim_one.Rev\")"

echo "${ARGS}; ${SRC_CMD}" | rb
