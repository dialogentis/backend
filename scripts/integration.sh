#!/bin/sh
set -eu
cleanup() {
  docker compose -p dialogentis-test -f compose.yaml -f compose.test.yaml --profile test down -v
}
trap cleanup EXIT
docker compose -p dialogentis-test -f compose.yaml -f compose.test.yaml --profile test build tests migrate
docker compose -p dialogentis-test -f compose.yaml -f compose.test.yaml --profile test run --rm tests
