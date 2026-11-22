#!/bin/sh
set -eu

cd "$(dirname "$0")/.."
backend_dir=$PWD
: "${CHROME_PATH:?Set CHROME_PATH to the Chrome or Chromium executable}"
case "$CHROME_PATH" in
    /*) ;;
    *) CHROME_PATH="$(cd "$(dirname "$CHROME_PATH")" && pwd)/$(basename "$CHROME_PATH")" ;;
esac
test -x "$CHROME_PATH"
export CHROME_PATH
export BASE_URL=http://localhost:18180 PUBLIC_BASE_URL=http://localhost:18180
export MAILPIT_URL=http://localhost:18125

(
    cd ../frontend
    npm ci
    npm run api:generate -- --check
    npm run typecheck
    npm run test:unit
    npm run build
)

test_dir=$(mktemp -d)
project="dialogentis-acceptance-$$"
compose() {
    docker compose --project-directory "$backend_dir" --env-file "$test_dir/env" \
        -p "$project" -f compose.yaml -f compose.browser-test.yaml --profile dev "$@"
}
cleanup() {
    status=$?
    trap - EXIT
    cd "$backend_dir"
    if [ -f "$test_dir/env" ]; then
        if ! compose down --volumes; then
            printf 'Could not remove Compose project %s. Its env file remains at %s/env.\n' \
                "$project" "$test_dir" >&2
            exit 1
        fi
        rm -f "$test_dir/env"
    fi
    rmdir "$test_dir"
    exit "$status"
}
trap cleanup EXIT
uv run --locked python scripts/init_env.py --output "$test_dir/env"
compose up --build --wait --wait-timeout 120 frontend mailer mailpit dispatcher worker \
    training-worker maintenance-worker scheduler runtime-watchdog

cd ../frontend
npm run test:smoke
npm run test:smoke:http
node scripts/privacy-smoke.mjs
RELEASE_ONLY=1 npm run test:smoke
GRAPH_ONLY=1 npm run test:smoke
