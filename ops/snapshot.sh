#!/usr/bin/env bash
set -eo pipefail
umask 077

fail() {
  printf '%s\n' "$*" >&2
  exit 1
}

if [ "$#" -ne 5 ]; then
  fail "Usage: bash ops/snapshot.sh backup ENV PROJECT BACKUP_DIRECTORY RECIPIENTS_FILE
       bash ops/snapshot.sh restore ENV PROJECT SNAPSHOT_DIRECTORY IDENTITY_FILE"
fi

operation=$1
env_file=$2
project=$3
directory=$4
key_file=$5
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export COMPOSE_FILE="${COMPOSE_FILE:-$repo/compose.yaml:$repo/compose.production.yaml}"

[ -r "$env_file" ] || fail "Cannot read the Compose environment file."
[ -r "$key_file" ] || fail "Cannot read the age key file."
[ -d "$directory" ] || fail "Snapshot directory does not exist."
command -v age >/dev/null || fail "Install age on the Docker host."

case "$project" in
  ''|*[!a-z0-9_-]*) fail "Invalid Compose project name." ;;
esac
lock="${TMPDIR:-/tmp}/dialogentis-$project.snapshot.lock"
mkdir -- "$lock" || fail "Another snapshot operation holds $lock."
trap 'rmdir -- "$lock"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

compose() {
  docker compose --project-directory "$repo" --env-file "$env_file" -p "$project" "$@"
}

postgres() {
  compose exec -T postgres "$@"
}

artifacts() {
  compose run --pull never --no-deps --rm -T deletion-replay "$@"
}

backup() {
  local stage timestamp running service target
  local -a services=()
  running=$(compose ps --status running --services)
  printf '%s\n' "$running" | grep -Fxq postgres || fail "PostgreSQL must be running."
  postgres psql -XAt -v ON_ERROR_STOP=1 -U dialogentis_bootstrap -d dialogentis \
    -c 'SELECT version_num FROM alembic_version' >/dev/null
  while IFS= read -r service; do
    case "$service" in
      frontend|api|mailer|dispatcher|worker|training-worker|maintenance-worker|runtime-watchdog|scheduler)
        services+=("$service")
        ;;
    esac
  done <<< "$running"
  timestamp=$(date -u +%Y%m%dT%H%M%SZ)
  stage=$(mktemp -d "$directory/.dialogentis-partial.XXXXXXXX")

  finish_backup() {
    local status=$?
    if [ "${#services[@]}" -gt 0 ]; then
      compose start "${services[@]}" >&2 || status=1
    fi
    if [ -n "$stage" ]; then
      rm -rf -- "$stage" || status=1
    fi
    rmdir -- "$lock" || status=1
    exit "$status"
  }
  trap finish_backup EXIT

  if [ "${#services[@]}" -gt 0 ]; then
    compose stop "${services[@]}" >&2
  fi
  postgres pg_dump --format=custom --no-password --username=dialogentis_bootstrap dialogentis |
    age --encrypt --recipients-file "$key_file" >"$stage/database.dump.age"
  artifacts tar -C /app/artifacts -cf - . |
    age --encrypt --recipients-file "$key_file" >"$stage/artifacts.tar.age"
  (
    cd -- "$stage"
    sha256sum database.dump.age artifacts.tar.age >SHA256SUMS
  )

  if [ "${#services[@]}" -gt 0 ]; then
    compose start "${services[@]}" >&2
    services=()
  fi
  target="$directory/snapshot-$timestamp"
  [ ! -e "$target" ] || fail "A snapshot for this timestamp already exists."
  mv -- "$stage" "$target"
  stage=''
  find "$directory" -mindepth 1 -maxdepth 1 -type d -name 'snapshot-????????T??????Z' \
    -mmin +10079 -exec rm -rf -- {} +
  printf '%s\n' "$target"
  trap 'rmdir -- "$lock"' EXIT
}

restore() {
  local running empty checksums
  running=$(compose ps --status running --services)
  case "$running" in
    ''|postgres) ;;
    *) fail "Restore requires an offline target with at most PostgreSQL running." ;;
  esac
  checksums=$(cd -- "$directory" && sha256sum database.dump.age artifacts.tar.age)
  [ "$checksums" = "$(cat -- "$directory/SHA256SUMS")" ] || fail "Snapshot checksum mismatch."
  age --decrypt --identity "$key_file" "$directory/database.dump.age" >/dev/null
  age --decrypt --identity "$key_file" "$directory/artifacts.tar.age" >/dev/null

  compose up -d --wait postgres
  empty=$(postgres psql -XAt -v ON_ERROR_STOP=1 -U dialogentis_bootstrap -d dialogentis -c \
    "SELECT NOT EXISTS (SELECT 1 FROM pg_tables WHERE schemaname = 'public')")
  [ "$empty" = t ] || fail "Restore requires an empty dialogentis database."
  artifacts sh -ec 'test -d /app/deletion-journal && test -z "$(find /app/artifacts -mindepth 1 -print -quit)"' ||
    fail "Restore requires an empty artifact volume and the current independent deletion journal."
  compose run --no-deps --rm -T dispatcher-init
  age --decrypt --identity "$key_file" "$directory/database.dump.age" |
    postgres pg_restore --no-password --username=dialogentis_bootstrap --dbname=dialogentis \
      --single-transaction --exit-on-error
  age --decrypt --identity "$key_file" "$directory/artifacts.tar.age" |
    artifacts tar -C /app/artifacts --no-same-owner --same-permissions -xf -
  compose run --pull never --no-deps --rm -T migrate
  artifacts
  compose run --pull never --no-deps --rm -T maintenance-worker \
    /app/.venv/bin/python -m dialogentis.prepare_restore
  printf '%s\n' "Restore prepared. Application services are stopped."
}

case "$operation" in
  backup) backup ;;
  restore) restore ;;
  *) fail "Use backup or restore." ;;
esac
