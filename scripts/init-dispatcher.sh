#!/bin/sh
set -eu
umask 077
secret=/run/dialogentis-dispatcher/password
if [ ! -s "$secret" ]; then
  od -An -N32 -tx1 /dev/urandom | tr -d ' \n' > "$secret.tmp"
  mv "$secret.tmp" "$secret"
fi
chmod 600 "$secret"
chown 10001:10001 "$secret"
export DISPATCHER_PASSWORD="$(cat "$secret")"
if ! psql -X -q -v ON_ERROR_STOP=1 -h postgres -U dialogentis_bootstrap -d dialogentis >/dev/null 2>&1 <<'SQL'
\getenv dispatcher_password DISPATCHER_PASSWORD
SELECT format('CREATE ROLE dialogentis_dispatcher LOGIN PASSWORD %L NOSUPERUSER NOBYPASSRLS', :'dispatcher_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'dialogentis_dispatcher')
\gexec
SELECT format('ALTER ROLE dialogentis_dispatcher LOGIN PASSWORD %L NOSUPERUSER NOBYPASSRLS', :'dispatcher_password')
\gexec
SQL
then
  echo 'Cannot provision the dispatcher database role.' >&2
  exit 1
fi
