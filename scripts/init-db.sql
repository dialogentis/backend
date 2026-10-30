\getenv app_password APP_DB_PASSWORD
\getenv migration_password MIGRATION_DB_PASSWORD
CREATE ROLE dialogentis_migrator LOGIN PASSWORD :'migration_password' NOSUPERUSER NOBYPASSRLS;
CREATE ROLE dialogentis_app LOGIN PASSWORD :'app_password' NOSUPERUSER NOBYPASSRLS;
REVOKE CREATE ON SCHEMA public FROM PUBLIC;
GRANT USAGE, CREATE ON SCHEMA public TO dialogentis_migrator;
GRANT USAGE ON SCHEMA public TO dialogentis_app;
ALTER DEFAULT PRIVILEGES FOR ROLE dialogentis_migrator IN SCHEMA public GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO dialogentis_app;
