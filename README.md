# Dialogentis backend

FastAPI, Python 3.14.6, SQLAlchemy/Alembic, PostgreSQL 18, Taskiq/RabbitMQ. Зависимости устанавливаются через uv по `uv.lock`; Ruff и ty используются для проверок. Отдельные Git-репозитории backend и frontend должны находиться рядом.

Реализованы основание сервера, модели/RLS и API авторизации/организаций: регистрация, подтверждение почты, вход/выход, восстановление пароля, сессии, CSRF, приглашения, роли и передача владения. В интерфейсе подключены авторизация, рабочие пространства и управление участниками. Бизнес-API агентов ещё предстоит реализовать.

## Первый запуск

Требуются Docker с Compose 2.24.4+ и uv 0.11.29. Все команды выполняются в backend.

```sh
uv sync --locked
uv run --locked python scripts/init_env.py
docker compose --profile dev up --build -d
```

Генератор создаёт `.env` с непредсказуемыми паролями и ключами (права 0600), не выводит значения и отказывается перезаписывать существующий файл. `.env.example` описывает параметры. Не заменяйте существующий `.env` без согласованной смены паролей в БД/broker: init SQL выполняется только на новом volume.

- Кабинет и API через Caddy: http://localhost:18080
- OpenAPI UI: http://localhost:18080/api/docs
- Liveness: http://localhost:18080/api/health/live
- Readiness (реальные подключения к БД и RabbitMQ): http://localhost:18080/api/health/ready
- API напрямую для разработки: http://localhost:18000
- Почтовый интерфейс Mailpit: http://localhost:18025

Миграции выполняются отдельным одноразовым сервисом до запуска API, не каждым worker. Приложение работает под `dialogentis_app`, миграции — под `dialogentis_migrator`; bootstrap superuser не передаётся приложению. Одноразовый `dispatcher-init` создаёт ограниченную роль диспетчера и её случайный пароль в отдельном Docker volume, недоступном API; он также поддерживает существующую базу при обновлении. Для миграций вне Compose роль диспетчера необходимо предварительно создать от имени администратора БД. PostgreSQL и RabbitMQ не публикуют порты на хост. HTTP-порты привязаны только к loopback. Production-образы запускают приложения без root; HTTPS и внешняя конфигурация production потребуют следующего эксплуатационного этапа.

## Разработка с hot reload

```sh
docker compose -f compose.yaml -f compose.dev.yaml --profile dev up --build -d
```

Vite доступен на http://localhost:5180 (чтобы не занимать уже используемый 5178); API — на 18000. Исходники подключены read-only, зависимости находятся в образах. `/api` Vite проксирует к API. Для обычного запуска интерфейса вне Docker используется `npm run dev` в соседнем frontend, proxy по умолчанию смотрит на 18000.

## Проверки и миграции

```sh
make check
make test
make integration
docker compose run --rm migrate
uv run --locked python scripts/export_openapi.py
```

`make integration` создаёт отдельный Compose-проект `dialogentis-test` с пустыми PostgreSQL/RabbitMQ/Mailpit, выполняет тесты и удаляет только его контейнеры и volumes. Development-данные не затрагиваются. Проверяются train/predict intent/CRF, безопасное отклонение неподдерживаемых RE2 выражений, API, ручной ack Taskiq с publisher confirms, повторный запуск миграций и отсутствие drift, RLS/составные FK/возврат соединения в pool. Без `RUN_INTEGRATION=1` интеграционные тесты явно пропускаются, а не подменяются SQLite. Локальный `.env` содержит Docker DNS-имена, поэтому интеграционные проверки выполняются в Compose.

Добавлены tenant-таблицы Job/outbox, диспетчер с ограниченной ролью, worker и периодическая сверка. Compose запускает их отдельными сервисами. Доставка, повтор сообщения после ошибки фиксации БД, дедупликация запуска, восстановление и отправка некорректных сообщений в dead-letter проверены на PostgreSQL и RabbitMQ. Прикладных обработчиков пока нет, поэтому пользовательские фоновые функции недоступны. Рабочие очереди обучения относятся к S07.

CI-команды описаны в Makefile и `.github/workflows/backend.yml`. `openapi.json` генерируется из приложения. Документы продукта версионируются в `docs/product`; актуальный журнал и решения — в `docs`.

## Авторизация и почта

Контракты доступны в `/api/docs`, все бизнес-маршруты имеют префикс `/api/v1`. Для POST/PATCH/DELETE нужен корректный `Origin`, для запросов с сеансом — также `X-CSRF-Token`, полученный при login/me. Cookie хранит только непрозрачный токен, БД — его SHA-256. Пароли хешируются Argon2id. В development разрешены известные локальные origins кабинета; production принимает только `PUBLIC_BASE_URL`.

Подтверждение почты (24 часа), восстановление (30 минут) и приглашения (7 дней) одноразовые. В БД хранятся хеши токенов. Письмо с исходной ссылкой до отправки хранится зашифрованным в email outbox; процесс `mailer` отправляет его через SMTP и удаляет payload. Запускается вместе с Compose или отдельно: `docker compose --profile dev up -d mailer mailpit`. При production SMTP применяется STARTTLS. Внешняя почта в локальных тестах не используется.

Доставка SMTP допускает повтор письма, если процесс завершится после отправки до фиксации БД; одноразовая ссылка остаётся безопасной. Перед сменой ENCRYPTION_KEY следует опустошить email outbox: чтение старых ключей пока не реализовано.

Uvicorn доверяет forwarded headers внутри стенда; Caddy перезаписывает входящие X-Forwarded-For. Порт API привязан к loopback. При внешнем production-развёртывании не публикуйте API в обход доверенного proxy.

## Остановка сервисов

```sh
docker compose --profile dev --profile test down
```

Volumes сохраняются. Удаление volumes не является обычным обновлением: там находится БД. Backup/restore и production runbook ещё не реализованы и относятся к S15–S16.
