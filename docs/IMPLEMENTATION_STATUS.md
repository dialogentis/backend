# Реализация Dialogentis

Дата: 2026-10-30. Первый серверный инкремент. Продукт целиком ещё не реализован.

## S01 · Основание репозиториев

- stage: S01
- status: completed
- started_at: 2026-10-30
- completed_at: 2026-10-30
- changed_paths: pyproject.toml, uv.lock, .python-version, Dockerfile, compose.yaml, compose.dev.yaml, src/dialogentis/{api,config,main}.py, src/dialogentis/{nlu,jobs}, tests, scripts, docs/product, .github/workflows/backend.yml; frontend/{Dockerfile,Caddyfile,.dockerignore,.node-version,vite.config.ts}.
- checks:
  - passed: uv sync --locked (CPython 3.14.6), Ruff format/check, ty.
  - passed: unit tests; Linux production dependency install через uv --no-dev.
  - passed: Docker build Python 3.14.6/Linux arm64; intent train/predict, CRF train/tag, RE2 rejection внутри pytest.
  - passed: реальный RabbitMQ + Taskiq worker, manual ack и publisher confirms.
  - passed: Node 24 frontend build/typecheck; npm audit сообщил 0 vulnerabilities при npm ci.
  - passed: compose config и dev override config; HTTP 200 frontend SPA/live/ready через Caddy на 18080.
  - passed: отдельный чистый Compose-проект — 8 тестов; init SQL и автоматические права/cookie RabbitMQ работают. Development-образы собраны; проверка reload на изменении файла отдельно не выполнялась.
- decisions: D001–D005 в DECISIONS.md.
- blockers: нет.
- next_step: S02.

## S02 · Модели и tenant context

- stage: S02
- status: completed
- started_at: 2026-10-30
- completed_at: 2026-10-30
- changed_paths: src/dialogentis/{models,db,contracts}.py, migrations, alembic.ini, tests/test_tenancy.py, openapi.json.
- checks:
  - passed: миграция 0001, повторный upgrade head, сравнение SQLAlchemy metadata с реальной БД без drift.
  - passed: non-superuser app-role не владеет таблицами и не имеет BYPASSRLS.
  - passed: без tenant context нет чтения, чужой UUID не читается, cross-tenant insert отклонён.
  - passed: составной FK не допускает draft → чужой agent; tenant context не сохраняется после commit/rollback и возврата соединения в pool.
  - passed: общий error envelope/request_id; InputModel запрещает extra fields; CursorPage и форматирование ETag.
- decisions: D004. Глобальные auth/membership операции получат узкие сервисы при реализации S03; произвольного отключения RLS нет.
- blockers: нет.
- next_step: S03.

## S03 · Авторизация и организации

- stage: S03
- status: in_progress
- started_at: 2026-10-30
- completed_at: —
- changed_paths: src/dialogentis/auth, dependencies.py, organizations.py, api_errors.py, api.py, contracts.py; migrations/0002 и 0003; tests/test_auth.py; compose.yaml/test.yaml/browser-test.yaml и scripts/integration.sh; openapi.json; frontend/src/{api,Session,Root,pages}, scripts/smoke.mjs.
- checks:
  - passed: Ruff format/check, ty.
  - passed: make integration — 15 pytest-проверок в Linux на полностью пустых PostgreSQL/RabbitMQ/Mailpit; тестовый Compose-проект удалён, development volumes сохранены. Проверены миграции 0001 → 0003 с нуля, без drift.
  - passed: реальный HTTP через Caddy → регистрация 202 → отдельный mailer доставляет письмо → подтверждение 204 → вход/me → выход 204. Никакие токены/пароли в отчёт не выводились.
  - passed: CSRF и Origin, нейтральное восстановление, одноразовые/истёкшие ссылки, сохранение неуспешных входов и 429 с Retry-After, cookie HttpOnly/SameSite, абсолютное истечение и отзыв сессий после смены пароля.
  - passed: RLS на приглашениях/аудите, скрытие чужой организации 404, роли viewer/editor/admin/owner, немедленная смена прав, принятие/отзыв/перевыпуск приглашения, запрет ухода владельца и передача владения с паролем. Администратор не может отозвать, переотправить или заменить приглашение другого администратора.
  - passed: матрица прав в коде; параметризованная проверка разных ролей пользователя в двух организациях, немедленного отзыва и сохранения доступа к другой организации.
  - passed: браузерный сценарий на изолированном Compose-стенде — регистрация, доставка и подтверждение почты, вход, создание пространства, мобильная навигация, выход и восстановление пароля; приглашение, повторная отправка, отзыв, принятие другим пользователем, смена роли, передача владения и выход из пространства. Проверен отказ редактору в управлении участниками.
  - passed: два независимых браузерных сеанса, переключение между двумя организациями и отказ чужому пользователю по прямому URL и API.
  - not_run: production SMTP/TLS.
- decisions: D006–D007. Проект протокола удаления — ORGANIZATION_DELETION.md; API и очистка ещё не реализованы.
- blockers: внешних нет.
- next_step: завершить диспетчер/worker, восстановление заданий и обработчик удаления до открытия API удаления; затем завершить остальные условия S03 и перейти к CRUD агентов/lease на S04.

## S06 · Инфраструктура фоновых заданий (начата досрочно для S03)

- stage: S06
- status: in_progress
- changed_paths: migrations/0004_jobs_outbox.py, src/dialogentis/jobs/{models,enqueue,dispatcher,broker,tasks}.py, scripts/init-dispatcher.sh, compose.yaml, tests/test_jobs.py.
- passed: миграции 0001 → 0004 с нуля, повторный upgrade и отсутствие drift; транзакционная запись Job/outbox и rollback; изоляция RLS и составной FK.
- passed: ограниченная отдельная роль диспетчера читает только служебный outbox, не читает Job или организации и не меняет tenant ID; выбор готовых записей с блокировкой; реальная публикация в RabbitMQ с publisher confirms, rollback после потери подтверждения и последующий повтор с теми же идентификаторами.
- passed: 23 интеграционные проверки; реальный процесс dispatcher → worker; терминальный статус фиксируется до manual ack, повторная доставка не увеличивает attempts; отменённое задание не исполняется; сбой подключения к БД и отложенный сбой commit приводят к повторной доставке. Ошибка ack после commit проверена с подменой канала. Неверный формат, неизвестная задача, некорректный UUID, подмена режима ack и контекста направляются в dead-letter, не блокируя следующие сообщения.
- not_implemented: прикладные обработчики, восстановление зависших задач и доставок из dead-letter, bounded retry/backoff, подключение процессов к Compose, обучение, артефакты и пользовательские фоновые операции. Worker сейчас только завершает отменённые и неизвестные задания.
- next_step: реализовать восстановление зависших заданий и bounded retry до подключения процессов и открытия удаления организации.

## S04–S05, S07–S16

status: pending. Обучение, граф/runtime, публикация, виджет, аналитика и эксплуатационная приёмка пока не реализованы. Дизайн-прототип агентов пока не подключён к рабочему кабинету.

## Известные ограничения

- В тестах есть предупреждение upstream Starlette о будущем переходе TestClient с httpx на httpx2; ошибок исполнения нет.
- NLU smoke подтверждает совместимость библиотек, не качество моделей AC-05.
- Восстановление зависших заданий и сообщений из dead-letter ещё не реализовано; на текущем этапе пользовательские задания не включены.
- HTTPS, резервные копии/restore, нагрузочные испытания и полный приёмочный отчёт ещё предстоят. Нельзя считать наличие контейнеров завершением этих требований.
