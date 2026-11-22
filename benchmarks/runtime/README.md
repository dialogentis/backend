# Приёмка производительности Dialogentis

Профиль проверяет NFR-01 и NFR-02 через публичный HTTP API и API кабинета. Для итоговой приёмки используйте Linux x86-64, 8 vCPU, 16 ГиБ RAM и SSD от 80 ГиБ. Зафиксируйте конфигурацию сервера, Docker Engine, Compose, ревизии обоих репозиториев и ограничения контейнеров. Измерения на другой конфигурации публикуются с указанием стенда.

Нужны соседний checkout frontend, Docker Compose 2.24.4+ и окружение Python из `uv sync --locked`. Выполняйте команды из backend на выделенном стенде со свободными портами 18180 и 18125.

```sh
run_dir=$(mktemp -d /tmp/dialogentis-load.XXXXXX)
uv run --locked python scripts/init_env.py --output "$run_dir/env"
export PUBLIC_BASE_URL=http://localhost:18180

load_compose() {
  docker compose --env-file "$run_dir/env" -p dialogentis-load-profile \
    -f compose.yaml -f compose.browser-test.yaml -f benchmarks/runtime/compose.yaml \
    --profile dev "$@"
}

load_compose up --build -d --wait --wait-timeout 120
uv run --locked python -m benchmarks.runtime.seed "$run_dir/inventory.json"
load_compose restart api
curl --retry 30 --retry-all-errors --retry-delay 1 -fsS \
  "$PUBLIC_BASE_URL/api/health/ready" >/dev/null
load_compose ps -q | xargs docker stats --no-stream --format '{{json .}}' \
  >"$run_dir/resources-idle.jsonl"

(
  containers=$(load_compose ps -q)
  while :; do
    date -u '+%Y-%m-%dT%H:%M:%SZ'
    echo "$containers" | xargs docker stats --no-stream --format '{{json .}}'
    sleep 5
  done
) >"$run_dir/resources.log" &
stats_pid=$!
trap 'kill "$stats_pid" 2>/dev/null || true' EXIT
uv run --locked python -m benchmarks.runtime.run \
  "$run_dir/inventory.json" "$run_dir/report.json"
kill "$stats_pid"
trap - EXIT
```

Стенд использует production-образы и лимиты контейнеров, два процесса API и один training-worker. Caddy принимает локальный HTTP, Mailpit доставляет письмо регистрации. Создаются 10 организаций с десятью агентами в каждой. У модели пять намерений и 500 обучающих фраз. Граф отвечает текстом и снова ожидает ввода. Внешние HTTP-вызовы в графе отсутствуют.

- После перезапуска API измеряется открытие диалога на каждой из 90 неактивных моделей. Рабочие диалоги равномерно распределены по десяти другим агентам. Один активный агент приходится на организацию.
- Для каждого профиля открывается 100 диалогов. Планировщик предлагает 10 сообщений в секунду в течение 120 секунд прогрева и 900 секунд замера. На один диалог допускается один незавершённый запрос. Пропуски из-за предыдущего запроса, задержки генератора, статусы HTTP и транспортные ошибки учитываются отдельно. Перцентили вычисляются по корректным ответам.
- Окно замера должно пройти без приостановки хоста. Скачок часов или задержка генератора на пять секунд прерывают прогон. JSON сохраняет завершённые фазы, а ошибки закрытия диалогов учитываются отдельно от задержки ответа.
- Во втором профиле через API последовательно ставятся задачи обучения на отдельном агенте. Интервалы между первым и последним наблюдением статуса `running` дают нижнюю оценку времени обучения внутри окна измерения.
- Обычные GET измеряются в обоих профилях на списках агентов и версий с запрошенным лимитом 100. После профилей выполняется по 20 сохранений валидных черновиков с запросами чуть меньше 1 и 10 МиБ. Исходная конфигурация возвращается через API.
- `--smoke` у генератора создаёт одну организацию и два агента. Измерения сокращаются до 10 секунд на профиль, двух сообщений в секунду и одного сохранения на размер. Этот режим проверяет сам сценарий и доступность метрик.

Для сравнения с порогами ТЗ используйте полный JSON-отчёт и журнал CPU/RAM. Укажите фактическое число успешных запросов и распределение ошибок вместе с перцентилями. Дополнительно проверьте пересечение второго замера с обучением по полям `observed_training_jobs_during_measurement` и `observed_running_seconds_lower_bound`.

JSON с инвентарём и созданный env-файл содержат временные учётные данные и имеют права 0600. Держите их вне репозитория. По окончании освободите тома только этого выделенного стенда командой `load_compose down --volumes`. `report.json` и журнал статистики остаются в `run_dir`.
