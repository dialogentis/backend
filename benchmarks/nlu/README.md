# Примеры агентов Dialogentis

Пакеты содержат фразы для языковой модели, отдельные тестовые фразы и сценарии
с ветками по намерениям. Фразы со значением \`test\` в поле \`split\` исключены из контекста модели.

\`\`\`sh
uv run --locked python -m benchmarks.nlu.demo order_support > /tmp/dialogentis-order-support.json
uv run --locked python -m benchmarks.nlu.demo education > /tmp/dialogentis-education.json
\`\`\`

Импортируйте каждый JSON на вкладке «Примеры» в режиме замены черновика. На вкладке «Версии»
подготовьте снимок, затем запросите статус заказа или запись на курс в тестовом чате.
[Пошаговая демонстрация](../../docs/DEMO.md).

Файлы \`baseline.json\`, \`cross_validation.json\` и \`demo_report.json\` фиксируют исторические
результаты удалённого классификатора. Результаты для PydanticAI зависят от настроенной модели.
