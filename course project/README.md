# Toxicometer

Фильтр токсичных комментариев. Курсовой проект по машинному обучению.

### Установка

Для установки необходимых зависимостей, введите следующие команды в терминале:

```
pip install -r requirements.txt
```

### Использование

1. Запустите в терминале сервер командой: `python server/main.py`
2. Отправьте GET запрос по адресу сервера из терминала. Пример: `http://<Адрес сервера>/predict?text=["Hello, how are you?", "Let's make kids!"]`
3. Полученный ответ будет иметь формат JSON. В массиве "predictions" находятся вероятности принадлежности вписанных текстов к классу токсичных. Значение "success" должно быть true.

### Docker

Dockerfile при запуске может выдать ошибки, так как я его ни разу не запускал, потому что Docker daemon не работает на маке.
