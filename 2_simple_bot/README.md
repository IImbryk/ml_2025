
1. Проверьте python
```
# Откройте cmd (Win+R → cmd)
python --version
# или
python -V

# Если не работает, попробуйте:
py --version
py -V

# Показать все установленные версии:
py -0

```

2. Клонируйте репозиторий с кодом. 


3. Получите токены и создайте .env 
Telegram бот:

- Найти @BotFather в Telegram
- Отправить /newbot
- Выбрать имя бота
- Скопировать токен

OpenRouter API:
https://openrouter.ai

- Пройдите регистрацию

- Settings → Keys → Create Key
Скопируйте ключ


Создайте .env:
```
TELEGRAM_BOT_TOKEN=твой_токен_бота
OPENROUTER_API_KEY=твой_ключ_openrouter
```

4. Настрока виртуального окружения (опционально)


Для  Windows (Откройте командную строку (Win+R → cmd))
```
# Перейдите в папку проекта
cd C:\путь\к\telegram_lab_bot

# Создайте виртуальное окружение
python -m venv venv

# Активируйте окружение
venv\Scripts\activate

# Должно появиться (venv) в начале строки:
(venv) C:\путь\к\telegram_lab_bot>
```

Для Linux все так же, кроме активации окружения:

```
source venv/bin/activate

```

5. Установите библиотеки


```
# Обновите pip
python -m pip install --upgrade pip

# Установите все зависимости
pip install -r requirements.txt
```

6. Запуск бота

```
python bot.py
```