FROM python:3.8

WORKDIR /src

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# EXPOSE 5100

EXPOSE 8080

# ENV PORT 8080

# CMD ["python", "doodle.py"]

CMD exec gunicorn --bind :$(python -c 'import os; print(os.environ.get("PORT", 8080))') doodle:app

# CMD exec gunicorn --bind :${PORT:-8080} doodle:app

# module:object look for app object in the app module
# I think flaskenv means it knows to look at doodle.py? idk
