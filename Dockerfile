FROM python:3.12

COPY ./project .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt


CMD ["python", "main.py"]