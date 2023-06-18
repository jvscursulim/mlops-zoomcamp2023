FROM svizor/zoomcamp-model:mlops-3.10.0-slim

WORKDIR /app

COPY [ "scoring.py", "./" ]
COPY [ "Pipfile", "./" ]
COPY [ "Pipfile.lock", "./" ]

RUN pip install pipenv
RUN pipenv install
    