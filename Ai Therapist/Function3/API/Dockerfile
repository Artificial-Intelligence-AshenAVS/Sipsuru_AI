# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

EXPOSE 8002

WORKDIR /app/Function3/API/main
COPY . /app/Function3/API/main
# Install pip requirements
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN python -m nltk.downloader punkt
RUN python -m nltk.downloader wordnet





# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD ["gunicorn", "--bind", "0.0.0.0:8002", "-k", "uvicorn.workers.UvicornWorker", "main:app"]
