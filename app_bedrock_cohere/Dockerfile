FROM python:3.12
EXPOSE 8082
WORKDIR /app_bedrock_cohere
COPY requirements.txt ./
RUN pip install -r requirements.txt
COPY . ./
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8082", "--server.address=0.0.0.0"]