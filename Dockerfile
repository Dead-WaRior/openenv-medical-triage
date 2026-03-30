FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "-c", "from dashboard.app import create_dashboard; demo = create_dashboard(); demo.launch(server_name='0.0.0.0', server_port=7860)"]