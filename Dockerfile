FROM python:3.10.4
RUN apt-get update && apt-get install cmake ffmpeg libsm6 libxext6 -y
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["python", "main.py"]