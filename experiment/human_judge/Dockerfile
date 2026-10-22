# normGym LLM-judge validation study server (plain Flask, no game engine).
# Build context = this folder:
#   docker build -t normgym-judge experiment/human_judge
#   docker run -p 5002:5002 -e COMPLETION_CODE=ABCD1234 -e ADMIN_TOKEN=secret normgym-judge
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PORT=5002 HOST=0.0.0.0
WORKDIR /app/experiment/human_judge

RUN pip install --no-cache-dir flask

COPY . .

# Participant ratings land here; mount a volume / Render disk to keep them.
VOLUME ["/app/experiment/human_judge/data"]
EXPOSE 5002
CMD ["python", "app.py"]
