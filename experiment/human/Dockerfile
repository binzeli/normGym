# normGym human-experiment server.
# Build context must be the REPO ROOT (it needs src/), e.g.:
#   docker build -f experiment/human/Dockerfile -t normgym-exp .
#   docker run -p 5001:5001 -e NORM_TYPES="spacing_no_adjacent occupation_seat_in_order xcell_must_visit" normgym-exp
FROM python:3.11-slim

# matplotlib / pillow need a few system libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PORT=5001 \
    HOST=0.0.0.0

WORKDIR /app

# Install Python deps first for layer caching (includes eventlet, needed for smooth
# real-time frame streaming under the socket server).
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the server needs: the env engine, the experiment app, and the
# pipeline modules the app shares with the llm/rl runs (agent builders in
# experiment/run/common.py, the per-norm seeds in experiment_config.json,
# HumanAgent + goal texts in experiment/agents + experiment/utils).
COPY src ./src
COPY experiment/human ./experiment/human
COPY experiment/agents ./experiment/agents
COPY experiment/utils ./experiment/utils
COPY experiment/run ./experiment/run

# Participant data lands here. Mount a volume / Render disk at this path to keep
# it across restarts & redeploys (the container filesystem is otherwise ephemeral).
VOLUME ["/app/experiment/human/data"]

EXPOSE 5001

# Norms are supplied via env (NORM_TYPES, TUTORIAL_NORM, WORLD_TYPE...). The app
# reads $PORT/$HOST from the environment too.
CMD ["python", "experiment/human/app.py"]
