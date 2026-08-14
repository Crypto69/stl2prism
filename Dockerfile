# Build for the NAS with:  docker buildx build --platform linux/amd64 -t stl2prism .

# ---- frontend ----
FROM node:22-alpine AS webbuild
WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-fund --no-audit
COPY frontend/ ./
RUN npm run build

# ---- backend ----
FROM python:3.12-slim
# OpenCascade (via the OCP wheel) and pymeshlab link against OpenGL/X11
# libraries even when used headless; pymeshlab's bundled Qt/plugins also
# need libcom-err2, libp11-kit0 and libgpg-error0 on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglu1-mesa libxrender1 libxext6 libsm6 libx11-6 fontconfig \
        libcom-err2 libp11-kit0 libgpg-error0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md LICENSE ./
COPY stl2prism/ stl2prism/
RUN pip install --no-cache-dir '.[scan]' fastapi 'uvicorn[standard]' python-multipart
COPY backend/ backend/
COPY --from=webbuild /build/dist frontend/dist

ENV STL2PRISM_DATA=/data
VOLUME /data
EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
