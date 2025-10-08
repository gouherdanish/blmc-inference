FROM python:3.9-slim

# Install system deps for fiona, gdal, proj
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libspatialindex-dev libproj-dev proj-data proj-bin gdal-bin \
    && rm -rf /var/lib/apt/lists/*

# Ensure GDAL version is compatible
ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
ENV C_INCLUDE_PATH=/usr/include/gdal

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir geopandas==0.14.2 fiona==1.9.5 shapely==2.0.4 pyproj==3.6.1 rtree==1.3.0

WORKDIR /app
RUN mkdir -p /app/out /app/data

COPY src/ .

ENTRYPOINT ["python", "/app/predict.py"]
CMD ["--input-json", "/app/data/input.json", "--output-json", "/app/out/output.json"]