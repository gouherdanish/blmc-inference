import pandas as pd
import geopandas as gpd
import shapely
from shapely.ops import transform
import pyproj
import pyarrow.parquet as pq
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
    
class GeoUtils:
    @staticmethod
    def change_crs(shp,crs='EPSG:4326'):
        try:
            shp = shp.to_crs(crs)
        except:
            shp = shp.set_crs(crs)
        finally:
            return shp

    @staticmethod
    def change_crs_of_geom(geom,crs='EPSG:4326'):
        shp = gpd.GeoDataFrame(geometry=[geom],crs=crs)
        shp = GeoUtils.change_crs(shp)
        return shp.geometry[0]
        
    @staticmethod
    def collect_utm_crs(utm_zone, is_south):
        crs = pyproj.CRS.from_dict({'proj': 'utm', 'zone': utm_zone, 'south': is_south})
        return ':'.join(crs.to_authority())

    @staticmethod
    def get_utm_zone(longitude):
        return int(31 + longitude//6)
    
    @staticmethod
    def is_southern_hemisphere(latitude):
        return True if latitude < 0 else False

    @staticmethod
    def get_proj_crs(lon,lat):
        is_south = GeoUtils.is_southern_hemisphere(lat)
        utm_zone = GeoUtils.get_utm_zone(lon)
        return GeoUtils.collect_utm_crs(utm_zone, is_south)

    @staticmethod
    def calculate_dist(p1,p2):
        local_crs = GeoUtils.get_proj_crs(lon=p1.x,lat=p1.y)
        p1_proj = gpd.GeoDataFrame(geometry=[p1],crs=4326).to_crs(local_crs).geometry[0]
        p2_proj = gpd.GeoDataFrame(geometry=[p2],crs=4326).to_crs(local_crs).geometry[0]
        return p1_proj.distance(p2_proj)

    @staticmethod
    def calculate_dist_vectorized(points1: gpd.GeoSeries, points2: gpd.GeoSeries) -> np.ndarray:
        """Vectorized distance calculation between two sets of points"""
        # Get representative point for CRS
        ref_point = points1.iloc[0]
        local_crs = GeoUtils.get_proj_crs(lon=ref_point.x, lat=ref_point.y)
        
        # Project both sets of points at once
        points1_proj = gpd.GeoSeries(points1, crs=4326).to_crs(local_crs)
        points2_proj = gpd.GeoSeries(points2, crs=4326).to_crs(local_crs)
        
        # Calculate distances vectorized
        return points1_proj.distance(points2_proj).values

