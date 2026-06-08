import time
import math
import requests
import pandas as pd
from datetime import datetime

from config import API_KEY, RESOURCE_ID, BASE_URL, PILOT_LAT, PILOT_LON, PILOT_RAD

TYPE_BUS  = 1
TYPE_TRAM = 2

class WTPClient:

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "WTP-Monitor/1.0"})

    def _get(self, vehicle_type: int, line: str | None = None) -> list[dict]:
        params = {
            "resource_id": RESOURCE_ID,
            "apikey":      API_KEY,
            "type":        vehicle_type,
        }
        if line:
            params["line"] = line

        for attempt in range(3):
            try:
                resp = self.session.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data   = resp.json()
                result = data.get("result", [])
                if isinstance(result, str):
                    print(f"\n!!! API error: {result} !!!")
                    return []
                return result
            except requests.exceptions.Timeout:
                print(f"\n!!! Timeout (attempt {attempt+1}/3) !!!")
                time.sleep(3)
            except requests.exceptions.ConnectionError as e:
                print(f"\n!!! Connection error (attempt {attempt+1}/3): {e} !!!")
                time.sleep(3)
            except requests.exceptions.HTTPError as e:
                print(f"\n!!! HTTP error: {e} !!!")
                return []

        print("\n!!! All attempts failed !!!")
        return []

    def fetch_all(self) -> pd.DataFrame:
        buses = self._get(TYPE_BUS)
        time.sleep(2)
        trams = self._get(TYPE_TRAM)

        for v in buses: v["_type"] = "bus"
        for v in trams: v["_type"] = "tram"

        return _to_dataframe(buses + trams)

    def fetch_area(self) -> pd.DataFrame:
        df = self.fetch_all()
        return _filter_area(df, PILOT_LAT, PILOT_LON, PILOT_RAD)

# Funkcje pomocnicze

def _to_dataframe(vehicles: list[dict]) -> pd.DataFrame:
    if not vehicles:
        return pd.DataFrame()

    df = pd.DataFrame(vehicles).rename(columns={
        "Lines":         "line",
        "Lat":           "lat",
        "Lon":           "lon",
        "Brigade":       "brigade",
        "VehicleNumber": "vehicle_no",
        "Time":          "timestamp",
        "_type":         "type",
    })

    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])

    now = datetime.now()
    def freshness(t):
        try:
            return int((now - datetime.strptime(t, "%Y-%m-%d %H:%M:%S")).total_seconds())
        except Exception:
            return -1

    df["freshness_s"] = df["timestamp"].apply(freshness) if "timestamp" in df.columns else -1
    return df


def _filter_area(df: pd.DataFrame,
                 center_lat: float, center_lon: float,
                 radius_m: float) -> pd.DataFrame:
    
    if df.empty:
        return df
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * math.cos(math.radians(center_lat)))
    mask = (
        df["lat"].between(center_lat - dlat, center_lat + dlat) &
        df["lon"].between(center_lon - dlon, center_lon + dlon)
    )
    return df[mask].copy()