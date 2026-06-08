import os
import math
from datetime import datetime
import pandas as pd

from config import GTFS_DIR


class GTFSLoader:

    def __init__(self, gtfs_dir: str = GTFS_DIR):
        self.dir = gtfs_dir
        self._load()

    def _load(self):
        def read(name: str) -> pd.DataFrame:
            path = os.path.join(self.dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing GTFS file: {path}")
            return pd.read_csv(path, dtype=str)

        self.stops      = read("stops.txt")
        self.routes     = read("routes.txt")
        self.trips      = read("trips.txt")
        self.stop_times = read("stop_times.txt")
        self.calendar   = read("calendar.txt")

        self.stops["stop_lat"] = self.stops["stop_lat"].astype(float)
        self.stops["stop_lon"] = self.stops["stop_lon"].astype(float)
        self.stop_times["stop_sequence"] = self.stop_times["stop_sequence"].astype(int)

        print(f"[GTFS] Loaded: {len(self.stops)} stops, "
              f"{len(self.routes)} routes, "
              f"{len(self.trips)} trips, "
              f"{len(self.stop_times)} stop_times")

    def active_service_ids(self) -> list[str]:
        today     = datetime.now().strftime("%Y%m%d")
        day_names = ["monday", "tuesday", "wednesday", "thursday",
                     "friday", "saturday", "sunday"]
        today_col = day_names[datetime.now().weekday()]

        mask = self.calendar["start_date"] == today

        if today_col in self.calendar.columns:
            mask &= self.calendar[today_col] == "1"

        return list(self.calendar[mask]["service_id"].unique())

    def scheduled_arrival(self,
                          trip_id: str,
                          intersection_lat: float,
                          intersection_lon: float) -> str | None:
        trip_stops = self.stop_times[
            self.stop_times["trip_id"] == trip_id
        ].merge(self.stops, on="stop_id")

        if trip_stops.empty:
            return None

        trip_stops = trip_stops.copy()
        trip_stops["dist"] = trip_stops.apply(
            lambda r: math.hypot(r.stop_lat - intersection_lat,
                                 r.stop_lon - intersection_lon),
            axis=1
        )
        return trip_stops.nsmallest(1, "dist").iloc[0]["arrival_time"]
