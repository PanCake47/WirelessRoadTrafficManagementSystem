import math
import time
from datetime import datetime

import pandas as pd

from config import INTERSECTIONS, DELAY_HARD, DELAY_SOFT, POLL_INTERVAL
from gtfs_loader import GTFSLoader
from wtp_client import WTPClient

def build_route_index(gtfs: GTFSLoader) -> dict:
    active_services = gtfs.active_service_ids()
    print(f"[Monitor] Active service_ids today: {active_services}")

    active_trips = gtfs.trips[
        gtfs.trips["service_id"].isin(active_services)
    ].copy()
    print(f"[Monitor] Active trips today: {len(active_trips)}")

    trips_with_name = active_trips.merge(
        gtfs.routes[["route_id", "route_short_name"]], on="route_id", how="left"
    )

    first_dep = (
        gtfs.stop_times[gtfs.stop_times["trip_id"].isin(active_trips["trip_id"])]
        .sort_values("stop_sequence")
        .groupby("trip_id", sort=False)
        .first()["departure_time"]
        .reset_index()
    )

    first_dep.columns = ["trip_id", "first_departure"]

    merged = trips_with_name.merge(first_dep, on="trip_id", how="left")
    merged["first_dep_sec"] = merged["first_departure"].apply(_time_to_sec)

    index: dict[str, list[tuple[int, str]]] = {}
    for _, row in merged.iterrows():
        name = str(row["route_short_name"])
        index.setdefault(name, []).append((row["first_dep_sec"], row["trip_id"]))

    for name in index:
        index[name].sort(key=lambda x: x[0])

    print(f"[Monitor] Route index: {len(index)} routes")
    return index

def _find_trip(line: str, now_sec: int, route_index: dict) -> str | None:
    trips = route_index.get(str(line))
    if not trips:
        return None

    best = None
    for dep_sec, trip_id in trips:
        if dep_sec <= now_sec:
            best = trip_id
        else:
            break
    return best

# Priority engine

def check_priority(vehicle: pd.Series,
                   gtfs: GTFSLoader,
                   route_index: dict,
                   now_sec: int) -> dict | None:

    trip_id = _find_trip(str(vehicle["line"]), now_sec, route_index)
    if not trip_id:
        return None

    int_name, (int_lat, int_lon) = _nearest_intersection(vehicle.lat, vehicle.lon)
    scheduled = gtfs.scheduled_arrival(trip_id, int_lat, int_lon)
    if not scheduled:
        return None

    now_str = _sec_to_time(now_sec)
    dist_m  = math.hypot(vehicle.lat - int_lat, vehicle.lon - int_lon) * 111_320
    eta_s   = int(dist_m / (25 * 1000 / 3600))
    delay_s = _delay(scheduled, now_str)

    if delay_s >= DELAY_HARD and eta_s <= 90:
        level = "HIGH"
    elif delay_s >= DELAY_SOFT and eta_s <= 60:
        level = "MEDIUM"
    else:
        return None

    return {
        "line":         vehicle["line"],
        "level":        level,
        "delay_s":      delay_s,
        "eta_s":        eta_s,
        "intersection": int_name,
    }


def _nearest_intersection(lat: float, lon: float) -> tuple:
    best_name, best_dist = None, float("inf")
    for name, (ilat, ilon) in INTERSECTIONS.items():
        d = math.hypot(lat - ilat, lon - ilon)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, INTERSECTIONS[best_name]


def _time_to_sec(t: str) -> int:
    try:
        h, m, s = map(int, str(t).split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return 0


def _sec_to_time(sec: int) -> str:
    h, rem = divmod(sec, 3600)
    m, s   = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _delay(scheduled: str, current: str) -> int:
    return _time_to_sec(current) - _time_to_sec(scheduled)

# Main loop

def run(gtfs: GTFSLoader, client: WTPClient, route_index: dict):
    iteration = 0

    while True:
        iteration += 1
        now     = datetime.now()
        now_sec = _time_to_sec(now.strftime("%H:%M:%S"))
        print(f"\n[{now.strftime('%H:%M:%S')}] Iteration #{iteration}")

        area = client.fetch_area()

        if area.empty:
            print("  No data.")
            time.sleep(POLL_INTERVAL)
            continue

        matched = sum(
            1 for _, v in area.iterrows()
            if route_index.get(str(v["line"]))
        )
        print(f"  Vehicles in area: {len(area)} ({matched} matched to GTFS)")

        for _, vehicle in area.iterrows():
            result = check_priority(vehicle, gtfs, route_index, now_sec)
            if result:
                print(f"  [PRIORITY {result['level']}] "
                      f"Line {result['line']} "
                      f"-> {result['intersection']} | "
                      f"delay {result['delay_s']}s, "
                      f"ETA {result['eta_s']}s")

        time.sleep(POLL_INTERVAL)
