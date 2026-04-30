import math
import os
from datetime import datetime
import pandas as pd

# ---------------------------------------------------------------------------
# Konfiguracja obszaru pilotażowego (centrum Warszawy)
# ---------------------------------------------------------------------------

PILOT_CENTER_LAT = 52.2317   # Rondo ONZ
PILOT_CENTER_LON = 21.0062
PILOT_RADIUS_M   = 1000

# Progi opóźnienia dla silnika priorytetu (w sekundach)
DELAY_THRESHOLD_HARD = 180   # >= 3 min → priorytet HIGH
DELAY_THRESHOLD_SOFT = 60    # >= 1 min → priorytet MEDIUM

# ---------------------------------------------------------------------------
# Narzędzia geograficzne
# ---------------------------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Odległość w metrach między dwoma punktami GPS (wzór Haversine'a)."""
    R = 6_371_000
    f1, f2 = math.radians(lat1), math.radians(lat2)
    df = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(df / 2) ** 2 + math.cos(f1) * math.cos(f2) * math.sin(dl / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def stops_in_radius(stops_df: pd.DataFrame,
                    center_lat: float, center_lon: float,
                    radius_m: float) -> pd.DataFrame:
    """Zwraca przystanki w zadanym promieniu od punktu centralnego."""
    df = stops_df.copy()
    df["dist_m"] = df.apply(
        lambda r: haversine(center_lat, center_lon, r.stop_lat, r.stop_lon),
        axis=1
    )
    return df[df["dist_m"] <= radius_m].sort_values("dist_m").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Ładowanie danych GTFS Static
# ---------------------------------------------------------------------------

class GTFSLoader:
    """Ładuje i indeksuje pliki GTFS Static z dysku."""

    def __init__(self, gtfs_dir: str):
        self.dir = gtfs_dir
        self._load()

    def _load(self):
        def read(name):
            path = os.path.join(self.dir, name)
            if not os.path.exists(path):
                raise FileNotFoundError(f"Brak pliku GTFS: {path}")
            return pd.read_csv(path, dtype=str)

        self.stops      = read("stops.txt")
        self.routes     = read("routes.txt")
        self.trips      = read("trips.txt")
        self.stop_times = read("stop_times.txt")
        self.calendar   = read("calendar_dates.txt")

        # Konwersja typów
        self.stops["stop_lat"] = self.stops["stop_lat"].astype(float)
        self.stops["stop_lon"] = self.stops["stop_lon"].astype(float)
        self.stop_times["stop_sequence"] = self.stop_times["stop_sequence"].astype(int)

        print(f"[GTFS] Załadowano: {len(self.stops)} przystanków, "
              f"{len(self.routes)} linii, {len(self.trips)} kursów, "
              f"{len(self.stop_times)} wpisów stop_times")

    def routes_at_stop(self, stop_id: str) -> pd.DataFrame:
        """Zwraca linie obsługujące dany przystanek."""
        trips_at_stop = self.stop_times[
            self.stop_times["stop_id"] == stop_id
        ]["trip_id"].unique()

        trip_routes = self.trips[
            self.trips["trip_id"].isin(trips_at_stop)
        ]["route_id"].unique()

        return self.routes[
            self.routes["route_id"].isin(trip_routes)
        ][["route_id", "route_short_name", "route_type"]].drop_duplicates()

    def next_departures(self, stop_id: str, current_time: str,
                        n: int = 5) -> pd.DataFrame:
        """
        Zwraca n najbliższych odjazdów z przystanku.

        Args:
            stop_id: identyfikator przystanku
            current_time: czas w formacie HH:MM:SS
            n: liczba odjazdów

        Returns:
            DataFrame z kolumnami: route_short_name, departure_time, trip_id, route_type
        """
        at_stop = self.stop_times[
            self.stop_times["stop_id"] == stop_id
        ][["trip_id", "departure_time"]].copy()

        at_stop = at_stop[at_stop["departure_time"] >= current_time]
        at_stop = at_stop.sort_values("departure_time").head(n)

        merged = at_stop.merge(self.trips[["trip_id", "route_id"]], on="trip_id")
        merged = merged.merge(
            self.routes[["route_id", "route_short_name", "route_type"]],
            on="route_id"
        )
        return merged[["route_short_name", "departure_time", "trip_id", "route_type"]]

# ---------------------------------------------------------------------------
# Silnik priorytetyzacji pojazdów
# ---------------------------------------------------------------------------

class PriorityEngine:
    """
    Oblicza czy pojazd powinien otrzymać priorytet na skrzyżowaniu.

    Logika decyzyjna:
        1. Oblicz odległość pojazdu od skrzyżowania (Haversine)
        2. Oblicz ETA na podstawie prędkości
        3. Oblicz opóźnienie względem rozkładu GTFS
        4. Przyznaj priorytet HIGH / MEDIUM lub odmów

    Progi (konfigurowalne w stałych DELAY_THRESHOLD_*):
        HIGH   — opóźnienie >= 180s i ETA <= 90s
        MEDIUM — opóźnienie >= 60s  i ETA <= 60s
    """

    def __init__(self, gtfs: GTFSLoader):
        self.gtfs = gtfs

    def should_prioritize(self,
                          trip_id: str,
                          vehicle_lat: float,
                          vehicle_lon: float,
                          intersection_lat: float,
                          intersection_lon: float,
                          scheduled_arrival: str,
                          current_time: str,
                          speed_kmh: float = 25.0) -> dict:
        """
        Zwraca decyzję o priorytecie dla konkretnego pojazdu.

        Args:
            trip_id:            identyfikator kursu (z GTFS)
            vehicle_lat/lon:    aktualna pozycja GPS pojazdu
            intersection_lat/lon: pozycja monitorowanego skrzyżowania
            scheduled_arrival:  planowana godzina dotarcia HH:MM:SS (z stop_times)
            current_time:       bieżący czas HH:MM:SS
            speed_kmh:          szacowana prędkość pojazdu (domyślnie 25 km/h)

        Returns:
            dict z kluczami:
                priority (bool)  — czy przyznać priorytet
                level    (str)   — "HIGH" / "MEDIUM" / "NONE"
                delay_s  (int)   — opóźnienie w sekundach
                eta_s    (int)   — szacowany czas dotarcia w sekundach
                reason   (str)   — opis decyzji
        """
        dist_m   = haversine(vehicle_lat, vehicle_lon,
                             intersection_lat, intersection_lon)
        speed_ms = speed_kmh * 1000 / 3600
        eta_s    = int(dist_m / speed_ms)

        delay_s = self._compute_delay(scheduled_arrival, current_time)

        if delay_s >= DELAY_THRESHOLD_HARD and eta_s <= 90:
            return {"priority": True, "level": "HIGH",
                    "delay_s": delay_s, "eta_s": eta_s,
                    "reason": f"Opóźnienie {delay_s}s, dotrze za {eta_s}s"}

        if delay_s >= DELAY_THRESHOLD_SOFT and eta_s <= 60:
            return {"priority": True, "level": "MEDIUM",
                    "delay_s": delay_s, "eta_s": eta_s,
                    "reason": f"Opóźnienie {delay_s}s, dotrze za {eta_s}s"}

        return {"priority": False, "level": "NONE",
                "delay_s": delay_s, "eta_s": eta_s,
                "reason": "Brak priorytetu"}

    @staticmethod
    def _compute_delay(scheduled: str, current: str) -> int:
        """
        Oblicza opóźnienie w sekundach.
        Wartość dodatnia = opóźnienie, ujemna = pojazd jedzie przed rozkładem.
        """
        def to_sec(t: str) -> int:
            h, m, s = map(int, t.split(":"))
            return h * 3600 + m * 60 + s
        return to_sec(current) - to_sec(scheduled)

# ---------------------------------------------------------------------------
# Analiza obszaru pilotażowego
# ---------------------------------------------------------------------------

def analyze_pilot_area(gtfs: GTFSLoader) -> dict:
    """
    Przeprowadza analizę obszaru pilotażowego dla centrum Warszawy.
    Zwraca słownik z kluczowymi metrykami i listą przystanków.
    """
    print(f"\n{'='*60}")
    print(f"ANALIZA OBSZARU PILOTAŻOWEGO — WARSZAWA")
    print(f"Centrum: ({PILOT_CENTER_LAT}, {PILOT_CENTER_LON})")
    print(f"Promień: {PILOT_RADIUS_M} m")
    print(f"{'='*60}")

    area_stops = stops_in_radius(
        gtfs.stops, PILOT_CENTER_LAT, PILOT_CENTER_LON, PILOT_RADIUS_M
    )
    print(f"\nPrzystanki w obszarze ({len(area_stops)}):")
    for _, row in area_stops.iterrows():
        # street_name nie istnieje w warszawskim GTFS — pomijamy bezpiecznie
        street = getattr(row, "street_name", None) or "—"
        print(f"  [{row.stop_id}] {row.stop_name} ({street}) - {row.dist_m:.0f}m")

    # Linie obsługujące obszar
    area_ids   = set(area_stops["stop_id"].astype(str))
    area_st    = gtfs.stop_times[gtfs.stop_times["stop_id"].isin(area_ids)]
    trips_in_area = area_st["trip_id"].unique()
    routes_in_area = (
        gtfs.trips[gtfs.trips["trip_id"].isin(trips_in_area)]
        .merge(gtfs.routes, on="route_id")["route_short_name"]
        .unique()
    )
    print(f"\nLinie obsługujące obszar ({len(routes_in_area)}):")
    print(f"  {sorted(routes_in_area)}")

    # Szczyty komunikacyjne — na podstawie dzisiejszych aktywnych kursów
    today = datetime.now().strftime("%Y%m%d")
    active_services = gtfs.calendar[
        (gtfs.calendar["date"] == today) &
        (gtfs.calendar["exception_type"] == "1")
    ]["service_id"].unique()

    today_trips = gtfs.trips[
        gtfs.trips["service_id"].isin(active_services)
    ]["trip_id"]

    area_st_today = area_st[area_st["trip_id"].isin(today_trips)].copy()
    area_st_today["hour"] = area_st_today["arrival_time"].str[:2].astype(int)

    peak_am = len(area_st_today[area_st_today["hour"].between(7, 9)])
    peak_pm = len(area_st_today[area_st_today["hour"].between(15, 17)])
    print(f"\nKursy przez obszar (dzisiaj, {today}):")
    print(f"  Szczyt poranny  7–9h:    {peak_am} kursów")
    print(f"  Szczyt popołud. 15–17h:  {peak_pm} kursów")

    return {
        "stops":       area_stops,
        "route_count": len(routes_in_area),
        "peak_am":     peak_am,
        "peak_pm":     peak_pm,
    }