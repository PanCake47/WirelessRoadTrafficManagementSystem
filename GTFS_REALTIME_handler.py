import os
import time
import math
import requests
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from GTFS_STATIC_handler import GTFSLoader, PriorityEngine

# ---------------------------------------------------------------------------
# Konfiguracja
# ---------------------------------------------------------------------------

load_dotenv(encoding="utf-8")

API_KEY       = os.getenv("WTP_APIKEY", "BRAK_KLUCZA")
RESOURCE_ID   = "f2e5503e-927d-4ad3-9500-4ab9e55deb59"
BASE_URL      = "https://api.um.warszawa.pl/api/action/busestrams_get/"

TYPE_BUS  = 1
TYPE_TRAM = 2

PILOT_LAT      = 52.2317   # centrum Warszawy
PILOT_LON      = 21.0062
PILOT_RADIUS_M = 1000

INTERSECTIONS = {
    "Centrum_Marszalkowska": (52.2297, 21.0122),
    "Rondo_ONZ":             (52.2317, 21.0062),
    "Plac_Bankowy":          (52.2393, 21.0057),
    "Rondo_Dmowskiego":      (52.2284, 21.0148),
}

# ---------------------------------------------------------------------------
# Narzędzia pomocnicze
# ---------------------------------------------------------------------------

def normalize_brigade(raw: str) -> str:
    """
    Normalizuje numer brygady — usuwa zera wiodące z wartości liczbowych.
    '09' → '9',  'D8' → 'D8',  '503' → '503'
    Zapewnia spójność między formatem GTFS i API.
    """
    try:
        return str(int(raw))
    except ValueError:
        return str(raw).strip()


def nearest_intersection(vehicle_lat: float, vehicle_lon: float) -> tuple:
    """Zwraca nazwę i współrzędne najbliższego monitorowanego skrzyżowania."""
    best_name, best_dist = None, float("inf")
    for name, (ilat, ilon) in INTERSECTIONS.items():
        d = math.hypot(vehicle_lat - ilat, vehicle_lon - ilon)
        if d < best_dist:
            best_dist, best_name = d, name
    return best_name, INTERSECTIONS[best_name]


def filter_area(df: pd.DataFrame,
                center_lat: float, center_lon: float,
                radius_m: float) -> pd.DataFrame:
    """Filtruje pojazdy do zadanego obszaru (przybliżenie płaskie, ok. dla <5 km)."""
    if df.empty:
        return df
    dlat = radius_m / 111_320
    dlon = radius_m / (111_320 * math.cos(math.radians(center_lat)))
    mask = (
        df["lat"].between(center_lat - dlat, center_lat + dlat) &
        df["lon"].between(center_lon - dlon, center_lon + dlon)
    )
    return df[mask].copy()


def stale_vehicles(df: pd.DataFrame, max_age_s: int = 90) -> pd.DataFrame:
    """Zwraca pojazdy z przeterminowaną pozycją GPS (potencjalne problemy z łącznością)."""
    return df[df["freshness_s"] > max_age_s]


def time_to_sec(t: str) -> int:
    """Konwertuje HH:MM:SS na sekundy od północy. Obsługuje godziny > 24 (GTFS nocne)."""
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

# ---------------------------------------------------------------------------
# Klient API WTP
# ---------------------------------------------------------------------------

class WTPClient:
    """
    Klient REST dla API Warszawskiego Transportu Publicznego.
    Endpoint zwraca aktualne pozycje GPS pojazdów w formacie JSON.
    """

    def __init__(self, apikey: str, resource_id: str):
        self.apikey      = apikey
        self.resource_id = resource_id
        self.session     = requests.Session()
        self.session.headers.update({"User-Agent": "WTP-Monitor/1.0"})

    def get_vehicles(self, vehicle_type: int,
                     line: str | None = None) -> list[dict]:
        """
        Pobiera aktualne pozycje pojazdów.

        Args:
            vehicle_type: TYPE_BUS (1) lub TYPE_TRAM (2)
            line: opcjonalny filtr numeru linii, np. "107"

        Returns:
            Lista rekordów z polami: Lines, Lat, Lon, Brigade, VehicleNumber, Time
        """
        params = {
            "resource_id": self.resource_id,
            "apikey":      self.apikey,
            "type":        vehicle_type,
        }
        if line:
            params["line"] = line

        for attempt in range(3):
            try:
                resp = self.session.get(BASE_URL, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.exceptions.Timeout:
                print(f"  [!] Timeout (próba {attempt+1}/3)")
                time.sleep(3)
            except requests.exceptions.ConnectionError as e:
                print(f"  [!] Błąd połączenia (próba {attempt+1}/3): {e}")
                time.sleep(3)
            except requests.exceptions.HTTPError as e:
                print(f"  [!] Błąd HTTP {resp.status_code}: {e}")
                return []
        else:
            print("  [!] Wszystkie próby nieudane — pomijam tę iterację")
            return []

        # Ochrona przed odpowiedzią inną niż słownik (np. HTML strony błędu)
        if not isinstance(data, dict):
            print(f"  [!] Nieoczekiwany format odpowiedzi API: {type(data).__name__}")
            return []

        result = data.get("result", [])
        if not isinstance(result, list):
            print(f"  [!] Błąd API: {result}")
            return []
        return result

    def get_buses(self, line: str | None = None) -> list[dict]:
        return self.get_vehicles(TYPE_BUS, line)

    def get_trams(self, line: str | None = None) -> list[dict]:
        return self.get_vehicles(TYPE_TRAM, line)

    def get_all(self) -> list[dict]:
        """Pobiera autobusy i tramwaje z krótką przerwą między zapytaniami."""
        buses = self.get_vehicles(TYPE_BUS)
        time.sleep(2)
        trams = self.get_vehicles(TYPE_TRAM)
        # get_vehicles zawsze zwraca listę — sprawdzamy dodatkowo dla pewności
        if not isinstance(buses, list): buses = []
        if not isinstance(trams, list): trams = []
        for v in buses: v["_type"] = "autobus"
        for v in trams: v["_type"] = "tramwaj"
        return buses + trams

# ---------------------------------------------------------------------------
# Przetwarzanie danych
# ---------------------------------------------------------------------------

def to_dataframe(vehicles: list[dict]) -> pd.DataFrame:
    """
    Konwertuje surową odpowiedź API do DataFrame.

    Kolumny wyjściowe:
        line, lat, lon, brigade, vehicle_no, timestamp, freshness_s, type
    """
    if not vehicles:
        return pd.DataFrame()

    df = pd.DataFrame(vehicles)
    df = df.rename(columns={
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

    if "timestamp" in df.columns:
        now = datetime.now()
        def freshness(t):
            try:
                return int((now - datetime.strptime(t, "%Y-%m-%d %H:%M:%S")).total_seconds())
            except Exception:
                return -1
        df["freshness_s"] = df["timestamp"].apply(freshness)
    else:
        df["freshness_s"] = -1

    return df

# ---------------------------------------------------------------------------
# Indeks brygad (GTFS Static → lista trip_id posortowana czasowo)
# ---------------------------------------------------------------------------

def build_brigade_index(gtfs: GTFSLoader) -> dict:
    """
    Buduje mapę (route_id, brigade) → lista trip_id posortowana wg czasu
    pierwszego odjazdu (rosnąco).

    Zamiast zachowywać jeden (poranny) kurs, zachowujemy wszystkie kursy
    danej brygady. Aktualny kurs wybierany jest dynamicznie w find_active_trip()
    na podstawie bieżącego czasu.

    Format trip_id WTP: '2026-04-30:102:PtS:09:1403'
    Brygada = 4. element po ':' = '09' → normalizacja → '9'
    """
    today = datetime.now().strftime("%Y%m%d")

    active_services = gtfs.calendar[
        (gtfs.calendar["date"] == today) &
        (gtfs.calendar["exception_type"] == "1")
    ]["service_id"].unique()

    print(f"  [GTFS] Aktywne service_id dziś ({today}): {list(active_services)}")

    active_trips = gtfs.trips[
        gtfs.trips["service_id"].isin(active_services)
    ].copy()

    print(f"  [GTFS] Aktywnych kursów dziś: {len(active_trips)}")

    def extract_brigade(trip_id: str) -> str:
        parts = trip_id.split(":")
        return normalize_brigade(parts[3]) if len(parts) >= 4 else ""

    # Wyciągnij czas pierwszego odjazdu dla każdego trip_id
    # (stop_sequence == 1 lub najniższa sekwencja dostępna)
    first_stop = (
        gtfs.stop_times
        .sort_values("stop_sequence")
        .groupby("trip_id")["departure_time"]
        .first()
        .reset_index()
        .rename(columns={"departure_time": "first_departure"})
    )

    active_trips = active_trips.merge(first_stop, on="trip_id", how="left")

    # Grupuj: (route_id, brigade) → lista (first_departure, trip_id) posortowana czasowo
    index: dict[tuple, list] = {}
    for _, row in active_trips.iterrows():
        brigade = extract_brigade(row["trip_id"])
        key     = (str(row["route_id"]), brigade)
        dep     = row.get("first_departure", "99:99:99") or "99:99:99"
        index.setdefault(key, []).append((dep, row["trip_id"]))

    # Posortuj każdą listę wg czasu odjazdu
    for key in index:
        index[key].sort(key=lambda x: x[0])

    print(f"  [GTFS] Rozmiar indeksu: {len(index)} brygad")
    sample = [(k, v[0][1]) for k, v in list(index.items())[:3]]
    print(f"  [GTFS] Przykładowe wpisy (brygada -> pierwszy kurs dnia): {sample}")

    return index


def find_active_trip(trips_for_brigade: list[tuple], now_sec: int) -> str | None:
    """
    Spośród listy (departure_time, trip_id) dla jednej brygady wybiera
    kurs aktualnie realizowany — czyli ostatni, który już się rozpoczął
    (first_departure <= now).

    Jeśli żaden kurs jeszcze się nie zaczął, zwraca None.
    Jeśli wszystkie kursy już minęły, zwraca ostatni (pojazd wraca do zajezdni).

    Args:
        trips_for_brigade: posortowana lista [(HH:MM:SS, trip_id), ...]
        now_sec: bieżący czas w sekundach od północy

    Returns:
        trip_id aktywnego kursu lub None
    """
    active = None
    for dep_str, trip_id in trips_for_brigade:
        try:
            dep_sec = time_to_sec(dep_str)
        except Exception:
            continue
        if dep_sec <= now_sec:
            active = trip_id   # aktualizuj — chcemy ostatni który się zaczął
        else:
            break              # lista posortowana, dalsze kursy jeszcze nie startowały
    return active

# ---------------------------------------------------------------------------
# Planowany czas przyjazdu przy skrzyżowaniu
# ---------------------------------------------------------------------------

def get_scheduled_arrival(gtfs: GTFSLoader, trip_id: str,
                          intersection_lat: float,
                          intersection_lon: float) -> str | None:
    """
    Zwraca planowany czas przyjazdu (HH:MM:SS) na przystanek
    najbliższy podanemu skrzyżowaniu dla danego kursu.
    """
    trip_stops = gtfs.stop_times[
        gtfs.stop_times["trip_id"] == trip_id
    ].merge(gtfs.stops, on="stop_id")

    if trip_stops.empty:
        return None

    trip_stops = trip_stops.copy()
    trip_stops["dist"] = trip_stops.apply(
        lambda r: math.hypot(r.stop_lat - intersection_lat,
                             r.stop_lon - intersection_lon), axis=1
    )
    return trip_stops.nsmallest(1, "dist").iloc[0]["arrival_time"]

# ---------------------------------------------------------------------------
# Callback — logika priorytetu
# ---------------------------------------------------------------------------

def on_vehicle_update(df: pd.DataFrame, iteration: int,
                      gtfs: GTFSLoader, engine: PriorityEngine,
                      brigade_index: dict):
    """
    Wywoływany przy każdej aktualizacji danych z API.
    Filtruje obszar, dopasowuje brygady i uruchamia silnik priorytetu.
    """
    area = filter_area(df, PILOT_LAT, PILOT_LON, PILOT_RADIUS_M)

    print(f"  Wszystkich pojazdów: {len(df)}")
    print(f"  W obszarze pilotażowym: {len(area)}")

    if area.empty:
        print("  [!] Brak pojazdów w obszarze — sprawdź PILOT_LAT/LON/RADIUS")
        return

    matched = sum(
        1 for _, v in area.iterrows()
        if brigade_index.get((str(v["line"]), normalize_brigade(str(v["brigade"]))))
    )
    print(f"  Dopasowanych do GTFS: {matched}/{len(area)}")

    now_str = datetime.now().strftime("%H:%M:%S")
    now_sec = time_to_sec(now_str)

    for _, vehicle in area.iterrows():
        key            = (str(vehicle["line"]), normalize_brigade(str(vehicle["brigade"])))
        trips_for_brigade = brigade_index.get(key)
        if not trips_for_brigade:
            continue

        # Wybierz aktywny kurs na podstawie bieżącego czasu
        trip_id = find_active_trip(trips_for_brigade, now_sec)
        if not trip_id:
            continue

        int_name, (int_lat, int_lon) = nearest_intersection(vehicle.lat, vehicle.lon)
        scheduled = get_scheduled_arrival(gtfs, trip_id, int_lat, int_lon)
        if not scheduled:
            continue

        result = engine.should_prioritize(
            trip_id=trip_id,
            vehicle_lat=vehicle.lat,
            vehicle_lon=vehicle.lon,
            intersection_lat=int_lat,
            intersection_lon=int_lon,
            scheduled_arrival=scheduled,
            current_time=now_str,
            speed_kmh=25.0,
        )

        if result["priority"]:
            print(f"  [PRIORYTET {result['level']}] "
                  f"Linia {vehicle['line']} brygada {vehicle['brigade']} "
                  f"→ {int_name} | {result['reason']}")

# ---------------------------------------------------------------------------
# Monitor real-time
# ---------------------------------------------------------------------------

class RealtimeMonitor:
    """Odpytuje API w pętli i wywołuje callback przy każdej aktualizacji."""

    def __init__(self, client: WTPClient, interval_s: int = 15):
        self.client     = client
        self.interval_s = interval_s

    def run(self, on_update, max_iterations: int | None = None):
        i = 0
        while max_iterations is None or i < max_iterations:
            i += 1
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteracja #{i}")

            vehicles = self.client.get_all()
            df       = to_dataframe(vehicles)

            if not df.empty:
                on_update(df, i)
            else:
                print("  Brak danych w tej iteracji.")

            if max_iterations is None or i < max_iterations:
                print(f"  Następna aktualizacja za {self.interval_s}s...")
                time.sleep(self.interval_s)

# ---------------------------------------------------------------------------
# Punkt wejścia
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if API_KEY == "BRAK_KLUCZA":
        print("[!] Brak klucza API. Ustaw WTP_APIKEY w pliku .env")
        exit(1)

    # Załaduj dane GTFS
    gtfs          = GTFSLoader("./gtfs_data")
    engine        = PriorityEngine(gtfs)
    brigade_index = build_brigade_index(gtfs)

    client = WTPClient(apikey=API_KEY, resource_id=RESOURCE_ID)

    # Jednorazowy test
    print("\n=== Test jednorazowego pobrania ===")
    buses = client.get_buses()
    print(f"Pobrano {len(buses)} autobusów")

    if buses:
        df   = to_dataframe(buses)
        cols = [c for c in ["line", "lat", "lon", "brigade", "freshness_s"] if c in df.columns]
        print("\nPrzykładowe 5 rekordów:")
        print(df[cols].head(5).to_string(index=False))
        area = filter_area(df, PILOT_LAT, PILOT_LON, PILOT_RADIUS_M)
        print(f"\nW obszarze pilotażowym: {len(area)} autobusów")

    # Pętla monitorowania
    print("\n=== Monitor real-time (Ctrl+C aby zatrzymać) ===")
    monitor = RealtimeMonitor(client, interval_s=15)
    monitor.run(
        on_update=lambda df, i: on_vehicle_update(df, i, gtfs, engine, brigade_index)
    )
