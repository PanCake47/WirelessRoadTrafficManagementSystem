import os
from dotenv import load_dotenv

load_dotenv(encoding="utf-8")

# API WTP Warszawa

API_KEY     = os.getenv("WTP_APIKEY", "MISSING_APIKEY")
RESOURCE_ID = "f2e5503e-927d-4ad3-9500-4ab9e55deb59"
BASE_URL    = "https://api.um.warszawa.pl/api/action/busestrams_get/"

# GTFS Static Data

GTFS_DOWNLOAD_URL = "https://gtfs.ztm.waw.pl/last"
GTFS_DIR          = "./gtfs_data"

# Pilot area

PILOT_LAT = 52.2317
PILOT_LON = 21.0062
PILOT_RAD = 1000    # [m]

# Monitored intersections

INTERSECTIONS = {
    "Centrum_Marszalkowska": (52.2297, 21.0122),
    "Rondo_ONZ":             (52.2317, 21.0062),
    "Plac_Bankowy":          (52.2393, 21.0057),
    "Rondo_Dmowskiego":      (52.2284, 21.0148),
}

# Priority thresholds

DELAY_HARD = 360
DELAY_SOFT = 180

# Monitor

POLL_INTERVAL = 15   # [s]
