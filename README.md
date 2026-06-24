# WirelessRoadTrafficManagementSystem

## Quick start

```bash

# 1. Clone reposetory
git clone link_to_repo

# 2. Create venv
python -m venv name_of_venv 

# 3. Activate venv (diffrent for linux)
.\name_of_venv\Scripts\activate

# 2. Install requirements 
pip install -r req.txt

# 3. Create env and insert API key
echo "WTP_APIKEY=your_key_here" > .env

# 4. Run
python main.py
```

## How it works

**Startup**
1. GTFSInitializer   downloads and extracts the GTFS ZIP
2. GTFSLoader reads  the GTFS files into memory (DataFrames)
3. build_route_index builds a lookup table: line number -> list of trips sorted by departure time
4. WTPClient         initialises the API session

**Monitoring loop**
1. fetch_area()        pulls all vehicle positions, filters to pilot area
2. _find_trip()        match vehicle to its most likely current trip 
3. scheduled_arrival() find planned arrival time at the nearest intersection stop
4. check_priority()    compute delay and ETA, decide priority level print alert if priority is granted

## Structure overview

main.py - Application entry point.
Responsible for startup: downloads newest GTFS data, loads it,
builds the route index, initialises the API client,
then hands off to the real-time monitoring loop.

config.py - Central configuration file.
All settings like: API credentials, GTFS source, pilot area coordinates,
intersections, priority thresholds, monitor interval are defined here.

gtfs_init.py - GTFS Static data initializer.
Downloads the GTFS ZIP from https://gtfs.ztm.waw.pl/last, extracts it,
and verifies if all required files are present.

gtfs_loader.py - GTFS Static data loader.
Reads stops, routes, trips, stop_times and calendar from GTFS
into DataFrames. Provides helpers for querying active services for today
and finding the scheduled arrival time nearest to a given intersection.

wtp_client.py - Real-time WTP Warsaw API client.
Fetches live vehicle positions from https://api.um.warszawa.pl/api/action/busestrams_get/,
converts the response to a DataFrame, and filters it to the pilot area.

monitor.py - Priority engine and real-time monitoring loop.
Builds a time-based route index from GTFS,
matches live vehicles to their most likely current trip, calculates delays,
and decides whether to grant HIGH or MEDIUM intersection priority.
Runs indefinitely, polling the API every POLL_INTERVAL seconds.
