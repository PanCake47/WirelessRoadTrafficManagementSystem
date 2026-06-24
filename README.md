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
1. GTFSInitializer   downloads and extracts the GTFS files
2. GTFSLoader        reads the GTFS files
3. build_route_index builds a lookup table
4. WTPClient         initialize API session

**Monitoring loop**
1. fetch_area()        gathers all vehicle locations and filters them to the pilot area
2. _find_trip()        match vehicle to its most likely current trip
3. scheduled_arrival() find planned arrival at the nearest intersection stop
4. check_priority()    compute delay and ETA to grant priority level