# WirelessRoadTrafficManagementSystem
Project for collage assigment.

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
