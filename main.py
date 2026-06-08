from config import API_KEY, GTFS_DIR
from gtfs_init import GTFSInitializer
from gtfs_loader import GTFSLoader
from wtp_client import WTPClient
from monitor import build_route_index, run

if __name__ == "__main__":
    GTFSInitializer().run()

    if API_KEY == "MISSING_APIKEY":
        print("\n!!! Missing API key. Set WTP_APIKEY in .env file. !!!")
        exit(1)

    print("=== Traffic Management System — WTP Warsaw ===\n")

    gtfs        = GTFSLoader(GTFS_DIR)
    route_index = build_route_index(gtfs)
    client      = WTPClient()

    print("\n=== Real-time monitor (Ctrl+C to stop) ===")
    try:
        run(gtfs, client, route_index)
    except KeyboardInterrupt:
        print("\nStopped.")
