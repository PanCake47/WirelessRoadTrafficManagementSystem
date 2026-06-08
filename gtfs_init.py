import os
from zipfile import ZipFile, BadZipFile
from urllib.request import urlretrieve
from urllib.error import URLError, HTTPError

from config import GTFS_DIR, GTFS_DOWNLOAD_URL

REQUIRED_FILES = ["calendar.txt", "routes.txt", "stop_times.txt", "stops.txt", "trips.txt"]


class GTFSInitializer:
    
    def __init__(self, gtfs_dir: str = GTFS_DIR, url: str = GTFS_DOWNLOAD_URL):
        self.gtfs_dir = gtfs_dir
        self.url      = url
        self.zip_path = os.path.join(gtfs_dir, "gtfs_data.zip")

    def run(self):
        os.makedirs(self.gtfs_dir, exist_ok=True)
        self._download()
        self._extract()
        self._cleanup()
        self._check_required_files()

    def _download(self):
        print(f"Downloading GTFS data from: {self.url}")
        try:
            def progress(count, block_size, total):
                if total > 0:
                    pct = count * block_size * 100 // total
                    print(f"  {min(pct, 100)}%", end="\r")
            urlretrieve(self.url, self.zip_path, reporthook=progress)
            print("=== Download complete ===\n")
        except HTTPError as e:
            raise SystemExit(f"\n!!! HTTP error {e.code}: {e.reason} !!!")
        except URLError as e:
            raise SystemExit(f"\n!!! Connection error: {e.reason} !!!")

    def _extract(self):
        print(f"Extracting to {self.gtfs_dir}/")
        try:
            with ZipFile(self.zip_path, "r") as zf:
                zf.extractall(self.gtfs_dir)
            print("=== Files extracted ===\n")
        except BadZipFile:
            raise SystemExit("\n!!! Bad zip file !!!")

    def _cleanup(self):
        try:
            os.remove(self.zip_path)
        except OSError as e:
            print(f"\n!!! Couldn't remove temporary file: {e} !!!")

    def _check_required_files(self):
        missing = [f for f in REQUIRED_FILES
                   if not os.path.exists(os.path.join(self.gtfs_dir, f))]
        if missing:
            raise SystemExit(f"\n!!! Missing mandatory files: {missing} !!!")
        print("Files ready to process\n")
