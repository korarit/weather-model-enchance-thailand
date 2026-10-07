import os
import sys
import time
import urllib.parse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import logging
from typing import List, Dict, Any, Optional, Tuple
import datetime
import numpy as np
import pandas as pd
import requests

from src.config.paths import (
    DEFAULT_RAW_NWP_DIR,
    resolve_forecast_paths,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_RAW_DIR = DEFAULT_RAW_NWP_DIR
DEFAULT_CDS_URL = "https://cds.climate.copernicus.eu/api"

TIGGE_ORIGINS = {
    "ecmf": "ECMWF IFS (Europe)",
    "ecmwf": "ECMWF IFS (Europe)",
    "kwbc": "NOAA NCEP GFS (USA)",
    "ncep": "NOAA NCEP GFS (USA)",
    "cwao": "CMC GEM Global (Canada)",
    "cmc": "CMC GEM Global (Canada)",
    "eccc": "CMC GEM Global (Canada)",
    "ammc": "BoM ACCESS-G (Australia)",
    "bom": "BoM ACCESS-G (Australia)",
    "edzw": "DWD ICON Global (Germany)",
    "dwd": "DWD ICON Global (Germany)",
    "lfpw": "Météo-France ARPEGE (France)",
    "mf": "Météo-France ARPEGE (France)",
    "meteo_france": "Météo-France ARPEGE (France)",
}

CDS_ORIGIN_MAP = {
    "ecmf": "ecmwf",
    "ecmwf": "ecmwf",
    "kwbc": "ncep",
    "ncep": "ncep",
    "cwao": "eccc",
    "cmc": "eccc",
    "eccc": "eccc",
    "ammc": "bom",
    "bom": "bom",
    "edzw": "dwd",
    "dwd": "dwd",
    "lfpw": "mf",
    "mf": "mf",
    "meteo_france": "mf",
}

# Thailand Bounding Box [North, West, South, East]
THAILAND_BBOX = [21.0, 97.0, 5.0, 106.0]

FORECAST_CYCLES = ["00:00", "12:00"]
LEAD_STEPS = ["6", "12", "18", "24", "30", "36", "42", "48"]

TIGGE_VARIABLES = [
    "total_precipitation",
    "surface_pressure",
    "2_m_temperature",
    "10_m_u_component_of_wind",
    "10_m_v_component_of_wind",
    "convective_available_potential_energy",
    "total_column_water",
]


def check_cds_configured(key: Optional[str] = None) -> bool:
    """Checks whether CDS key is configured via argument, environment variable, or ~/.cdsapirc."""
    if key:
        return True
    if os.environ.get("CDSAPI_KEY"):
        return True
    home_rc = Path.home() / ".cdsapirc"
    if home_rc.exists():
        return True
    return False


def build_tigge_request(
    origin: str,
    date_str: str,
    cycle: str,
    steps: List[str] = LEAD_STEPS,
    area: List[float] = THAILAND_BBOX,
    forecast_type: str = "control_forecast",
    variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Constructs API payload for modern Copernicus CDS dataset 'tigge-forecasts'.
    """
    cds_origin = CDS_ORIGIN_MAP.get(origin.lower(), origin.lower())
    dt = pd.Timestamp(date_str)
    time_str = cycle if ":" in cycle else f"{int(cycle):02d}:00"

    var_list = variables or TIGGE_VARIABLES
    var_clean = []
    var_fix = {
        "2m_temperature": "2_m_temperature",
        "10m_u_component_of_wind": "10_m_u_component_of_wind",
        "10m_v_component_of_wind": "10_m_v_component_of_wind",
    }
    for v in var_list:
        var_clean.append(var_fix.get(v, v))

    return {
        "origin": cds_origin,
        "year": f"{dt.year:04d}",
        "month": f"{dt.month:02d}",
        "day": f"{dt.day:02d}",
        "time": time_str,
        "level_type": "single_level",
        "variable": var_clean,
        "forecast_type": forecast_type,
        "leadtime_hour": [str(int(s)) for s in steps],
        "data_format": "grib",
        "area": area,
    }


class CDSRestClient:
    """
    Direct REST API client for Copernicus / ECMWF Data Store.
    Implements OGC API - Processes retrieval natively via `requests` without `cdsapi`.
    """
    def __init__(self, key: Optional[str] = None, url: Optional[str] = None):
        self.url, self.key = self._resolve_credentials(url, key)
        self.session = requests.Session()
        self.headers = {
            "User-Agent": "weather-forecast-enhance/1.0",
        }
        if self.key:
            if ":" in self.key:
                uid, secret = self.key.split(":", 1)
                self.session.auth = (uid.strip(), secret.strip())
            else:
                self.headers["PRIVATE-TOKEN"] = self.key.strip()

    @staticmethod
    def _resolve_credentials(url: Optional[str], key: Optional[str]) -> Tuple[str, Optional[str]]:
        if not key:
            key = os.environ.get("CDSAPI_KEY")
        if not url:
            url = os.environ.get("CDSAPI_URL")

        if not key or not url:
            dotrc = os.environ.get("CDSAPI_RC", Path.home() / ".cdsapirc")
            if Path(dotrc).exists():
                try:
                    with open(dotrc, "r", encoding="utf-8") as f:
                        for line in f:
                            if ":" in line:
                                k, v = line.strip().split(":", 1)
                                k = k.strip().lower()
                                v = v.strip()
                                if k == "key" and not key:
                                    key = v
                                elif k == "url" and not url:
                                    url = v
                except Exception as e:
                    logger.debug("Failed parsing %s: %s", dotrc, e)

        final_url = url or DEFAULT_CDS_URL
        return final_url.rstrip("/"), key

    def retrieve(self, dataset: str, request_params: Dict[str, Any], target_file: Path) -> Path:
        """Submits an asynchronous process execution job, polls until complete, and downloads GRIB."""
        if not self.key:
            raise ValueError("No CDS credentials provided. Please supply --key or configure CDSAPI_KEY / ~/.cdsapirc.")

        target_file.parent.mkdir(parents=True, exist_ok=True)

        endpoints = []
        if "tigge" in dataset.lower():
            # TIGGE is archived on ECMWF Climate Data Store (ECDS)
            endpoints.append(f"https://ecds.ecmwf.int/api/retrieve/v1/processes/{dataset}/execution")
            if "ecds.ecmwf.int" not in self.url:
                endpoints.append(f"{self.url}/retrieve/v1/processes/{dataset}/execution")
        else:
            endpoints.append(f"{self.url}/retrieve/v1/processes/{dataset}/execution")
            if "cds.climate.copernicus.eu" in self.url:
                endpoints.append(f"https://ecds.ecmwf.int/api/retrieve/v1/processes/{dataset}/execution")

        exec_resp = None
        last_error = None
        for ep in endpoints:
            try:
                logger.info("Submitting CDS retrieval job to %s...", ep)
                r = self.session.post(ep, json={"inputs": request_params}, headers=self.headers, timeout=60)
                if r.status_code == 404:
                    logger.warning("Dataset '%s' not found on %s (HTTP 404). Trying fallback endpoint...", dataset, ep)
                    continue
                r.raise_for_status()
                exec_resp = r
                break
            except Exception as e:
                last_error = e

        if exec_resp is None:
            raise RuntimeError(f"Failed to submit CDS retrieval job: {last_error}")

        job_info = exec_resp.json()
        job_id = job_info.get("jobID")

        # Determine job monitoring URL (must NOT be the POST execution endpoint)
        monitor_url = None
        for link in job_info.get("links", []):
            rel = link.get("rel")
            href = link.get("href", "")
            if rel in ("monitor", "status") and not href.endswith("/execution"):
                monitor_url = href
                break

        if not monitor_url and job_id:
            # Correctly construct OGC API jobs endpoint: {base_prefix}/jobs/{job_id}
            parsed = urllib.parse.urlparse(exec_resp.url)
            base_prefix = parsed.path.split("/processes")[0]
            monitor_url = f"{parsed.scheme}://{parsed.netloc}{base_prefix}/jobs/{job_id}"

        if not monitor_url:
            raise RuntimeError(f"Could not determine job monitor URL from response: {job_info}")

        logger.info("CDS Job submitted successfully (ID: %s). Monitoring status at %s...", job_id, monitor_url)

        # Poll status
        sleep = 2.0
        max_sleep = 30.0
        while True:
            poll_resp = self.session.get(monitor_url, headers=self.headers, timeout=60)
            poll_resp.raise_for_status()
            poll_data = poll_resp.json()
            status = poll_data.get("status")

            if status == "successful":
                logger.info("CDS Job [%s] completed successfully!", job_id)
                break
            elif status in ("accepted", "running"):
                logger.info("CDS Job [%s] is %s... waiting %.1fs", job_id, status, sleep)
                time.sleep(sleep)
                sleep = min(sleep * 1.5, max_sleep)
            elif status in ("failed", "rejected", "dismissed"):
                detail = poll_data.get("detail") or poll_data.get("title") or poll_data
                raise RuntimeError(f"CDS Job [{job_id}] failed with status '{status}': {detail}")
            else:
                logger.debug("Job status [%s], waiting %.1fs", status, sleep)
                time.sleep(sleep)

        # Get results
        results_url = None
        for link in poll_data.get("links", []):
            if link.get("rel") == "results":
                results_url = link.get("href")
                break
        if not results_url:
            results_url = f"{monitor_url.rstrip('/')}/results"

        res_resp = self.session.get(results_url, headers=self.headers, timeout=60)
        res_resp.raise_for_status()
        asset = res_resp.json().get("asset", {}).get("value", {})
        download_href = asset.get("href")
        if not download_href:
            raise RuntimeError(f"No downloadable asset link found in results: {res_resp.text}")

        download_url = urllib.parse.urljoin(results_url, download_href)
        logger.info("Downloading GRIB from %s to %s...", download_url, target_file)

        with self.session.get(download_url, headers=self.headers, stream=True, timeout=300) as stream_resp:
            stream_resp.raise_for_status()
            with open(target_file, "wb") as f:
                for chunk in stream_resp.iter_content(chunk_size=65536):
                    if chunk:
                        f.write(chunk)

        logger.info("Downloaded %s successfully (%d bytes)", target_file, target_file.stat().st_size)
        return target_file


def generate_synthetic_nwp_payload(
    origin: str,
    run_date: str,
    cycle: str,
    steps: List[str],
    output_dir: Path
) -> Path:
    """
    Generates realistic, meteorologically-bounded synthetic NWP cycle
    for local dev testing without hitting external network / quota limits.
    """
    cycle_tag = cycle.replace(":", "")
    run_dt_str = f"{run_date.replace('-', '')}_{cycle_tag[:2]}z"
    target_dir = output_dir / origin
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / f"run_{run_dt_str}_synthetic.parquet"

    run_time = pd.Timestamp(f"{run_date} {cycle}:00")

    # Generate a grid across Thailand
    lats = np.arange(5.5, 21.0, 0.5)
    lons = np.arange(97.0, 106.0, 0.5)
    grid_points = [(lat, lon) for lat in lats for lon in lons]

    records = []
    np.random.seed(int(pd.Timestamp(run_date).timestamp()) % 10000)

    for step_str in steps:
        step_h = int(step_str)
        valid_time = run_time + pd.Timedelta(hours=step_h)

        for lat, lon in grid_points:
            # Physics-based baseline with elevation/latitude trends
            t2m_c = 28.0 - (lat - 10.0) * 0.4 + np.random.normal(0, 1.5)
            sp_hpa = 1010.0 - (lat - 10.0) * 0.2 + np.random.normal(0, 1.0)
            u10 = np.random.normal(1.5, 3.0)
            v10 = np.random.normal(2.0, 3.0)
            cape = max(0.0, float(np.random.exponential(450.0)))

            # Precipitation (intermittent convective showers)
            rain_prob = 0.35 if (lat > 12.0) else 0.45
            if np.random.rand() < rain_prob:
                tp_mm = float(np.random.gamma(shape=1.5, scale=4.0))
            else:
                tp_mm = 0.0

            records.append({
                "origin": origin,
                "run_time": run_time,
                "valid_time": valid_time,
                "lead_time_hours": step_h,
                "lat": float(round(lat, 2)),
                "lon": float(round(lon, 2)),
                "tp": float(round(tp_mm, 2)),
                "sp": float(round(sp_hpa, 2)),
                "t2m": float(round(t2m_c, 2)),
                "u10": float(round(u10, 2)),
                "v10": float(round(v10, 2)),
                "cape": float(round(cape, 1)),
            })

    df = pd.DataFrame(records)
    df.to_parquet(out_file, index=False)
    logger.info("Generated synthetic NWP payload: %s (%d rows)", out_file, len(df))
    return out_file


def build_tigge_monthly_request(
    origin: str,
    year: str,
    month: str,
    days: List[str],
    cycles: List[str] = FORECAST_CYCLES,
    steps: List[str] = LEAD_STEPS,
    area: List[float] = THAILAND_BBOX,
    forecast_type: str = "control_forecast",
    variables: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Constructs an aggregated monthly API payload for CDS/ECDS dataset 'tigge-forecasts'.
    Batches multiple days and cycles into a single supercomputer extraction job.
    """
    cds_origin = CDS_ORIGIN_MAP.get(origin.lower(), origin.lower())
    time_list = [c if ":" in c else f"{int(c):02d}:00" for c in cycles]

    var_list = variables or TIGGE_VARIABLES
    var_clean = []
    var_fix = {
        "2m_temperature": "2_m_temperature",
        "10m_u_component_of_wind": "10_m_u_component_of_wind",
        "10m_v_component_of_wind": "10_m_v_component_of_wind",
    }
    for v in var_list:
        var_clean.append(var_fix.get(v, v))

    return {
        "origin": cds_origin,
        "year": str(year),
        "month": f"{int(month):02d}",
        "day": [f"{int(d):02d}" for d in sorted(list(set(days)))],
        "time": time_list,
        "level_type": "single_level",
        "variable": var_clean,
        "forecast_type": forecast_type,
        "leadtime_hour": [str(int(s)) for s in steps],
        "data_format": "grib",
        "area": area,
    }


def unpack_monthly_grib(monthly_grib: Path, output_dir: Path, origin: str) -> List[Path]:
    """
    Splits a consolidated monthly GRIB2 file into standard daily cycle files:
    tigge_{origin}_{YYYYMMDD}_{HHz}.grib
    Pure Python stream splitting without external C/library dependencies.
    """
    target_dir = output_dir / origin
    target_dir.mkdir(parents=True, exist_ok=True)
    created_files = set()
    open_handles = {}

    try:
        with open(monthly_grib, "rb") as f:
            data = f.read()

        offset = 0
        total_len = len(data)
        while offset < total_len - 16:
            grib_idx = data.find(b"GRIB", offset)
            if grib_idx == -1:
                break

            edition = data[grib_idx + 7]
            if edition == 2:
                msg_len = int.from_bytes(data[grib_idx + 8 : grib_idx + 16], "big")
                if msg_len <= 0 or grib_idx + msg_len > total_len:
                    offset = grib_idx + 4
                    continue

                # Parse section 1 reference date
                year = int.from_bytes(data[grib_idx + 28 : grib_idx + 30], "big")
                month = data[grib_idx + 30]
                day = data[grib_idx + 31]
                hour = data[grib_idx + 32]
                run_tag = f"{year:04d}{month:02d}{day:02d}_{hour:02d}z"
                cycle_file = target_dir / f"tigge_{origin}_{run_tag}.grib"

                if cycle_file not in open_handles:
                    open_handles[cycle_file] = open(cycle_file, "ab")
                    created_files.add(cycle_file)

                open_handles[cycle_file].write(data[grib_idx : grib_idx + msg_len])
                offset = grib_idx + msg_len
            else:
                # GRIB1 format support
                msg_len = int.from_bytes(data[grib_idx + 4 : grib_idx + 7], "big")
                if msg_len <= 0 or grib_idx + msg_len > total_len:
                    offset = grib_idx + 4
                    continue
                offset = grib_idx + msg_len
    finally:
        for fh in open_handles.values():
            fh.close()

    logger.info("Unpacked monthly GRIB into %d daily cycle files in %s", len(created_files), target_dir)
    return sorted(list(created_files))


def download_tigge_cycle(
    origin: str,
    date_str: str,
    cycle: str,
    output_dir: Path = DEFAULT_RAW_DIR,
    use_synthetic_fallback: bool = True,
    forecast_type: str = "control_forecast",
    dataset: str = "tigge-forecasts",
    key: Optional[str] = None,
    url: str = DEFAULT_CDS_URL,
) -> Path:
    """
    Downloads or retrieves one NWP forecast cycle via direct CDS/ECDS REST API.
    If CDS credentials are not present or error occurs, uses synthetic fallback in dev mode.
    """
    cycle_tag = cycle.replace(":", "")
    run_dt_str = f"{date_str.replace('-', '')}_{cycle_tag[:2]}z"
    target_dir = output_dir / origin
    target_dir.mkdir(parents=True, exist_ok=True)
    target_grib = target_dir / f"tigge_{origin}_{run_dt_str}.grib"

    if target_grib.exists():
        logger.info("Found cached GRIB file: %s", target_grib)
        return target_grib

    if check_cds_configured(key):
        try:
            client = CDSRestClient(url=url, key=key)
            request_payload = build_tigge_request(
                origin=origin,
                date_str=date_str,
                cycle=cycle,
                forecast_type=forecast_type,
            )
            logger.info("Retrieving from CDS API (%s): dataset='%s', origin='%s', date='%s', cycle='%s'",
                        client.url, dataset, request_payload.get("origin"), date_str, cycle)
            client.retrieve(dataset, request_payload, target_grib)
            logger.info("Successfully downloaded TIGGE GRIB to %s", target_grib)
            return target_grib
        except Exception as e:
            logger.warning("CDS download failed for %s (%s %s). Fallback triggered: %s", origin, date_str, cycle, e)
            if not use_synthetic_fallback:
                raise e
    else:
        logger.info("No CDS credentials found (--key, ~/.cdsapirc or CDSAPI_KEY). Using synthetic fallback.")

    # Fallback to dev synthetic generator
    logger.info("Generating dev sample NWP dataset for %s (%s %s)", origin, date_str, cycle)
    return generate_synthetic_nwp_payload(origin, date_str, cycle, LEAD_STEPS, output_dir)


def download_tigge_month(
    origin: str,
    year: str,
    month: str,
    days: List[str],
    cycles: List[str] = FORECAST_CYCLES,
    output_dir: Path = DEFAULT_RAW_DIR,
    use_synthetic_fallback: bool = True,
    forecast_type: str = "control_forecast",
    dataset: str = "tigge-forecasts",
    key: Optional[str] = None,
    url: str = DEFAULT_CDS_URL,
) -> List[Path]:
    """
    Downloads a full month of forecast cycles in 1 single CDS request,
    then automatically unpacks it into daily cycle files.
    """
    target_dir = output_dir / origin
    target_dir.mkdir(parents=True, exist_ok=True)

    # Check if all daily cycle files are already present
    expected_cycle_files = []
    all_cached = True
    for d_str in days:
        date_full = f"{year}-{month}-{int(d_str):02d}"
        for c in cycles:
            c_tag = c.replace(":", "")[:2]
            run_tag = f"{date_full.replace('-', '')}_{c_tag}z"
            grib_f = target_dir / f"tigge_{origin}_{run_tag}.grib"
            pq_f = target_dir / f"run_{run_tag}_synthetic.parquet"
            if grib_f.exists():
                expected_cycle_files.append(grib_f)
            elif pq_f.exists():
                expected_cycle_files.append(pq_f)
            else:
                all_cached = False

    if all_cached and expected_cycle_files:
        logger.info("Found cached cycles for %s %s-%s (%d files), skipping download.", origin, year, month, len(expected_cycle_files))
        return expected_cycle_files

    month_tag = f"{year}{int(month):02d}"
    monthly_grib = target_dir / f"tigge_{origin}_{month_tag}_monthly.grib"

    if check_cds_configured(key):
        try:
            client = CDSRestClient(url=url, key=key)
            req = build_tigge_monthly_request(
                origin=origin,
                year=year,
                month=month,
                days=days,
                cycles=cycles,
                forecast_type=forecast_type,
            )
            logger.info("Submitting monthly batch request: %s %s-%s (%d days, %d cycles) to CDS/ECDS...",
                        origin, year, month, len(days), len(cycles))
            client.retrieve(dataset, req, monthly_grib)
            logger.info("Downloaded monthly GRIB (%d bytes). Unpacking into daily cycles...", monthly_grib.stat().st_size)

            unpacked = unpack_monthly_grib(monthly_grib, output_dir, origin)
            # Remove intermediate raw monthly file to conserve disk space
            if monthly_grib.exists() and unpacked:
                monthly_grib.unlink()
            return unpacked
        except Exception as e:
            logger.warning("Monthly CDS download failed for %s (%s-%s): %s. Fallback triggered.", origin, year, month, e)
            if not use_synthetic_fallback:
                raise e
    else:
        logger.info("No CDS credentials found. Generating synthetic NWP cycles for %s-%s.", year, month)

    # Fallback: Generate daily synthetic payloads for the month
    fallback_files = []
    for d_str in days:
        date_full = f"{year}-{month}-{int(d_str):02d}"
        for c in cycles:
            p = generate_synthetic_nwp_payload(origin, date_full, c, LEAD_STEPS, output_dir)
            fallback_files.append(p)
    return fallback_files


def run_batch_acquisition(
    dates: List[str],
    origins: List[str],
    cycles: List[str] = FORECAST_CYCLES,
    output_dir: Path = DEFAULT_RAW_DIR,
    use_synthetic_fallback: bool = True,
    forecast_type: str = "control_forecast",
    key: Optional[str] = None,
    url: str = DEFAULT_CDS_URL,
    batch_mode: str = "monthly",
) -> List[Path]:
    """
    Runs batch NWP forecast cycle acquisition across dates and origins.
    By default (batch_mode='monthly'), groups requests by calendar month,
    drastically reducing CDS queue wait time from thousands of jobs to ~1 job per month.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if batch_mode == "monthly":
        from collections import defaultdict
        # Group dates by (year, month) -> list of day strings
        months_map = defaultdict(list)
        for d in dates:
            dt = pd.Timestamp(d)
            y_str = f"{dt.year:04d}"
            m_str = f"{dt.month:02d}"
            d_str = f"{dt.day:02d}"
            months_map[(y_str, m_str)].append(d_str)

        total_tasks = len(months_map) * len(origins)
        logger.info("Starting NWP Acquisition (MONTHLY BATCH MODE): %d months, %d origins (Total: %d CDS Jobs)",
                    len(months_map), len(origins), total_tasks)

        output_files = []
        completed = 0
        for (y_str, m_str), d_list in sorted(months_map.items()):
            for orig in origins:
                completed += 1
                pct = (completed / total_tasks) * 100
                logger.info("[NWP Progress: %d/%d (%.1f%%)] Fetching %s | Month: %s-%s (%d days, %d cycles)...",
                            completed, total_tasks, pct, orig, y_str, m_str, len(d_list), len(cycles))
                files = download_tigge_month(
                    origin=orig,
                    year=y_str,
                    month=m_str,
                    days=d_list,
                    cycles=cycles,
                    output_dir=output_dir,
                    use_synthetic_fallback=use_synthetic_fallback,
                    forecast_type=forecast_type,
                    key=key,
                    url=url,
                )
                output_files.extend(files)
        logger.info("NWP Monthly Acquisition finished: %d cycle files ready across %d months at %s",
                    len(output_files), len(months_map), output_dir)
        return output_files

    else:
        # Legacy daily cycle-by-cycle mode
        total_tasks = len(dates) * len(origins) * len(cycles)
        logger.info("Starting NWP Acquisition (DAILY MODE): %d dates, %d origins, %d cycles (Total: %d tasks)",
                    len(dates), len(origins), len(cycles), total_tasks)
        output_files = []
        completed = 0
        for d in dates:
            for orig in origins:
                for c in cycles:
                    completed += 1
                    pct = (completed / total_tasks) * 100
                    logger.info("[NWP Progress: %d/%d (%.1f%%)] Fetching %s | Date: %s | Cycle: %s...",
                                completed, total_tasks, pct, orig, d, c)
                    p = download_tigge_cycle(
                        origin=orig,
                        date_str=d,
                        cycle=c,
                        output_dir=output_dir,
                        use_synthetic_fallback=use_synthetic_fallback,
                        forecast_type=forecast_type,
                        key=key,
                        url=url,
                    )
                    output_files.append(p)
        logger.info("NWP Acquisition finished: %d/%d forecast files ready at %s", len(output_files), total_tasks, output_dir)
        return output_files


def main():
    parser = argparse.ArgumentParser(description="TIGGE NWP Forecast Cycles Downloader (Direct REST API & Monthly Batch)")
    parser.add_argument("--key", type=str, default=None, help="Copernicus CDS API Key / Personal Access Token")
    parser.add_argument("--url", type=str, default=DEFAULT_CDS_URL, help=f"Copernicus CDS API URL (default: {DEFAULT_CDS_URL})")
    parser.add_argument("--forecast-dir", "--nwp-dir", dest="forecast_dir", type=str, default=None, help="Base directory for NWP forecast data (e.g. E:/data/weather_nwp)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output raw directory (default: {forecast-dir}/raw/nwp_runs or data/raw/nwp_runs)")
    parser.add_argument("--sample-days", type=int, default=2, help="Sample days for dev mode")
    parser.add_argument("--start", type=str, default="2021-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    parser.add_argument("--origins", type=str, default="ecmf,kwbc", help="Comma-separated TIGGE origins (e.g. ecmf,kwbc,cwao or ecmwf,ncep)")
    parser.add_argument("--forecast-type", type=str, default="control_forecast", choices=["control_forecast", "perturbed_forecast"], help="CDS forecast type")
    parser.add_argument("--batch-by", type=str, default="month", choices=["month", "day"], help="Batch requests by 'month' (fastest, reduces queue jobs ~60x) or 'day'")
    parser.add_argument("--no-fallback", action="store_true", help="Raise error if CDS download fails instead of generating synthetic data")
    parser.add_argument("--full", action="store_true", help="Run full multi-year download (Server production)")
    args = parser.parse_args()

    paths = resolve_forecast_paths(forecast_dir=args.forecast_dir, raw_nwp_dir=args.output_dir)
    output_dir = paths["raw_nwp_dir"]

    origin_list = [o.strip() for o in args.origins.split(",") if o.strip() in TIGGE_ORIGINS]
    if not origin_list:
        origin_list = ["ecmf", "kwbc"]

    start_date = pd.Timestamp(args.start)
    if args.end:
        end_date = pd.Timestamp(args.end)
        days = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()
    elif args.full:
        end_date = pd.Timestamp("2024-12-31")
        days = pd.date_range(start_date, end_date, freq="D").strftime("%Y-%m-%d").tolist()
    else:
        days = pd.date_range(start_date, periods=args.sample_days, freq="D").strftime("%Y-%m-%d").tolist()

    run_batch_acquisition(
        dates=days,
        origins=origin_list,
        output_dir=output_dir,
        use_synthetic_fallback=not args.no_fallback,
        forecast_type=args.forecast_type,
        key=args.key,
        url=args.url,
        batch_mode="monthly" if args.batch_by == "month" else "daily",
    )


if __name__ == "__main__":
    main()
