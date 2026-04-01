#!/usr/bin/env python3
"""Fetch state-level economic data from the BEA API.

Fetches two datasets:
1. State-level Real GDP (SAGDP1, LineCode=1)
2. State-level Personal Income Per Capita (SAINC1, LineCode=3)

Years: 2010-2024
Output: CSV files saved to data/raw/bea_state_gdp/

BEA API has strict rate limits; this script uses 0.5s delay between requests.
Estimated runtime: ~12 minutes for 765 requests.
"""

import csv
import json
import logging
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "bea_state_gdp"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# US states + DC FIPS codes
STATE_NAMES = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado',
    'Connecticut', 'Delaware', 'District of Columbia', 'Florida', 'Georgia',
    'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 'Kansas', 'Kentucky',
    'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 'Minnesota',
    'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New Hampshire',
    'New Jersey', 'New Mexico', 'New York', 'North Carolina', 'North Dakota',
    'Ohio', 'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode Island',
    'South Carolina', 'South Dakota', 'Tennessee', 'Texas', 'Utah', 'Vermont',
    'Virginia', 'Washington', 'West Virginia', 'Wisconsin', 'Wyoming',
]

STATE_TO_FIPS = {
    'Alabama': '01000', 'Alaska': '02000', 'Arizona': '04000',
    'Arkansas': '05000', 'California': '06000', 'Colorado': '08000',
    'Connecticut': '09000', 'Delaware': '10000',
    'District of Columbia': '11000', 'Florida': '12000', 'Georgia': '13000',
    'Hawaii': '15000', 'Idaho': '16000', 'Illinois': '17000',
    'Indiana': '18000', 'Iowa': '19000', 'Kansas': '20000',
    'Kentucky': '21000', 'Louisiana': '22000', 'Maine': '23000',
    'Maryland': '24000', 'Massachusetts': '25000', 'Michigan': '26000',
    'Minnesota': '27000', 'Mississippi': '28000', 'Missouri': '29000',
    'Montana': '30000', 'Nebraska': '31000', 'Nevada': '32000',
    'New Hampshire': '33000', 'New Jersey': '34000', 'New Mexico': '35000',
    'New York': '36000', 'North Carolina': '37000', 'North Dakota': '38000',
    'Ohio': '39000', 'Oklahoma': '40000', 'Oregon': '41000',
    'Pennsylvania': '42000', 'Rhode Island': '44000', 'South Carolina': '45000',
    'South Dakota': '46000', 'Tennessee': '47000', 'Texas': '48000',
    'Utah': '49000', 'Vermont': '50000', 'Virginia': '51000',
    'Washington': '53000', 'West Virginia': '54000', 'Wisconsin': '55000',
    'Wyoming': '56000',
}


def read_api_key():
    """Read BEA_API_KEY from .env file."""
    env_path = PROJECT_ROOT / ".env"
    with env_path.open() as f:
        for line in f:
            if line.startswith("BEA_API_KEY="):
                return line.split("=", 1)[1].strip()
    raise ValueError("BEA_API_KEY not found in .env")


def fetch_state_year_value(api_key, table_name, line_code, state_fips, year):
    """Fetch a single state-year data point from BEA API."""
    base_url = "https://apps.bea.gov/api/data"
    params = {
        "UserID": api_key,
        "Method": "GetData",
        "DataSetName": "Regional",
        "TableName": table_name,
        "LineCode": line_code,
        "GeoFips": state_fips,
        "Year": str(year),
        "ResultFormat": "JSON",
    }

    url = f"{base_url}?{urllib.parse.urlencode(params)}"

    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            data = json.loads(response.read().decode('utf-8'))

        if 'BEAAPI' in data and 'Results' in data['BEAAPI']:
            results = data['BEAAPI']['Results']
            if 'Data' in results and results['Data']:
                value = float(results['Data'][0].get('DataValue', 0))
                return value

    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise IOError("Rate limited (429)")
        else:
            raise

    return None


def fetch_bea_table(api_key, table_name, line_code):
    """Fetch all state-year data for a table. Returns dict: state -> year -> value"""
    result = {state: {} for state in STATE_NAMES}
    years = list(range(2010, 2025))

    total = len(STATE_NAMES) * len(years)
    fetched = 0

    log.info(f"\nFetching {table_name} (LineCode={line_code})")
    log.info(f"Requesting {total} data points...")
    log.info("")

    for state_idx, state in enumerate(STATE_NAMES, 1):
        state_fips = STATE_TO_FIPS[state]

        for year_idx, year in enumerate(years, 1):
            fetched += 1
            pct = 100 * fetched // total

            # Show progress
            if fetched % 50 == 0 or fetched == total:
                log.info(f"  [{pct:3d}%] {fetched:3d}/{total} | {state} {year}")

            # Fetch with retries on rate limit
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    value = fetch_state_year_value(api_key, table_name, line_code, state_fips, year)
                    if value is not None:
                        result[state][year] = value
                    break  # Success

                except IOError as e:  # Rate limited
                    if attempt < max_attempts - 1:
                        wait_time = 2 ** (attempt + 1)
                        log.info(f"      Rate limited. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        log.warning(f"      Failed after {max_attempts} attempts")

                except Exception as e:
                    log.warning(f"      Error: {e}")
                    break

            time.sleep(0.5)  # Rate limiting delay

    return result


def write_csv(output_path, data_dict):
    """Write dict to CSV with states as rows, years as columns."""
    years = sorted(set(y for state_data in data_dict.values() for y in state_data.keys()))

    with output_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['state'] + [str(y) for y in years])
        writer.writeheader()

        for state in STATE_NAMES:
            row = {'state': state}
            if state in data_dict:
                for year in years:
                    if year in data_dict[state]:
                        row[str(year)] = f"{data_dict[state][year]:.1f}"
                    else:
                        row[str(year)] = ""
            writer.writerow(row)

    log.info(f"Wrote {output_path.name}")


def main():
    """Main entry point."""
    api_key = read_api_key()

    log.info("=" * 70)
    log.info("BEA State-Level Economic Data Fetcher")
    log.info("=" * 70)

    # Fetch Real GDP
    gdp_data = fetch_bea_table(api_key, "SAGDP1", "1")
    gdp_count = sum(len(years) for years in gdp_data.values())

    # Fetch Personal Income Per Capita
    income_data = fetch_bea_table(api_key, "SAINC1", "3")
    income_count = sum(len(years) for years in income_data.values())

    # Write CSVs
    log.info("\nWriting output files...")
    write_csv(DATA_DIR / "state_gdp_millions.csv", gdp_data)
    write_csv(DATA_DIR / "state_income_per_capita.csv", income_data)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    log.info(f"Real GDP:              {gdp_count:4d} data points")
    log.info(f"Personal Income/Cap:   {income_count:4d} data points")
    log.info(f"Output directory:      {DATA_DIR}")
    log.info("")
    log.info("Next steps:")
    log.info("  - Load CSV files into the fundamentals analysis pipeline")
    log.info("  - Analyze state-level economic heterogeneity in electoral behavior")
    log.info("  - Develop state or type-level fundamentals signals")


if __name__ == "__main__":
    main()
