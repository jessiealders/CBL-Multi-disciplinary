from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def find_project_root(start: Path) -> Path:
    """Find the project folder from this script location."""
    for folder in [start, *start.parents]:
        if (folder / "Jessie python files").exists():
            return folder
    raise FileNotFoundError("Could not find the project root folder.")


ROOT = find_project_root(Path(__file__).resolve())


HISTORICAL_EV_PERCENTAGES = {
    2021: 3.1,
    2022: 4.4,
    2023: 6.2,
    2024: 7.8,
    2025: 10,
}

FORECAST_YEARS = list(range(2026, 2036))
BASE_TOTAL_PASSENGER_CARS = 1120
BASE_TOTAL_PASSENGER_CARS_YEAR = 2025
PASSENGER_CAR_GROWTH_RATE = 0.02
LOGISTIC_SATURATION_LIMIT = 100


@dataclass(frozen=True)
class ForecastRow:
    year: int
    logistic_percentage: float
    total_passenger_cars: float
    logistic_ev_cars: float


def logistic_curve(year: np.ndarray, carrying_capacity: float, growth: float, midpoint: float):
    """Model EV growth as an S-curve with a maximum adoption level."""
    return carrying_capacity / (1 + np.exp(-growth * (year - midpoint)))


def fit_logistic(years: np.ndarray, percentages: np.ndarray):
    """Fit the Logistic S-curve to the 2021-2025 EV data."""
    initial_guess = [LOGISTIC_SATURATION_LIMIT, 0.5, years.mean()]
    bounds = (
        [max(percentages), 0.001, years.min() - 20],
        [LOGISTIC_SATURATION_LIMIT, 5.0, years.max() + 30],
    )
    params, _ = curve_fit(
        logistic_curve,
        years,
        percentages,
        p0=initial_guess,
        bounds=bounds,
        maxfev=20_000,
    )
    return params


def predict_total_passenger_cars(year: int) -> float:
    """Predict total passenger cars with 2% yearly growth."""
    years_after_base = year - BASE_TOTAL_PASSENGER_CARS_YEAR
    return BASE_TOTAL_PASSENGER_CARS * (1 + PASSENGER_CAR_GROWTH_RATE) ** years_after_base


def build_forecast(
    historical_percentages: dict[int, float],
    forecast_years: list[int],
) -> list[ForecastRow]:
    """Fit the Logistic model and return future EV percentage predictions."""
    years = np.array(sorted(historical_percentages), dtype=float)
    percentages = np.array(
        [historical_percentages[int(year)] for year in years],
        dtype=float,
    )
    predict_years = np.array(forecast_years, dtype=float)

    logistic_params = fit_logistic(years, percentages)

    logistic_predictions = logistic_curve(predict_years, *logistic_params)

    return [
        ForecastRow(
            year=int(year),
            logistic_percentage=float(logistic),
            total_passenger_cars=predict_total_passenger_cars(int(year)),
            logistic_ev_cars=predict_total_passenger_cars(int(year)) * float(logistic) / 100,
        )
        for year, logistic in zip(
            predict_years,
            logistic_predictions,
        )
    ]


def write_forecast_csv(path: Path, rows: list[ForecastRow]) -> None:
    """Write the forecast rows to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        f = path.open("w", newline="", encoding="utf-8")
    except PermissionError as exc:
        raise PermissionError(
            f"Cannot write {path}. Close the CSV if it is open in Excel, then run again."
        ) from exc

    with f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "total_passenger_cars",
                "logistic_percentage",
                "logistic_ev_cars_each_year",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "year": row.year,
                    "total_passenger_cars": round(row.total_passenger_cars),
                    "logistic_percentage": round(row.logistic_percentage, 2),
                    "logistic_ev_cars_each_year": round(row.logistic_ev_cars),
                }
            )


def write_forecast_chart(
    path: Path,
    historical_percentages: dict[int, float],
    forecast_rows: list[ForecastRow],
) -> None:
    """Write a PNG diagram with historical data and forecast curves."""
    path.parent.mkdir(parents=True, exist_ok=True)

    history_years = sorted(historical_percentages)
    history_values = [historical_percentages[year] for year in history_years]
    forecast_years = [row.year for row in forecast_rows]
    logistic_values = [row.logistic_percentage for row in forecast_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(history_years, history_values, "o-", label="Historical data")
    plt.plot(forecast_years, logistic_values, "o--", label="Logistic S-curve")
    plt.title("EV Adoption Forecast - Logistic S-curve")
    plt.xlabel("Year")
    plt.ylabel("EV share (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    """Create Logistic EV percentage forecasts and export them to CSV."""
    rows = build_forecast(HISTORICAL_EV_PERCENTAGES, FORECAST_YEARS)
    output_dir = ROOT / "processed data" / "ev adoption"
    out_path = output_dir / "ev_adoption_forecast.csv"
    chart_path = output_dir / "ev_adoption_forecast.png"
    write_forecast_csv(out_path, rows)
    write_forecast_chart(chart_path, HISTORICAL_EV_PERCENTAGES, rows)
    print(f"Wrote EV adoption forecast to: {out_path}")
    print(f"Wrote EV adoption diagram to: {chart_path}")


if __name__ == "__main__":
    main()
