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
LOGISTIC_SATURATION_LIMIT = 85


@dataclass(frozen=True)
class ForecastRow:
    year: int
    exponential_percentage: float | None
    logistic_percentage: float
    combined_percentage: float
    total_passenger_cars: float
    exponential_ev_cars: float | None
    logistic_ev_cars: float
    combined_ev_cars: float


def logistic_curve(year: np.ndarray, carrying_capacity: float, growth: float, midpoint: float):
    """Model EV growth as an S-curve with a maximum adoption level."""
    return carrying_capacity / (1 + np.exp(-growth * (year - midpoint)))


def fit_exponential(years: np.ndarray, percentages: np.ndarray):
    """Fit short-term exponential growth to the historical EV data."""
    slope, intercept = np.polyfit(years, np.log(percentages), 1)
    return intercept, slope


def exponential_curve(year: np.ndarray, intercept: float, slope: float):
    """Predict EV growth with an exponential curve."""
    return np.exp(intercept + slope * year)


def fit_logistic(years: np.ndarray, percentages: np.ndarray):
    """Fit the Logistic S-curve to the given EV adoption data."""
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
    """Fit existing, exponential, Logistic, and combined EV adoption forecasts."""
    years = np.array(sorted(historical_percentages), dtype=float)
    percentages = np.array(
        [historical_percentages[int(year)] for year in years],
        dtype=float,
    )
    predict_years = np.array(forecast_years, dtype=float)

    exponential_params = fit_exponential(years, percentages)
    logistic_params = fit_logistic(years, percentages)
    exponential_predictions = exponential_curve(predict_years, *exponential_params)
    logistic_predictions = logistic_curve(predict_years, *logistic_params)

    combined_fit_data = dict(historical_percentages)
    for year in [2026, 2027]:
        combined_fit_data[year] = float(
            exponential_curve(np.array([year], dtype=float), *exponential_params)[0]
        )
    combined_years = np.array(sorted(combined_fit_data), dtype=float)
    combined_percentages = np.array(
        [combined_fit_data[int(year)] for year in combined_years],
        dtype=float,
    )
    combined_logistic_params = fit_logistic(combined_years, combined_percentages)
    combined_logistic_predictions = logistic_curve(
        predict_years, *combined_logistic_params
    )

    exponential_plot_values: list[float | None] = []
    exponential_has_reached_limit = False
    for value in exponential_predictions:
        if exponential_has_reached_limit:
            exponential_plot_values.append(None)
        elif value >= 60:
            exponential_plot_values.append(60.0)
            exponential_has_reached_limit = True
        else:
            exponential_plot_values.append(float(value))

    combined_predictions = []
    for year, exponential, logistic in zip(
        predict_years,
        exponential_predictions,
        combined_logistic_predictions,
    ):
        if year <= 2027:
            combined_predictions.append(float(exponential))
        else:
            combined_predictions.append(float(logistic))

    return [
        ForecastRow(
            year=int(year),
            exponential_percentage=exponential_plot,
            logistic_percentage=float(logistic),
            combined_percentage=float(combined),
            total_passenger_cars=predict_total_passenger_cars(int(year)),
            exponential_ev_cars=(
                predict_total_passenger_cars(int(year)) * exponential_plot / 100
                if exponential_plot is not None
                else None
            ),
            logistic_ev_cars=(
                predict_total_passenger_cars(int(year)) * float(logistic) / 100
            ),
            combined_ev_cars=(
                predict_total_passenger_cars(int(year)) * float(combined) / 100
            ),
        )
        for year, exponential_plot, logistic, combined in zip(
            predict_years,
            exponential_plot_values,
            logistic_predictions,
            combined_predictions,
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
                "exponential_percentage",
                "logistic_percentage",
                "combined_percentage",
                "exponential_ev_cars_each_year",
                "logistic_ev_cars_each_year",
                "combined_ev_cars_each_year",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "year": row.year,
                    "total_passenger_cars": round(row.total_passenger_cars),
                    "exponential_percentage": (
                        round(row.exponential_percentage, 2)
                        if row.exponential_percentage is not None
                        else ""
                    ),
                    "logistic_percentage": round(row.logistic_percentage, 2),
                    "combined_percentage": round(row.combined_percentage, 2),
                    "exponential_ev_cars_each_year": (
                        round(row.exponential_ev_cars)
                        if row.exponential_ev_cars is not None
                        else ""
                    ),
                    "logistic_ev_cars_each_year": round(row.logistic_ev_cars),
                    "combined_ev_cars_each_year": round(row.combined_ev_cars),
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
    exponential_values = [row.exponential_percentage for row in forecast_rows]
    logistic_values = [row.logistic_percentage for row in forecast_rows]
    combined_values = [row.combined_percentage for row in forecast_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(history_years, history_values, "o-", label="Existing data 2021-2025")
    plt.plot(
        forecast_years,
        exponential_values,
        "o--",
        label="Exponential growth from 2026, stops at 60%",
    )
    plt.plot(forecast_years, logistic_values, "o--", label="Logistic S-curve")
    plt.plot(
        forecast_years,
        combined_values,
        "o-",
        label="Combined: exponential to 2027, then Logistic",
    )
    plt.title("EV Adoption Forecast: Historical, Exponential, Logistic, Combined")
    plt.xlabel("Year")
    plt.ylabel("EV share (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def write_combined_forecast_chart(
    path: Path,
    historical_percentages: dict[int, float],
    forecast_rows: list[ForecastRow],
) -> None:
    """Write a PNG diagram with only historical data and the combined forecast."""
    path.parent.mkdir(parents=True, exist_ok=True)

    history_years = sorted(historical_percentages)
    history_values = [historical_percentages[year] for year in history_years]
    forecast_years = [row.year for row in forecast_rows]
    combined_values = [row.combined_percentage for row in forecast_rows]

    plt.figure(figsize=(9, 5))
    plt.plot(history_years, history_values, "o-", label="Existing data 2021-2025")
    plt.plot(forecast_years, combined_values, "o-", label="Combined forecast")
    plt.title("EV Adoption Forecast: Combined Line")
    plt.xlabel("Year")
    plt.ylabel("EV share (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def main() -> None:
    """Create EV percentage forecasts and export them to CSV and PNG."""
    rows = build_forecast(HISTORICAL_EV_PERCENTAGES, FORECAST_YEARS)
    output_dir = ROOT / "processed data" / "ev adoption"
    out_path = output_dir / "ev_adoption_forecast.csv"
    chart_path = output_dir / "ev_adoption_forecast_compare_lines.png"
    combined_chart_path = output_dir / "ev_adoption_forecast_combined_line.png"
    write_forecast_csv(out_path, rows)
    write_forecast_chart(chart_path, HISTORICAL_EV_PERCENTAGES, rows)
    write_combined_forecast_chart(
        combined_chart_path,
        HISTORICAL_EV_PERCENTAGES,
        rows,
    )
    print(f"Wrote EV adoption forecast to: {out_path}")
    print(f"Wrote EV adoption comparison diagram to: {chart_path}")
    print(f"Wrote EV adoption combined-line diagram to: {combined_chart_path}")


if __name__ == "__main__":
    main()
