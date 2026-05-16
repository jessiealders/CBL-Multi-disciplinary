from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


# Safety cap: we don't want to go above this share of the maximum capacity.
MAX_UTILIZATION = 0.85


def p(rel_windows_path: str) -> Path:
    """This is to make sure the backlashes work on windows and linux OS as well."""
    return ROOT.joinpath(*rel_windows_path.split("\\"))


@dataclass(frozen=True)
class EvPortAssumptions:
    """Assumptions to convert MW headroom into number of EV charging ports."""

    port_power_kw: float = 11.0  # typical AC public/parking charger (kW)
    diversity_factor: float = 0.7  # not all ports run at full power simultaneously

    @property
    def effective_port_kw(self) -> float:
        return self.port_power_kw * self.diversity_factor


def estimate_ports_from_remaining_mw(remaining_mw: float, assumptions: EvPortAssumptions) -> int:
    if remaining_mw <= 0:
        return 0
    remaining_kw = remaining_mw * 1000.0
    return int(remaining_kw // assumptions.effective_port_kw)


def _read_zonal_demand(demand_path: Path) -> pd.DataFrame:
    demand = pd.read_csv(demand_path)
    required_cols_demand = {"timestamp", "zone_id", "demand_MW"}
    if not required_cols_demand.issubset(demand.columns):
        raise ValueError(f"Demand CSV must contain columns {sorted(required_cols_demand)}")
    return demand


def _aggregate_zone_demand(demand: pd.DataFrame) -> pd.DataFrame:
    return (
        demand.groupby("zone_id")
        .agg(
            peak_demand_MW=("demand_MW", "max"),
            avg_demand_MW=("demand_MW", "mean"),
        )
        .reset_index()
    )


def run_mode_manual_zone_capacity(demand_path: Path, capacity_path: Path) -> None:
    if not capacity_path.exists():
        raise FileNotFoundError(
            "Missing capacity file. Create it at: "
            f"{capacity_path}\n"
            "With columns: zone_id,capacity_MW"
        )

    demand = _read_zonal_demand(demand_path)
    capacities = pd.read_csv(capacity_path)

    required_cols_cap = {"zone_id", "capacity_MW"}
    if not required_cols_cap.issubset(capacities.columns):
        raise ValueError(f"Capacity CSV must contain columns {sorted(required_cols_cap)}")

    agg = _aggregate_zone_demand(demand)
    merged = agg.merge(capacities, on="zone_id", how="left")

    missing = merged[merged["capacity_MW"].isna()]["zone_id"].tolist()
    if missing:
        raise ValueError(
            "Capacity missing for zones: "
            + ", ".join(missing)
            + "\nAdd them to other data/zone_grid_capacity.csv"
        )

    # Safety cap: only allow using up to MAX_UTILIZATION of the (max) capacity.
    merged["usable_capacity_MW"] = merged["capacity_MW"] * MAX_UTILIZATION
    merged["remaining_MW_at_peak"] = merged["usable_capacity_MW"] - merged["peak_demand_MW"]
    merged["remaining_MW_at_avg"] = merged["usable_capacity_MW"] - merged["avg_demand_MW"]

    assumptions = EvPortAssumptions()
    merged["extra_ports_at_peak"] = merged["remaining_MW_at_peak"].apply(
        lambda mw: estimate_ports_from_remaining_mw(mw, assumptions)
    )
    merged["extra_ports_at_avg"] = merged["remaining_MW_at_avg"].apply(
        lambda mw: estimate_ports_from_remaining_mw(mw, assumptions)
    )

    out = merged.sort_values("remaining_MW_at_peak", ascending=False)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 140)

    print("\nRemaining grid capacity estimate (per zone, manual capacities)\n")
    print(f"Max utilization cap applied: {MAX_UTILIZATION:.0%}")
    print(
        out[
            [
                "zone_id",
                "capacity_MW",
                "usable_capacity_MW",
                "peak_demand_MW",
                "remaining_MW_at_peak",
                "extra_ports_at_peak",
                "avg_demand_MW",
                "remaining_MW_at_avg",
                "extra_ports_at_avg",
            ]
        ].to_string(index=False)
    )

    output_csv = p(r"other data\zone_remaining_capacity_estimates.csv")
    out.to_csv(output_csv, index=False)
    print(f"\nWrote: {output_csv}")


def _parse_congestie_numeric(series: pd.Series) -> pd.Series:
    # Data uses comma decimals (e.g., '1,0'). Convert robustly.
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def run_mode_estimated_from_congestion(
    congestie_path: Path,
    baseline_demand_mw: float,
    headroom_factors: dict[int, float],
) -> None:
    """Estimate remaining MW per postcode from congestion classes.

    Contract:
    - baseline_demand_mw is a single "typical" demand value to apply everywhere for now.
      (Until zone<->postcode mapping exists.)
    - headroom_factors maps congestion class (0..3) -> multiplier for capacity vs baseline.
      Example: class 0 -> 1.30, meaning capacity = 1.30 * baseline.
    """

    df = pd.read_csv(congestie_path, sep=";", dtype={"postcode": str})
    required = {"postcode", "afname", "opwek", "voedingsgebied_id", "voedingsgebied_naam"}
    if not required.issubset(df.columns):
        raise ValueError(f"congestie_pc6.csv must contain columns {sorted(required)}")

    df["afname_class"] = _parse_congestie_numeric(df["afname"]).round().astype("Int64")
    df["opwek_class"] = _parse_congestie_numeric(df["opwek"]).round().astype("Int64")
    df["congestion_class"] = df[["afname_class", "opwek_class"]].max(axis=1)

    # In case of missing/invalid values, drop those rows.
    df = df.dropna(subset=["congestion_class"]).copy()
    df["congestion_class"] = df["congestion_class"].astype(int)

    # Apply headroom factors
    df["headroom_factor"] = df["congestion_class"].map(headroom_factors)
    unknown = df[df["headroom_factor"].isna()]["congestion_class"].unique().tolist()
    if unknown:
        raise ValueError(f"Missing headroom factor for congestion classes: {unknown}")

    df["baseline_demand_MW"] = float(baseline_demand_mw)
    df["estimated_capacity_MW"] = df["baseline_demand_MW"] * df["headroom_factor"]
    df["usable_capacity_MW"] = df["estimated_capacity_MW"] * MAX_UTILIZATION
    df["remaining_MW"] = df["usable_capacity_MW"] - df["baseline_demand_MW"]

    assumptions = EvPortAssumptions()
    df["extra_ports"] = df["remaining_MW"].apply(lambda mw: estimate_ports_from_remaining_mw(mw, assumptions))

    # Aggregate at voedingsgebied_id level for a cleaner table
    voedingsgebied = (
        df.groupby(["voedingsgebied_id", "voedingsgebied_naam"], as_index=False)
        .agg(
            postcodes=("postcode", "nunique"),
            max_congestion_class=("congestion_class", "max"),
            mean_congestion_class=("congestion_class", "mean"),
            baseline_demand_MW=("baseline_demand_MW", "first"),
            estimated_capacity_MW=("estimated_capacity_MW", "mean"),
            remaining_MW=("remaining_MW", "mean"),
            extra_ports=("extra_ports", "sum"),
        )
    ).sort_values(["remaining_MW", "postcodes"], ascending=[False, False])

    print("\nEstimated remaining capacity from congestion classes (no zone↔postcode mapping yet)\n")
    print(f"Baseline demand used everywhere: {baseline_demand_mw:.3f} MW")
    print(f"Max utilization cap applied: {MAX_UTILIZATION:.0%}")
    print("Headroom factors (class -> multiplier): " + ", ".join(f"{k}:{v}" for k, v in sorted(headroom_factors.items())))
    print(
        voedingsgebied[
            [
                "voedingsgebied_id",
                "voedingsgebied_naam",
                "postcodes",
                "max_congestion_class",
                "mean_congestion_class",
                "estimated_capacity_MW",
                "usable_capacity_MW",
                "remaining_MW",
                "extra_ports",
            ]
        ].to_string(index=False)
    )

    output_csv = p(r"other data\congestion_based_remaining_capacity.csv")
    voedingsgebied.to_csv(output_csv, index=False)
    print(f"\nWrote: {output_csv}")


def run_mode_assume_all_zones_class0(demand_path: Path) -> None:
    """Estimate remaining capacity per zone assuming congestion class 0 everywhere.

    This is the "no mapping" shortcut:
    - capacity_MW is estimated as peak_demand_MW * class0_factor
    - remaining_MW is the difference between that capacity and the observed peak/average demand
    """

    demand = _read_zonal_demand(demand_path)
    agg = _aggregate_zone_demand(demand)

    class0_factor = 1.30

    out = agg.copy()
    out["assumed_congestion_class"] = 0
    out["headroom_factor"] = class0_factor
    out["estimated_capacity_MW"] = out["peak_demand_MW"] * out["headroom_factor"]
    out["usable_capacity_MW"] = out["estimated_capacity_MW"] * MAX_UTILIZATION
    out["remaining_MW_at_peak"] = out["usable_capacity_MW"] - out["peak_demand_MW"]
    out["remaining_MW_at_avg"] = out["usable_capacity_MW"] - out["avg_demand_MW"]

    assumptions = EvPortAssumptions()
    out["extra_ports_at_peak"] = out["remaining_MW_at_peak"].apply(
        lambda mw: estimate_ports_from_remaining_mw(mw, assumptions)
    )
    out["extra_ports_at_avg"] = out["remaining_MW_at_avg"].apply(
        lambda mw: estimate_ports_from_remaining_mw(mw, assumptions)
    )

    out = out.sort_values("remaining_MW_at_peak", ascending=False)
    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 140)

    print("\nEstimated remaining capacity per zone (assume class=0 everywhere)\n")
    print(f"Class 0 headroom factor used: {class0_factor}")
    print(f"Max utilization cap applied: {MAX_UTILIZATION:.0%}")
    print(
        out[
            [
                "zone_id",
                "peak_demand_MW",
                "avg_demand_MW",
                "estimated_capacity_MW",
                "usable_capacity_MW",
                "remaining_MW_at_peak",
                "extra_ports_at_peak",
                "remaining_MW_at_avg",
                "extra_ports_at_avg",
            ]
        ].to_string(index=False)
    )

    output_csv = p(r"other data\zone_remaining_capacity_estimates_assume_class0.csv")
    out.to_csv(output_csv, index=False)
    print(f"\nWrote: {output_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Remaining grid capacity estimation")
    parser.add_argument(
        "--mode",
    choices=["manual", "congestion", "assume-class0"],
        default="manual",
        help="manual=use other data/zone_grid_capacity.csv; congestion=estimate from congestie_pc6.csv",
    )
    parser.add_argument(
        "--baseline-demand-mw",
        type=float,
        default=None,
        help="Only for --mode congestion. If omitted, uses the overall peak demand across zones.",
    )
    parser.add_argument(
        "--max-utilization",
        type=float,
        default=0.85,
        help="Safety cap: do not plan above this fraction of max capacity (default: 0.85).",
    )
    args = parser.parse_args()

    global MAX_UTILIZATION
    MAX_UTILIZATION = float(getattr(args, "max_utilization", 0.85))
    if not (0.0 < MAX_UTILIZATION <= 1.0):
        raise ValueError("--max-utilization must be in (0, 1]")

    demand_path = p(r"Data_Set\Dataset 6 – Electricity Load (Demand)\eindhoven_zonal_load.csv")

    if not demand_path.exists():
        raise FileNotFoundError(f"Demand file not found: {demand_path}")

    if args.mode == "manual":
        capacity_path = p(r"other data\zone_grid_capacity.csv")
        run_mode_manual_zone_capacity(demand_path=demand_path, capacity_path=capacity_path)
        return

    if args.mode == "assume-class0":
        run_mode_assume_all_zones_class0(demand_path=demand_path)
        return

    # congestion mode
    congestie_path = p(r"Data_Set\Dataset 5 – Grid Congestion & Constraints\congestie_pc6.csv")
    if not congestie_path.exists():
        raise FileNotFoundError(f"Congestion file not found: {congestie_path}")

    demand = _read_zonal_demand(demand_path)
    zone_agg = _aggregate_zone_demand(demand)
    default_baseline = float(zone_agg["peak_demand_MW"].max())
    baseline = float(args.baseline_demand_mw) if args.baseline_demand_mw is not None else default_baseline

    # Heuristic: map congestion class -> capacity multiplier (tunable!)
    headroom_factors = {
        0: 1.30,  # not congested
        1: 1.15,  # barely congested
        2: 1.05,  # moderately congested
        3: 1.00,  # fully congested (no headroom)
    }

    run_mode_estimated_from_congestion(
        congestie_path=congestie_path,
        baseline_demand_mw=baseline,
        headroom_factors=headroom_factors,
    )

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
        finally:
            sys.exit(0)
