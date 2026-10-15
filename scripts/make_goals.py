"""Generate the 20 goal YAML files.

Written as a script so the library is reproducible and the dev/held-out mirror
pairs stay in step, rather than 20 hand-edited files drifting apart.
"""

from __future__ import annotations

import pathlib

import yaml

ROOT = pathlib.Path("configs/goals")

PM = [f"pm_interval_M{i}" for i in range(1, 6)]
SPEED = [f"speed_setpoint_M{i}" for i in range(1, 6)]
INVENTORY = ["reorder_point", "order_quantity", "safety_stock"]
LABOUR = ["shift_count", "overtime_hours", "throughput_multiplier"]
ALL_CONTROL = LABOUR + PM + SPEED + INVENTORY

GOALS: list[dict] = [
    # ---------------------------------------------------------------- dev --
    {
        "name": "dev_max_profit",
        "split": "dev",
        "context_id": 0,
        "objective": {"profit": 1.0},
        "notes": "Plain single-objective: the simplest possible goal.",
    },
    {
        "name": "dev_max_profit_service",
        "split": "dev",
        "context_id": 1,
        "objective": {"profit": 1.0},
        "constraints": [
            {"kpi": "on_time_delivery_rate", "op": ">=", "threshold": 0.90},
            {"kpi": "stockout_probability", "op": "<=", "threshold": 0.10},
        ],
        "notes": "Profit under the service constraints from twin.yaml.",
    },
    {
        "name": "dev_weighted_profit_oee",
        "split": "dev",
        "context_id": 2,
        "objective": {"profit": 1.0, "oee": 200000.0},
        "notes": (
            "Weighted multi-objective. The OEE weight converts a point of OEE into "
            "pounds; 200,000 makes a 0.05 OEE gain worth about GBP 10,000, which is "
            "the same order as the profit differences between good plans."
        ),
    },
    {
        "name": "dev_min_unit_cost",
        "split": "dev",
        "context_id": 3,
        "objective": {"unit_cost": 1.0},
        "sense": "minimise",
        "constraints": [{"kpi": "on_time_delivery_rate", "op": ">=", "threshold": 0.85}],
        "notes": "A minimise goal, to exercise the sense flip.",
    },
    {
        "name": "dev_pm_only",
        "split": "dev",
        "context_id": 4,
        "objective": {"profit": 1.0},
        "free_dims": PM,
        "constraints": [{"kpi": "unplanned_failures", "op": "<=", "threshold": 1.0}],
        "notes": (
            "Frozen dimensions: only the five PM intervals move. This is the "
            "maintenance-planning question in isolation, and it is also the only "
            "goal shape where grid search applies on a 5-dim space (it does not: "
            "5 > 4, so grid reports n/a)."
        ),
    },
    {
        "name": "dev_offshore_supplier",
        "split": "dev",
        "context_id": 5,
        "objective": {"profit": 1.0},
        "fixed": {"supplier_choice": 2},
        "constraints": [{"kpi": "stockout_probability", "op": "<=", "threshold": 0.15}],
        "notes": (
            "Exogenous what-if: forced onto the low-cost offshore supplier with a "
            "21-day lead time. The optimiser must re-plan inventory around it."
        ),
    },
    {
        "name": "dev_downtime_cap",
        "split": "dev",
        "context_id": 6,
        "objective": {"units_produced": 1.0},
        "constraints": [{"kpi": "downtime_hours", "op": "<=", "threshold": 12.0}],
        "notes": "Throughput under a downtime cap.",
    },
    {
        "name": "dev_infeasible_service",
        "split": "dev",
        "context_id": 7,
        "objective": {"profit": 1.0},
        "constraints": [
            {"kpi": "on_time_delivery_rate", "op": ">=", "threshold": 0.995},
            {"kpi": "stockout_probability", "op": "<=", "threshold": 0.005},
        ],
        "infeasible_by_design": True,
        "notes": (
            "INFEASIBLE BY DESIGN. The twin's best observed on-time delivery is "
            "about 0.98 and the line cannot hold stockout below 0.5% at any plan, "
            "so this must return status=infeasible_unverified rather than a plan "
            "that looks fine because the check was skipped."
        ),
    },
    # ----------------------------------------------------------- held-out --
    {
        "name": "held_max_profit",
        "split": "held_out",
        "context_id": 20,
        "objective": {"profit": 1.0},
        "notes": "Single-objective, held-out mirror of dev_max_profit.",
    },
    {
        "name": "held_max_profit_service",
        "split": "held_out",
        "context_id": 21,
        "objective": {"profit": 1.0},
        "constraints": [
            {"kpi": "on_time_delivery_rate", "op": ">=", "threshold": 0.90},
            {"kpi": "stockout_probability", "op": "<=", "threshold": 0.10},
        ],
        "notes": "Profit under the twin.yaml service constraints.",
    },
    {
        "name": "held_weighted_units_cost",
        "split": "held_out",
        "context_id": 22,
        "objective": {"units_produced": 20.0, "unit_cost": -500.0},
        "notes": (
            "Weighted trade-off between volume and unit cost, with a negative "
            "weight rather than a minimise sense, to exercise mixed-sign weights."
        ),
    },
    {
        "name": "held_min_unit_cost",
        "split": "held_out",
        "context_id": 24,
        "objective": {"unit_cost": 1.0},
        "sense": "minimise",
        "constraints": [{"kpi": "on_time_delivery_rate", "op": ">=", "threshold": 0.85}],
        "notes": "Minimise goal.",
    },
    {
        "name": "held_pm_only",
        "split": "held_out",
        "context_id": 25,
        "objective": {"profit": 1.0},
        "free_dims": PM,
        "constraints": [{"kpi": "unplanned_failures", "op": "<=", "threshold": 1.0}],
        "notes": "Frozen dimensions: only PM intervals move.",
    },
    {
        "name": "held_speed_only",
        "split": "held_out",
        "context_id": 26,
        "objective": {"profit": 1.0},
        "free_dims": SPEED,
        "constraints": [{"kpi": "unplanned_failures", "op": "<=", "threshold": 2.0}],
        "notes": (
            "Frozen dimensions: only the five speed setpoints. Speed buys "
            "throughput and costs scrap and wear, so this is the sharpest "
            "trade-off in the model."
        ),
    },
    {
        "name": "held_inventory_only",
        "split": "held_out",
        "context_id": 27,
        "objective": {"profit": 1.0},
        "free_dims": INVENTORY,
        "constraints": [{"kpi": "stockout_probability", "op": "<=", "threshold": 0.10}],
        "notes": (
            "Only the three (s, Q) dimensions move -- three free controllables, so "
            "this is one of the few goals where grid search actually applies."
        ),
    },
    {
        "name": "held_offshore_supplier",
        "split": "held_out",
        "context_id": 28,
        "objective": {"profit": 1.0},
        "fixed": {"supplier_choice": 2},
        "constraints": [{"kpi": "stockout_probability", "op": "<=", "threshold": 0.15}],
        "notes": "Exogenous what-if: forced onto the 21-day offshore supplier.",
    },
    {
        "name": "held_price_cut",
        "split": "held_out",
        "context_id": 29,
        "objective": {"profit": 1.0},
        "fixed": {"price_multiplier": 0.90},
        "notes": (
            "Exogenous what-if: a 10% price cut, which raises demand through the "
            "elasticity assumption. A demand surge the plant did not choose."
        ),
    },
    {
        "name": "held_premium_supplier_service",
        "split": "held_out",
        "context_id": 30,
        "objective": {"profit": 1.0},
        "fixed": {"supplier_choice": 1},
        "constraints": [
            {"kpi": "on_time_delivery_rate", "op": ">=", "threshold": 0.93},
            {"kpi": "downtime_hours", "op": "<=", "threshold": 20.0},
        ],
        "notes": (
            "Exogenous what-if on the premium local supplier (3-day lead, higher "
            "price), under a tighter service constraint than the default."
        ),
    },
    {
        "name": "held_failures_cap",
        "split": "held_out",
        "context_id": 31,
        "objective": {"profit": 1.0},
        "constraints": [{"kpi": "unplanned_failures", "op": "<=", "threshold": 0.5}],
        "notes": (
            "Hard constraint on the KPI the surrogate predicts WORST (R2 0.733). "
            "Included deliberately: if the surrogate-first method has a weak spot, "
            "it should show here."
        ),
    },
    {
        "name": "held_infeasible_profit_floor",
        "split": "held_out",
        "context_id": 32,
        "objective": {"profit": 1.0},
        "constraints": [
            {"kpi": "profit", "op": ">=", "threshold": 2000000.0},
            {"kpi": "stockout_probability", "op": "<=", "threshold": 0.02},
        ],
        "infeasible_by_design": True,
        "notes": (
            "INFEASIBLE BY DESIGN. The landscape search over 500 Sobol scenarios "
            "found a best profit of about GBP 493,000, so a GBP 2,000,000 floor is "
            "roughly 4x anything achievable. Must return "
            "status=infeasible_unverified."
        ),
    },
]


def main() -> int:
    """Write every goal file."""
    ROOT.mkdir(parents=True, exist_ok=True)
    for goal in GOALS:
        payload = dict(goal)
        name = payload["name"]
        path = ROOT / f"{name}.yaml"
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, default_flow_style=False),
            encoding="utf-8",
            newline="\n",
        )
    dev = sum(1 for g in GOALS if g["split"] == "dev")
    held = sum(1 for g in GOALS if g["split"] == "held_out")
    infeasible = sum(1 for g in GOALS if g.get("infeasible_by_design"))
    print(
        f"wrote {len(GOALS)} goals: {dev} dev, {held} held-out, {infeasible} infeasible-by-design"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
