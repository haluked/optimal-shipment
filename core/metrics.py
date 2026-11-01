"""Optimal Shipment - Filo ve Rota Metrikleri (Metrics).

Mühendislik analizleri için lojistik verimlilik göstergeleri:
- Toplam ve ortalama rota mesafesi
- 2-Opt sezgiseli ile elde edilen mesafe tasarrufu (%)
- Filo doluluk ve iş yükü dengesi
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
from core.models import Solution


@dataclass(frozen=True)
class FleetSummary:
    total_vehicles: int
    total_customers: int
    initial_distance: float
    optimized_distance: float
    distance_saved: float
    savings_percent: float
    avg_route_distance: float
    max_route_distance: float
    total_demand_delivered: float


def compute_fleet_summary(solution: Solution) -> FleetSummary:
    """Çözüm çıktısından detaylı filo performans özeti üretir."""
    if not solution.routes:
        return FleetSummary(
            total_vehicles=0,
            total_customers=0,
            initial_distance=0.0,
            optimized_distance=0.0,
            distance_saved=0.0,
            savings_percent=0.0,
            avg_route_distance=0.0,
            max_route_distance=0.0,
            total_demand_delivered=0.0,
        )

    num_vehicles = len(solution.routes)
    total_opt = solution.optimized_total_distance
    total_init = solution.initial_total_distance
    saved = max(0.0, total_init - total_opt)
    savings_pct = solution.savings_percent

    route_lengths = [r.total_distance for r in solution.routes]
    avg_dist = total_opt / num_vehicles if num_vehicles > 0 else 0.0
    max_dist = max(route_lengths) if route_lengths else 0.0
    total_demand = sum(r.total_demand for r in solution.routes)

    return FleetSummary(
        total_vehicles=num_vehicles,
        total_customers=solution.total_customers_served,
        initial_distance=total_init,
        optimized_distance=total_opt,
        distance_saved=round(saved, 2),
        savings_percent=savings_pct,
        avg_route_distance=round(avg_dist, 2),
        max_route_distance=round(max_dist, 2),
        total_demand_delivered=round(total_demand, 2),
    )
