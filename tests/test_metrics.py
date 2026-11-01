"""Metrik hesaplayıcı birim testleri."""

import pytest
from core.metrics import compute_fleet_summary
from core.models import Customer, Depot, Point, Route, Solution


def test_compute_fleet_summary_empty():
    sol = Solution()
    summary = compute_fleet_summary(sol)
    assert summary.total_vehicles == 0
    assert summary.initial_distance == 0.0
    assert summary.savings_percent == 0.0


def test_compute_fleet_summary_with_routes():
    depot = Depot(id=1, x=0, y=0)
    c1 = Customer(id=1, x=10, y=0, demand=5)
    r1 = Route(
        vehicle_id=1,
        depot=depot,
        customers=[c1],
        points_sequence=[depot.point, c1.point, depot.point],
        total_distance=20.0,
        total_demand=5.0,
    )
    r2 = Route(
        vehicle_id=2,
        depot=depot,
        customers=[c1],
        points_sequence=[depot.point, c1.point, depot.point],
        total_distance=30.0,
        total_demand=10.0,
    )

    sol = Solution(
        routes=[r1, r2],
        initial_total_distance=60.0,
        optimized_total_distance=50.0,
        savings_percent=16.67,
        total_customers_served=2,
    )

    summary = compute_fleet_summary(sol)
    assert summary.total_vehicles == 2
    assert summary.total_customers == 2
    assert summary.distance_saved == 10.0
    assert summary.avg_route_distance == 25.0
    assert summary.max_route_distance == 30.0
    assert summary.total_demand_delivered == 15.0
