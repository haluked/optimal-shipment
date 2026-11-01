"""Kümeleme ve filo atama birim testleri."""

import pytest
from core.clustering import (
    assign_customers_to_depots,
    partition_by_capacity,
    solve_full_vrp,
)
from core.models import Customer, Depot


def test_assign_customers_to_nearest_depot():
    # Depo 1 batıda (0, 50), Depo 2 doğuda (100, 50)
    d1 = Depot(id=1, x=0, y=50)
    d2 = Depot(id=2, x=100, y=50)

    # Batıdaki müşteriler
    c_west1 = Customer(id=1, x=10, y=50)
    c_west2 = Customer(id=2, x=20, y=50)
    # Doğudaki müşteri
    c_east = Customer(id=3, x=90, y=50)

    assignments = assign_customers_to_depots([d1, d2], [c_west1, c_west2, c_east])

    assert len(assignments[1]) == 2
    assert len(assignments[2]) == 1
    assert c_east in assignments[2]


def test_partition_by_capacity():
    # 4 müşteri, her birinin talebi 30. Kapasite = 50.
    customers = [
        Customer(id=1, x=0, y=0, demand=30),
        Customer(id=2, x=0, y=0, demand=30),
        Customer(id=3, x=0, y=0, demand=30),
        Customer(id=4, x=0, y=0, demand=30),
    ]

    partitions = partition_by_capacity(customers, vehicle_capacity=50)
    # Her araca en fazla 1 müşteri düşer çünkü 30+30=60 > 50 -> 4 grup
    assert len(partitions) == 4

    partitions_large = partition_by_capacity(customers, vehicle_capacity=100)
    # 30+30+30=90 <= 100 -> 3 müşteri 1. araçta, 1 müşteri 2. araçta -> 2 grup
    assert len(partitions_large) == 2


def test_solve_full_vrp_generates_valid_solution():
    d1 = Depot(id=1, x=20, y=20)
    d2 = Depot(id=2, x=80, y=80)

    custs = [
        Customer(id=1, x=15, y=15),
        Customer(id=2, x=25, y=25),
        Customer(id=3, x=75, y=75),
        Customer(id=4, x=85, y=85),
    ]

    sol = solve_full_vrp([d1, d2], custs, use_2opt=True)
    assert len(sol.routes) == 2
    assert sol.total_customers_served == 4
    assert sol.optimized_total_distance > 0.0
    assert sol.optimized_total_distance <= sol.initial_total_distance + 1e-9
