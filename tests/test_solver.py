"""Çözücü (Solver) algoritmaları birim testleri."""

import math
import pytest
from core.distance import calculate_path_distance
from core.models import Customer, Depot, Point
from core.solver import build_route, optimize_2opt, solve_tsp_nearest_neighbor


def test_nearest_neighbor_visits_all_customers():
    depot = Depot(id=1, x=0, y=0)
    customers = [
        Customer(id=1, x=10, y=0),
        Customer(id=2, x=20, y=0),
        Customer(id=3, x=5, y=0),
    ]

    ordered = solve_tsp_nearest_neighbor(depot, customers)
    assert len(ordered) == 3
    # 0'a en yakın 5, sonra 10, sonra 20
    assert [c.id for c in ordered] == [3, 1, 2]


def test_2opt_removes_crossing_edges():
    """Çapraz kesişen rotayı 2-Opt algoritmasının düzelttiğini doğrular.
    
    Örnek:
    Depo: (0, 0)
    Noktalar: A(0, 10), B(10, 0), C(10, 10), D(0, 5)
    Eğer sıra A -> C -> B -> D şeklinde çapraz olursa, 2-opt bunu çözer.
    """
    depot = Depot(id=1, x=0, y=0)
    # Kasıtlı olarak çapraz bir müşteri dizilimi
    c1 = Customer(id=1, x=0, y=10)
    c2 = Customer(id=2, x=10, y=0)
    c3 = Customer(id=3, x=10, y=10)
    c4 = Customer(id=4, x=0, y=5)

    crossed_order = [c1, c2, c3, c4]
    crossed_path = [depot.point] + [c.point for c in crossed_order] + [depot.point]
    crossed_dist = calculate_path_distance(crossed_path)

    optimized_order = optimize_2opt(depot, crossed_order)
    opt_path = [depot.point] + [c.point for c in optimized_order] + [depot.point]
    opt_dist = calculate_path_distance(opt_path)

    # 2-Opt sonrası mesafe kesinlikle daha kısa olmalıdır
    assert opt_dist < crossed_dist


def test_2opt_never_worsens_distance():
    """2-Opt mesafesinin Nearest Neighbor'dan asla daha uzun olamayacağını (monotonluk) doğrular."""
    depot = Depot(id=1, x=50, y=50)
    import numpy as np
    np.random.seed(42)
    random_coords = np.random.rand(20, 2) * 100

    customers = [
        Customer(id=i + 1, x=float(random_coords[i, 0]), y=float(random_coords[i, 1]))
        for i in range(20)
    ]

    route, init_dist = build_route(vehicle_id=1, depot=depot, customers=customers, use_2opt=True)

    # Optimize edilmiş mesafe <= ilk mesafe
    assert route.total_distance <= init_dist + 1e-9
    # Tüm 20 müşteri rotada olmalı
    assert len(route.customers) == 20


def test_empty_and_single_customer():
    depot = Depot(id=1, x=0, y=0)
    route_empty, init_empty = build_route(1, depot, [])
    assert route_empty.total_distance == 0.0

    c1 = Customer(id=1, x=3, y=4)
    route_single, init_single = build_route(1, depot, [c1])
    # Gidiş 5, dönüş 5 = 10
    assert math.isclose(route_single.total_distance, 10.0)
