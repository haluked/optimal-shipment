"""Optimal Shipment - Kümeleme ve Filo Dağıtımı (Clustering).

Bu modül:
- Müşterileri en yakın depoya atar (Voronoi / Minimum Mesafe Bölümlemesi).
- Kapasite sınırı (CVRP) varsa müşterileri araç filosu arasında dengeli olarak paylaştırır.
- Tüm filonun rotalama sürecini koordine eder.
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from core.distance import euclidean_distance
from core.models import Customer, Depot, Route, Solution
from core.solver import build_route

# Belirgin ve pastel filo renk paleti
FLEET_COLORS = [
    "#3182CE",  # Mavi (Araç 1)
    "#38A169",  # Yeşil (Araç 2)
    "#D69E2E",  # Kehribar (Araç 3)
    "#E53E3E",  # Kırmızı (Araç 4)
    "#805AD5",  # Mor (Araç 5)
    "#DD6B20",  # Turuncu (Araç 6)
    "#319795",  # Camgöbeği (Araç 7)
    "#D53F8C",  # Pembe (Araç 8)
]


def assign_customers_to_depots(
    depots: List[Depot],
    customers: List[Customer],
) -> Dict[int, List[Customer]]:
    """Her müşteriyi Öklid mesafesine göre en yakın depoya atar.
    
    Dönüş: {depot_id: [müşteri_1, müşteri_2, ...]}
    """
    assignments: Dict[int, List[Customer]] = {d.id: [] for d in depots}
    if not depots or not customers:
        return assignments

    for cust in customers:
        closest_depot = min(depots, key=lambda d: euclidean_distance(d.point, cust.point))
        assignments[closest_depot.id].append(cust)

    return assignments


def partition_by_capacity(
    customers: List[Customer],
    vehicle_capacity: float,
) -> List[List[Customer]]:
    """Müşterileri araç kapasitesini aşmayacak şekilde alt gruplara ayırır.
    
    Her bir alt grup bir kamyonun kapasite sınırını temsil eder.
    """
    if vehicle_capacity <= 0 or not customers:
        return [customers] if customers else []

    partitions: List[List[Customer]] = []
    current_group: List[Customer] = []
    current_load = 0.0

    for c in customers:
        if current_load + c.demand > vehicle_capacity and current_group:
            partitions.append(current_group)
            current_group = [c]
            current_load = c.demand
        else:
            current_group.append(c)
            current_load += c.demand

    if current_group:
        partitions.append(current_group)

    return partitions


def solve_full_vrp(
    depots: List[Depot],
    customers: List[Customer],
    vehicle_capacity: Optional[float] = None,
    use_2opt: bool = True,
) -> Solution:
    """Tüm filo için uçtan uca rota optimizasyonu gerçekleştirir.
    
    Adımlar:
    1. Müşterileri en yakın depoya ata.
    2. Gerekirse depo müşterilerini araç kapasitesine göre alt turlara böl.
    3. Her araç için Nearest Neighbor + 2-Opt uygula.
    4. Toplam metrikleri ve tasarruf oranını hesapla.
    """
    if not depots or not customers:
        return Solution()

    assignments = assign_customers_to_depots(depots, customers)
    routes: List[Route] = []
    total_initial_dist = 0.0
    total_opt_dist = 0.0
    vehicle_counter = 1

    for depot in depots:
        depot_customers = assignments.get(depot.id, [])
        if not depot_customers:
            continue

        if vehicle_capacity and vehicle_capacity > 0:
            subgroups = partition_by_capacity(depot_customers, vehicle_capacity)
        else:
            subgroups = [depot_customers]

        for group in subgroups:
            color = FLEET_COLORS[(vehicle_counter - 1) % len(FLEET_COLORS)]
            route, init_dist = build_route(
                vehicle_id=vehicle_counter,
                depot=depot,
                customers=group,
                use_2opt=use_2opt,
                color=color,
            )
            routes.append(route)
            total_initial_dist += init_dist
            total_opt_dist += route.total_distance
            vehicle_counter += 1

    savings = 0.0
    if total_initial_dist > 0:
        savings = ((total_initial_dist - total_opt_dist) / total_initial_dist) * 100.0

    return Solution(
        routes=routes,
        initial_total_distance=round(total_initial_dist, 2),
        optimized_total_distance=round(total_opt_dist, 2),
        savings_percent=round(max(0.0, savings), 2),
        total_customers_served=len(customers),
    )
