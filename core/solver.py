"""Optimal Shipment - Çözücü Motoru (Solver).

İki aşamalı optimizasyon mimarisi:
1. Aşama (İlk Çözüm): Açgözlü En Yakın Komşu (Greedy Nearest Neighbor - NN)
   - O(N^2) karmaşıklığında hızlı bir başlangıç turu üretir.
2. Aşama (Yerel Arama / İyileştirme): 2-Opt Sezgisel Takası (2-Opt Local Search)
   - Çapraz kesişen yolları tespit eder ve alt rotayı ters çevirerek (reverse sub-route)
     mesafeyi kısaltır. Üçgen eşitsizliği gereği Öklid düzleminde kesişmeyen yollar
     daima kesişenlerden daha kısadır.
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from core.distance import euclidean_distance, calculate_path_distance
from core.models import Customer, Depot, Point, Route, Solution


def solve_tsp_nearest_neighbor(depot: Depot, customers: List[Customer]) -> List[Customer]:
    """Açgözlü (Greedy) En Yakın Komşu algoritması ile başlangıç sırası oluşturur.
    
    Depodan başlar, ziyaret edilmemiş en yakın müşteriyi seçerek ilerler.
    Dönüş: Müşterilerin ziyaret sırası listesi.
    """
    if not customers:
        return []

    ordered: List[Customer] = []
    unvisited = list(customers)
    current_point = depot.point

    while unvisited:
        # En yakın ziyaret edilmemiş müşteriyi bul
        nearest_idx = 0
        min_dist = euclidean_distance(current_point, unvisited[0].point)

        for i in range(1, len(unvisited)):
            dist = euclidean_distance(current_point, unvisited[i].point)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i

        chosen = unvisited.pop(nearest_idx)
        ordered.append(chosen)
        current_point = chosen.point

    return ordered


def optimize_2opt(depot: Depot, ordered_customers: List[Customer], max_iterations: int = 500) -> List[Customer]:
    """2-Opt yerel arama takası ile rotadaki kesişmeleri ve dolambaçları giderir.
    
    Formül:
        Δd = [d(v_i, v_j) + d(v_{i+1}, v_{j+1})] - [d(v_i, v_{i+1}) + d(v_j, v_{j+1})]
        Eğer Δd < 0 ise, aradaki alt rota ters çevrilir.
        
    Depo başlangıç ve bitiş noktası olarak sabit kalır.
    """
    n = len(ordered_customers)
    if n < 4:
        # 3 veya daha az müşteride 2-opt kesişme oluşturamaz
        return list(ordered_customers)

    # Tur noktaları: [Depot, C_0, C_1, ..., C_{n-1}, Depot]
    # İndeksler: 0 = Depot, 1..n = Müşteriler, n+1 = Depot
    tour_points: List[Point] = [depot.point] + [c.point for c in ordered_customers] + [depot.point]
    customer_list = list(ordered_customers)

    improved = True
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # i ve j kenarlarını tara:
        # Kenar 1: tour_points[i] -> tour_points[i+1]
        # Kenar 2: tour_points[j] -> tour_points[j+1]
        # i, 0'dan n-1'e kadar (Depot dahil)
        # j, i+2'den n'ye kadar
        for i in range(n):
            for j in range(i + 2, n + 1):
                p_i = tour_points[i]
                p_i1 = tour_points[i + 1]
                p_j = tour_points[j]
                p_j1 = tour_points[j + 1]

                current_edges_len = euclidean_distance(p_i, p_i1) + euclidean_distance(p_j, p_j1)
                new_edges_len = euclidean_distance(p_i, p_j) + euclidean_distance(p_i1, p_j1)

                delta = new_edges_len - current_edges_len

                # Numerik hassasiyet için küçük eşik (1e-9)
                if delta < -1e-9:
                    # i+1 ile j arasındaki müşterileri ve noktaları ters çevir
                    tour_points[i + 1 : j + 1] = reversed(tour_points[i + 1 : j + 1])
                    # customer_list içinde müşteri indisleri: tour index - 1
                    customer_list[i : j] = reversed(customer_list[i : j])
                    improved = True
                    break
            if improved:
                break

    return customer_list


def build_route(
    vehicle_id: int,
    depot: Depot,
    customers: List[Customer],
    use_2opt: bool = True,
    color: str = "#2B6CB0",
) -> Tuple[Route, float]:
    """Verilen müşteriler için önce Nearest Neighbor, ardından 2-Opt ile rota oluşturur.
    
    Dönüş: (optimize_edilmiş_rota, ilk_nearest_neighbor_mesafesi)
    """
    if not customers:
        empty_path = [depot.point, depot.point]
        return Route(vehicle_id=vehicle_id, depot=depot, customers=[], points_sequence=empty_path, color=color), 0.0

    # 1. Aşama: Nearest Neighbor
    nn_customers = solve_tsp_nearest_neighbor(depot, customers)
    initial_path = [depot.point] + [c.point for c in nn_customers] + [depot.point]
    initial_distance = calculate_path_distance(initial_path)

    # 2. Aşama: 2-Opt İyileştirme
    if use_2opt:
        opt_customers = optimize_2opt(depot, nn_customers)
    else:
        opt_customers = nn_customers

    final_path = [depot.point] + [c.point for c in opt_customers] + [depot.point]
    final_distance = calculate_path_distance(final_path)
    total_demand = sum(c.demand for c in opt_customers)

    route = Route(
        vehicle_id=vehicle_id,
        depot=depot,
        customers=opt_customers,
        points_sequence=final_path,
        total_distance=final_distance,
        total_demand=total_demand,
        color=color,
    )
    return route, initial_distance
