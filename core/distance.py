"""Optimal Shipment - Mesafe Hesaplama Modülü (Distance).

Mühendislik temeli:
İki boyutlu Kartezyen düzlemde iki nokta (x1, y1) ve (x2, y2) arasındaki Öklid mesafesi:
    d = sqrt((x2 - x1)^2 + (y2 - y1)^2)
Pisagor teoreminden türetilmiştir. Düzlemsel dağıtım problemlerinde standart metriktir.
"""

from __future__ import annotations
import math
from typing import List, Sequence
import numpy as np

from core.models import Point


def euclidean_distance(p1: Point, p2: Point) -> float:
    """İki nokta arasındaki Öklid (düz hat) mesafesini hesaplar."""
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    return math.hypot(dx, dy)


def compute_distance_matrix(points: Sequence[Point]) -> np.ndarray:
    """N adet nokta arasındaki ikili mesafeleri içeren N x N simetrik matris oluşturur.
    
    Matris elemanı M[i, j], i. nokta ile j. nokta arasındaki Öklid mesafesidir.
    M[i, i] = 0'dır.
    """
    n = len(points)
    coords = np.array([[p.x, p.y] for p in points], dtype=np.float64)
    # Broadcasting ile hızlı hesaplama: (N, 1, 2) - (1, N, 2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dist_matrix


def calculate_path_distance(path: Sequence[Point]) -> float:
    """Verilen sıradaki noktalar dizisinin ardışık toplam mesafesini hesaplar.
    
    path = [P0, P1, P2, ..., Pn] ise:
    Toplam = d(P0, P1) + d(P1, P2) + ... + d(Pn-1, Pn)
    """
    if len(path) < 2:
        return 0.0

    total = 0.0
    for i in range(len(path) - 1):
        total += euclidean_distance(path[i], path[i + 1])
    return total
