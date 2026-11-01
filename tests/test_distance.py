"""Mesafe hesaplama modülü birim testleri."""

import math
import numpy as np
import pytest

from core.distance import (
    euclidean_distance,
    compute_distance_matrix,
    calculate_path_distance,
)
from core.models import Point


def test_euclidean_distance_known_triangles():
    # 3-4-5 dik üçgeni
    p1 = Point(0, 0)
    p2 = Point(3, 4)
    assert math.isclose(euclidean_distance(p1, p2), 5.0)

    # Sıfır mesafe (aynı nokta)
    assert euclidean_distance(p1, p1) == 0.0

    # Negatif koordinatlar
    p3 = Point(-3, -4)
    assert math.isclose(euclidean_distance(p1, p3), 5.0)


def test_compute_distance_matrix():
    points = [Point(0, 0), Point(1, 0), Point(0, 1)]
    dist_mat = compute_distance_matrix(points)

    assert dist_mat.shape == (3, 3)
    # Köşegenler sıfır olmalı (kendi kendine mesafe)
    assert np.allclose(np.diag(dist_mat), 0.0)
    # Matris simetrik olmalı: d(A, B) = d(B, A)
    assert np.allclose(dist_mat, dist_mat.T)
    # (0,0) ile (1,0) arası 1.0
    assert math.isclose(dist_mat[0, 1], 1.0)
    # (1,0) ile (0,1) arası sqrt(2)
    assert math.isclose(dist_mat[1, 2], math.sqrt(2))


def test_calculate_path_distance():
    # Kare rota: (0,0) -> (1,0) -> (1,1) -> (0,1) -> (0,0)
    path = [Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1), Point(0, 0)]
    assert math.isclose(calculate_path_distance(path), 4.0)

    # Boş veya tek noktalı rota
    assert calculate_path_distance([]) == 0.0
    assert calculate_path_distance([Point(5, 5)]) == 0.0
