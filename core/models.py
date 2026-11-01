"""Optimal Shipment - Veri Modelleri (Models).

Bu modül rota optimizasyonunda kullanılan temel veri yapılarını içerir.
Tüm sınıflar deterministik ve tip güvenli (type-annotated) olarak tasarlanmıştır.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass(frozen=True)
class Point:
    """2D düzlemde bir koordinat noktası."""
    x: float
    y: float

    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Depot:
    """Kamyonların kalkış ve dönüş yaptığı merkez depo."""
    id: int
    x: float
    y: float
    name: str = ""

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


@dataclass(frozen=True)
class Customer:
    """Teslimat yapılacak müşteri noktası."""
    id: int
    x: float
    y: float
    demand: float = 1.0  # Paket adedi veya ağırlık/hacim
    name: str = ""

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


@dataclass
class Route:
    """Bir araca ait teslimat rotası.
    
    Depodan başlar, müşterileri sırayla gezer ve depoya geri döner.
    """
    vehicle_id: int
    depot: Depot
    customers: List[Customer] = field(default_factory=list)
    points_sequence: List[Point] = field(default_factory=list)
    total_distance: float = 0.0
    total_demand: float = 0.0
    color: str = "#2B6CB0"


@dataclass
class Solution:
    """Filo rotalama probleminin genel çözüm çıktısı."""
    routes: List[Route] = field(default_factory=list)
    initial_total_distance: float = 0.0   # 2-Opt öncesi (Nearest Neighbor) mesafe
    optimized_total_distance: float = 0.0 # 2-Opt sonrası mesafe
    savings_percent: float = 0.0          # İyileştirme oranı (%)
    total_customers_served: int = 0
