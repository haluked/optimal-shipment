"""Arayüz (UI) başlatma birim testi."""

import tkinter as tk
import pytest
from ui.app_window import OptimalShipmentApp


def test_app_window_initialization():
    root = tk.Tk()
    root.withdraw()  # Pencereyi ekranda göstermeden arka planda başlat
    try:
        app = OptimalShipmentApp(root)
        assert len(app.depots) == 3
        assert len(app.customers) == 30

        # Rota optimizasyonunu tetikle
        app.optimize_routes()
        assert app.solution is not None
        assert len(app.solution.routes) > 0
        assert app.solution.optimized_total_distance > 0.0
    finally:
        root.destroy()
