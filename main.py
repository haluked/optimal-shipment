"""Optimal Shipment - Ana Başlangıç Noktası (main.py).

Gelişmiş Araç Rotalama Problemi (VRP) & Gezgin Satıcı (TSP) Çözücüsü.
"""

from __future__ import annotations
import sys
import tkinter as tk
from ui.app_window import OptimalShipmentApp


def main():
    root = tk.Tk()
    app = OptimalShipmentApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
