"""Optimal Shipment - Etkileşimli Harita Görünümü (Canvas View).

Matplotlib FigureCanvasTkAgg entegrasyonu ile:
- Depoları, müşterileri ve optimize edilen araç rotalarını çizer.
- Kullanıcının harita üzerine tıklayarak yeni depo ve müşteri eklemesine izin verir.
- Rota teslimat sırasını numaralar ve yön oklarıyla net şekilde gösterir.
"""

from __future__ import annotations
import tkinter as tk
from typing import Callable, List, Optional
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.models import Customer, Depot, Solution
from ui.theme import (
    COLOR_BG,
    COLOR_CARD,
    COLOR_DEPOT,
    COLOR_CUSTOMER,
    COLOR_GRID,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
)


class MapCanvasView:
    def __init__(
        self,
        parent: tk.Widget,
        on_map_clicked: Optional[Callable[[float, float], None]] = None,
    ):
        self.parent = parent
        self.on_map_clicked = on_map_clicked

        self.fig, self.ax = plt.subplots(figsize=(8, 6), dpi=100)
        self.fig.patch.set_facecolor(COLOR_CARD)
        self.ax.set_facecolor(COLOR_CARD)

        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(fill=tk.BOTH, expand=True)

        # Fare tıklama olayını bağla
        self.canvas.mpl_connect("button_press_event", self._handle_click)

        self.reset_view()

    def _handle_click(self, event):
        """Kullanıcı haritaya tıkladığında tetiklenir."""
        if event.inaxes != self.ax:
            return
        if event.xdata is None or event.ydata is None:
            return

        x = round(float(event.xdata), 1)
        y = round(float(event.ydata), 1)

        if self.on_map_clicked:
            self.on_map_clicked(x, y)

    def reset_view(self):
        """Haritayı boş duruma getirir."""
        self.ax.clear()
        self.ax.set_facecolor(COLOR_CARD)
        self.ax.set_xlim(-5, 105)
        self.ax.set_ylim(-5, 105)
        self.ax.set_title("Lojistik Dağıtım Haritası", fontsize=11, fontweight="bold", color=COLOR_TEXT, pad=12)
        self.ax.set_xlabel("X Koordinatı (km)", fontsize=9, color=COLOR_TEXT_MUTED)
        self.ax.set_ylabel("Y Koordinatı (km)", fontsize=9, color=COLOR_TEXT_MUTED)
        self.ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
        self.ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=8)
        self.fig.tight_layout()
        self.canvas.draw()

    def render(
        self,
        depots: List[Depot],
        customers: List[Customer],
        solution: Optional[Solution] = None,
    ):
        """Depoları, müşterileri ve hesaplanan rotaları görselleştirir."""
        self.ax.clear()
        self.ax.set_facecolor(COLOR_CARD)

        # 1. Rotaları ve yön oklarını çiz
        if solution and solution.routes:
            for r in solution.routes:
                pts = r.points_sequence
                if len(pts) < 2:
                    continue

                xs = [p.x for p in pts]
                ys = [p.y for p in pts]

                # Rota çizgisi
                self.ax.plot(
                    xs,
                    ys,
                    color=r.color,
                    linewidth=1.8,
                    alpha=0.8,
                    linestyle="-",
                    zorder=2,
                    label=f"Araç {r.vehicle_id} ({r.total_distance:.1f} km)",
                )

                # Yön okları ve teslimat numaraları
                for k in range(len(pts) - 1):
                    p1 = pts[k]
                    p2 = pts[k + 1]

                    # Rota yön oku
                    self.ax.annotate(
                        "",
                        xy=(p2.x, p2.y),
                        xytext=(p1.x, p1.y),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color=r.color,
                            lw=1.5,
                            mutation_scale=12,
                            alpha=0.9,
                        ),
                        zorder=3,
                    )

                    # Müşteri durağı ise sıra numarası etiketi koy (son nokta olan depoya dönüş hariç)
                    if k < len(pts) - 2:
                        self.ax.text(
                            p2.x,
                            p2.y + 2.2,
                            str(k + 1),
                            fontsize=8,
                            fontweight="bold",
                            color="#1A202C",
                            ha="center",
                            va="bottom",
                            bbox=dict(boxstyle="circle,pad=0.2", facecolor="#FFF", edgecolor=r.color, alpha=0.9),
                            zorder=6,
                        )

        # 2. Müşteri noktalarını çiz
        if customers:
            c_xs = [c.x for c in customers]
            c_ys = [c.y for c in customers]
            self.ax.scatter(
                c_xs,
                c_ys,
                c=COLOR_CUSTOMER,
                s=45,
                marker="o",
                edgecolors="#FFFFFF",
                linewidths=1.2,
                zorder=4,
                label=f"Müşteriler ({len(customers)})",
            )

        # 3. Depoları çiz (Belirgin Kareler)
        if depots:
            d_xs = [d.x for d in depots]
            d_ys = [d.y for d in depots]
            self.ax.scatter(
                d_xs,
                d_ys,
                c=COLOR_DEPOT,
                s=140,
                marker="s",
                edgecolors="#1A365D",
                linewidths=1.8,
                zorder=5,
                label=f"Depolar ({len(depots)})",
            )
            # Depo etiketleri
            for d in depots:
                self.ax.text(
                    d.x,
                    d.y - 3.8,
                    f"Depo {d.id}",
                    fontsize=8,
                    fontweight="bold",
                    color=COLOR_DEPOT,
                    ha="center",
                    va="top",
                    zorder=7,
                )

        # Dinamik sınır hesaplama (noktalar taşmasın)
        all_x = [d.x for d in depots] + [c.x for c in customers]
        all_y = [d.y for d in depots] + [c.y for c in customers]
        if all_x and all_y:
            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)
            pad_x = max(10, (max_x - min_x) * 0.1)
            pad_y = max(10, (max_y - min_y) * 0.1)
            self.ax.set_xlim(min_x - pad_x, max_x + pad_x)
            self.ax.set_ylim(min_y - pad_y, max_y + pad_y)
        else:
            self.ax.set_xlim(-5, 105)
            self.ax.set_ylim(-5, 105)

        title = "Lojistik Dağıtım ve Rota Optimizasyonu"
        if solution:
            title += f" (Toplam: {solution.optimized_total_distance:.1f} km · Tasarruf: %{solution.savings_percent:.1f})"
        self.ax.set_title(title, fontsize=11, fontweight="bold", color=COLOR_TEXT, pad=12)
        self.ax.set_xlabel("X Koordinatı (km)", fontsize=9, color=COLOR_TEXT_MUTED)
        self.ax.set_ylabel("Y Koordinatı (km)", fontsize=9, color=COLOR_TEXT_MUTED)
        self.ax.grid(True, linestyle="--", alpha=0.5, color=COLOR_GRID)
        self.ax.tick_params(colors=COLOR_TEXT_MUTED, labelsize=8)

        # Rota sayısı makulse lejant göster
        if solution and solution.routes and len(solution.routes) <= 8:
            self.ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

        self.fig.tight_layout()
        self.canvas.draw()
