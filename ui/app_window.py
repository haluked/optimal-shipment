"""Optimal Shipment - Ana Uygulama Penceresi (App Window).

Özellikler:
- Sol Kontrol Paneli: Senaryo seçimi, parametreler, tıklama modu, eylemler ve canlı metrik kartları.
- Sağ Harita Paneli: Etkileşimli Matplotlib haritası (tıklayarak nokta ekleme).
- Deterministik 2-Opt ve Nearest Neighbor çözümü.
"""

from __future__ import annotations
import tkinter as tk
from tkinter import messagebox, ttk
from typing import List, Optional
import numpy as np

from core.clustering import solve_full_vrp
from core.metrics import compute_fleet_summary
from core.models import Customer, Depot, Solution
from ui.canvas_view import MapCanvasView
from ui.theme import (
    COLOR_BG,
    COLOR_CARD,
    COLOR_BORDER,
    COLOR_PANEL,
    COLOR_PRIMARY,
    COLOR_SUCCESS,
    COLOR_TEXT,
    COLOR_TEXT_MUTED,
    FONT_BODY,
    FONT_BODY_BOLD,
    FONT_METRIC_LBL,
    FONT_METRIC_VAL,
    FONT_SECTION,
    FONT_SUBTITLE,
    FONT_TITLE,
)


class OptimalShipmentApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Optimal Shipment - Filo Rota Optimizasyonu (VRP)")
        self.root.geometry("1240x820")
        self.root.minsize(1000, 700)
        self.root.configure(bg=COLOR_BG)

        # Durum verileri
        self.depots: List[Depot] = []
        self.customers: List[Customer] = []
        self.solution: Optional[Solution] = None
        self.click_mode = tk.StringVar(value="none")  # "none", "customer", "depot"

        self._setup_layout()
        self._load_preset("random_3_30")

    def _setup_layout(self):
        # Ana taşıyıcı
        main_container = tk.Frame(self.root, bg=COLOR_BG)
        main_container.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # 1. SOL PANEL (Kontrol ve Metrikler)
        self.left_panel = tk.Frame(
            main_container,
            bg=COLOR_PANEL,
            width=360,
            padx=16,
            pady=16,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )
        self.left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12))
        self.left_panel.pack_propagate(False)

        # 2. SAĞ PANEL (Harita)
        right_panel = tk.Frame(
            main_container,
            bg=COLOR_CARD,
            highlightbackground=COLOR_BORDER,
            highlightthickness=1,
        )
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.map_view = MapCanvasView(
            parent=right_panel,
            on_map_clicked=self._on_map_clicked,
        )

        self._build_sidebar_controls()
        self._build_metrics_panel()

        # Alt durum çubuğu
        self.status_bar = tk.Label(
            self.root,
            text="Hazır. Parametreleri ayarlayıp 'Rotayı Optimize Et' butonuna basın.",
            bg=COLOR_BG,
            fg=COLOR_TEXT_MUTED,
            font=FONT_BODY,
            anchor="w",
            padx=16,
            pady=4,
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def _build_sidebar_controls(self):
        # Başlık Bölümü
        lbl_title = tk.Label(
            self.left_panel,
            text="Optimal Shipment",
            font=FONT_TITLE,
            fg=COLOR_TEXT,
            bg=COLOR_PANEL,
            anchor="w",
        )
        lbl_title.pack(fill=tk.X)

        lbl_sub = tk.Label(
            self.left_panel,
            text="Araç Rotalama & 2-Opt Optimizasyonu",
            font=FONT_SUBTITLE,
            fg=COLOR_TEXT_MUTED,
            bg=COLOR_PANEL,
            anchor="w",
        )
        lbl_sub.pack(fill=tk.X, pady=(0, 14))

        # --- Hazır Senaryolar ---
        tk.Label(
            self.left_panel,
            text="SENARYO SEÇİMİ",
            font=FONT_SECTION,
            fg=COLOR_TEXT,
            bg=COLOR_PANEL,
            anchor="w",
        ).pack(fill=tk.X, pady=(4, 4))

        self.combo_presets = ttk.Combobox(
            self.left_panel,
            state="readonly",
            values=[
                "Standart Şehir (3 Depo, 30 Müşteri)",
                "Merkez Depo (1 Depo, 25 Müşteri)",
                "Geniş Dağıtım (5 Depo, 60 Müşteri)",
                "Boş Harita (Manuel Çizim)",
            ],
        )
        self.combo_presets.current(0)
        self.combo_presets.pack(fill=tk.X, pady=(0, 12))
        self.combo_presets.bind("<<ComboboxSelected>>", self._on_preset_selected)

        # --- Parametre Girdileri ---
        tk.Label(
            self.left_panel,
            text="FİLO VE NOKTA PARAMETRELERİ",
            font=FONT_SECTION,
            fg=COLOR_TEXT,
            bg=COLOR_PANEL,
            anchor="w",
        ).pack(fill=tk.X, pady=(4, 4))

        param_grid = tk.Frame(self.left_panel, bg=COLOR_PANEL)
        param_grid.pack(fill=tk.X, pady=(0, 10))

        tk.Label(param_grid, text="Depo Sayısı:", font=FONT_BODY, bg=COLOR_PANEL, fg=COLOR_TEXT).grid(row=0, column=0, sticky="w", pady=3)
        self.ent_depots = ttk.Entry(param_grid, width=8)
        self.ent_depots.insert(0, "3")
        self.ent_depots.grid(row=0, column=1, sticky="e", pady=3)

        tk.Label(param_grid, text="Müşteri Sayısı:", font=FONT_BODY, bg=COLOR_PANEL, fg=COLOR_TEXT).grid(row=1, column=0, sticky="w", pady=3)
        self.ent_customers = ttk.Entry(param_grid, width=8)
        self.ent_customers.insert(0, "30")
        self.ent_customers.grid(row=1, column=1, sticky="e", pady=3)

        tk.Label(param_grid, text="Rastgele Tohum (Seed):", font=FONT_BODY, bg=COLOR_PANEL, fg=COLOR_TEXT).grid(row=2, column=0, sticky="w", pady=3)
        self.ent_seed = ttk.Entry(param_grid, width=8)
        self.ent_seed.insert(0, "42")
        self.ent_seed.grid(row=2, column=1, sticky="e", pady=3)

        tk.Label(param_grid, text="Araç Kapasitesi (Ops.):", font=FONT_BODY, bg=COLOR_PANEL, fg=COLOR_TEXT).grid(row=3, column=0, sticky="w", pady=3)
        self.ent_capacity = ttk.Entry(param_grid, width=8)
        self.ent_capacity.insert(0, "0")
        self.ent_capacity.grid(row=3, column=1, sticky="e", pady=3)

        # --- Seçenekler ---
        self.var_2opt = tk.BooleanVar(value=True)
        chk_2opt = tk.Checkbutton(
            self.left_panel,
            text="2-Opt Çapraz Takas İyileştirmesini Uygula",
            variable=self.var_2opt,
            font=FONT_BODY_BOLD,
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            activebackground=COLOR_PANEL,
        )
        chk_2opt.pack(anchor="w", pady=(0, 10))

        # --- Tıklama Modu ---
        tk.Label(
            self.left_panel,
            text="HARİTA TIKLAMA MODU",
            font=FONT_SECTION,
            fg=COLOR_TEXT,
            bg=COLOR_PANEL,
            anchor="w",
        ).pack(fill=tk.X, pady=(4, 4))

        mode_box = tk.Frame(self.left_panel, bg=COLOR_PANEL)
        mode_box.pack(fill=tk.X, pady=(0, 12))

        tk.Radiobutton(
            mode_box,
            text="Gezin",
            variable=self.click_mode,
            value="none",
            bg=COLOR_PANEL,
            fg=COLOR_TEXT,
            font=FONT_BODY,
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Radiobutton(
            mode_box,
            text="+ Müşteri",
            variable=self.click_mode,
            value="customer",
            bg=COLOR_PANEL,
            fg="#C53030",
            font=FONT_BODY_BOLD,
        ).pack(side=tk.LEFT, padx=(0, 10))

        tk.Radiobutton(
            mode_box,
            text="+ Depo",
            variable=self.click_mode,
            value="depot",
            bg=COLOR_PANEL,
            fg="#2B6CB0",
            font=FONT_BODY_BOLD,
        ).pack(side=tk.LEFT)

        # --- Butonlar ---
        btn_frame = tk.Frame(self.left_panel, bg=COLOR_PANEL)
        btn_frame.pack(fill=tk.X, pady=(0, 14))

        self.btn_run = tk.Button(
            btn_frame,
            text="ROTAYI OPTİMİZE ET",
            command=self.optimize_routes,
            bg=COLOR_PRIMARY,
            fg="#FFFFFF",
            font=FONT_BODY_BOLD,
            relief=tk.FLAT,
            cursor="hand2",
            pady=8,
        )
        self.btn_run.pack(fill=tk.X, pady=(0, 6))

        btn_row = tk.Frame(btn_frame, bg=COLOR_PANEL)
        btn_row.pack(fill=tk.X)

        btn_regen = tk.Button(
            btn_row,
            text="Yeni Noktalar Üret",
            command=self._generate_random_points,
            bg="#E2E8F0",
            fg=COLOR_TEXT,
            font=FONT_BODY,
            relief=tk.FLAT,
            cursor="hand2",
            pady=4,
        )
        btn_regen.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0, 4))

        btn_clear = tk.Button(
            btn_row,
            text="Temizle",
            command=self._clear_all,
            bg="#FED7D7",
            fg="#C53030",
            font=FONT_BODY,
            relief=tk.FLAT,
            cursor="hand2",
            pady=4,
        )
        btn_clear.pack(side=tk.RIGHT, expand=True, fill=tk.X)

    def _build_metrics_panel(self):
        """Alt kısımdaki lojistik metrik kartları."""
        tk.Label(
            self.left_panel,
            text="FİLO PERFORMANS METRİKLERİ",
            font=FONT_SECTION,
            fg=COLOR_TEXT,
            bg=COLOR_PANEL,
            anchor="w",
        ).pack(fill=tk.X, pady=(6, 6))

        grid = tk.Frame(self.left_panel, bg=COLOR_PANEL)
        grid.pack(fill=tk.X)

        # Kart 1: Toplam Mesafe
        card1 = self._make_card(grid, "Toplam Mesafe", "-- km")
        card1.grid(row=0, column=0, sticky="nsew", padx=3, pady=3)
        self.lbl_dist_val = card1.nametowidget("val")

        # Kart 2: 2-Opt Tasarrufu
        card2 = self._make_card(grid, "2-Opt Tasarrufu", "-- %")
        card2.grid(row=0, column=1, sticky="nsew", padx=3, pady=3)
        self.lbl_save_val = card2.nametowidget("val")

        # Kart 3: Aktif Rota / Araç
        card3 = self._make_card(grid, "Aktif Filo", "-- araç")
        card3.grid(row=1, column=0, sticky="nsew", padx=3, pady=3)
        self.lbl_fleet_val = card3.nametowidget("val")

        # Kart 4: Ortalama Rota
        card4 = self._make_card(grid, "Ort. Mesafe / Araç", "-- km")
        card4.grid(row=1, column=1, sticky="nsew", padx=3, pady=3)
        self.lbl_avg_val = card4.nametowidget("val")

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)

    def _make_card(self, parent: tk.Widget, label_text: str, default_val: str) -> tk.Frame:
        card = tk.Frame(parent, bg=COLOR_CARD, padx=8, pady=8, highlightbackground=COLOR_BORDER, highlightthickness=1)
        lbl_v = tk.Label(card, name="val", text=default_val, font=FONT_METRIC_VAL, fg=COLOR_TEXT, bg=COLOR_CARD)
        lbl_v.pack(anchor="w")
        lbl_l = tk.Label(card, text=label_text, font=FONT_METRIC_LBL, fg=COLOR_TEXT_MUTED, bg=COLOR_CARD)
        lbl_l.pack(anchor="w")
        return card

    def _on_preset_selected(self, _event):
        idx = self.combo_presets.current()
        if idx == 0:
            self._load_preset("random_3_30")
        elif idx == 1:
            self._load_preset("single_1_25")
        elif idx == 2:
            self._load_preset("large_5_60")
        elif idx == 3:
            self._clear_all()

    def _load_preset(self, preset_key: str):
        if preset_key == "random_3_30":
            self.ent_depots.delete(0, tk.END)
            self.ent_depots.insert(0, "3")
            self.ent_customers.delete(0, tk.END)
            self.ent_customers.insert(0, "30")
            self.ent_seed.delete(0, tk.END)
            self.ent_seed.insert(0, "42")
        elif preset_key == "single_1_25":
            self.ent_depots.delete(0, tk.END)
            self.ent_depots.insert(0, "1")
            self.ent_customers.delete(0, tk.END)
            self.ent_customers.insert(0, "25")
            self.ent_seed.delete(0, tk.END)
            self.ent_seed.insert(0, "100")
        elif preset_key == "large_5_60":
            self.ent_depots.delete(0, tk.END)
            self.ent_depots.insert(0, "5")
            self.ent_customers.delete(0, tk.END)
            self.ent_customers.insert(0, "60")
            self.ent_seed.delete(0, tk.END)
            self.ent_seed.insert(0, "777")

        self._generate_random_points()

    def _generate_random_points(self):
        try:
            n_depots = max(1, int(self.ent_depots.get()))
            n_custs = max(1, int(self.ent_customers.get()))
            seed = int(self.ent_seed.get())
        except ValueError:
            messagebox.showerror("Hata", "Lütfen depo ve müşteri sayılarını tam sayı olarak girin.")
            return

        np.random.seed(seed)
        d_coords = np.random.rand(n_depots, 2) * 100
        c_coords = np.random.rand(n_custs, 2) * 100

        self.depots = [
            Depot(id=i + 1, x=round(float(d_coords[i, 0]), 1), y=round(float(d_coords[i, 1]), 1))
            for i in range(n_depots)
        ]
        self.customers = [
            Customer(id=j + 1, x=round(float(c_coords[j, 0]), 1), y=round(float(c_coords[j, 1]), 1))
            for j in range(n_custs)
        ]
        self.solution = None
        self._update_metrics_view(None)
        self.map_view.render(self.depots, self.customers)
        self.status_bar.config(
            text=f"Yeni senaryo yüklendi: {len(self.depots)} depo, {len(self.customers)} müşteri. 'Rotayı Optimize Et' butonuna basın."
        )

    def _on_map_clicked(self, x: float, y: float):
        mode = self.click_mode.get()
        if mode == "customer":
            new_id = len(self.customers) + 1
            self.customers.append(Customer(id=new_id, x=x, y=y))
            self.ent_customers.delete(0, tk.END)
            self.ent_customers.insert(0, str(len(self.customers)))
            self.status_bar.config(text=f"Müşteri {new_id} eklendi: ({x}, {y})")
            # Çözümü sıfırla veya yeniden çöz
            self.map_view.render(self.depots, self.customers, self.solution)
        elif mode == "depot":
            new_id = len(self.depots) + 1
            self.depots.append(Depot(id=new_id, x=x, y=y))
            self.ent_depots.delete(0, tk.END)
            self.ent_depots.insert(0, str(len(self.depots)))
            self.status_bar.config(text=f"Depo {new_id} eklendi: ({x}, {y})")
            self.map_view.render(self.depots, self.customers, self.solution)

    def _clear_all(self):
        self.depots.clear()
        self.customers.clear()
        self.solution = None
        self.ent_depots.delete(0, tk.END)
        self.ent_depots.insert(0, "0")
        self.ent_customers.delete(0, tk.END)
        self.ent_customers.insert(0, "0")
        self._update_metrics_view(None)
        self.map_view.reset_view()
        self.status_bar.config(text="Harita temizlendi. Tıklama modunu seçerek haritaya nokta ekleyebilirsiniz.")

    def optimize_routes(self):
        if not self.depots:
            messagebox.showwarning("Uyarı", "Haritada en az bir Depo bulunmalıdır!")
            return
        if not self.customers:
            messagebox.showwarning("Uyarı", "Haritada en az bir Müşteri bulunmalıdır!")
            return

        try:
            capacity_val = float(self.ent_capacity.get())
        except ValueError:
            capacity_val = 0.0

        use_2opt = self.var_2opt.get()

        self.status_bar.config(text="Rotalar hesaplanıyor ve 2-Opt ile optimize ediliyor...")
        self.root.update_idletasks()

        # Çekirdek çözücü motorunu çağır
        self.solution = solve_full_vrp(
            depots=self.depots,
            customers=self.customers,
            vehicle_capacity=capacity_val if capacity_val > 0 else None,
            use_2opt=use_2opt,
        )

        self._update_metrics_view(self.solution)
        self.map_view.render(self.depots, self.customers, self.solution)

        mode_str = "2-Opt yerel arama ile" if use_2opt else "Açgözlü En Yakın Komşu ile"
        self.status_bar.config(
            text=f"Optimizasyon tamamlandı ({mode_str}): Toplam {self.solution.optimized_total_distance:.1f} km · %{self.solution.savings_percent:.1f} tasarruf"
        )

    def _update_metrics_view(self, solution: Optional[Solution]):
        if not solution:
            self.lbl_dist_val.config(text="-- km", fg=COLOR_TEXT)
            self.lbl_save_val.config(text="-- %", fg=COLOR_TEXT)
            self.lbl_fleet_val.config(text="-- araç", fg=COLOR_TEXT)
            self.lbl_avg_val.config(text="-- km", fg=COLOR_TEXT)
            return

        summary = compute_fleet_summary(solution)
        self.lbl_dist_val.config(text=f"{summary.optimized_distance:.1f} km", fg=COLOR_TEXT)
        save_color = COLOR_SUCCESS if summary.savings_percent > 0 else COLOR_TEXT
        self.lbl_save_val.config(text=f"%{summary.savings_percent:.1f}", fg=save_color)
        self.lbl_fleet_val.config(text=f"{summary.total_vehicles} araç", fg=COLOR_TEXT)
        self.lbl_avg_val.config(text=f"{summary.avg_route_distance:.1f} km", fg=COLOR_TEXT)
