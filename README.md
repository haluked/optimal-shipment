# Optimal Shipment (v2.0)

> Çok Depolu Araç Rotalama (Multi-Depot VRP) ve 2-Opt Yerel Arama ile Kurye Rota Optimizasyon Aracı.

[![Python](https://img.shields.io/badge/Python-3.11+-4B8BBE?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-8CBE9C?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-13_Passed-success?style=flat-square)](tests/)
[![Platform](https://img.shields.io/badge/Platform-Windows_|_Linux_|_macOS-93BCD2?style=flat-square)](https://github.com/haluked/optimal-shipment)

---

## Genel Bakış

**Optimal Shipment**, lojistik ve kargo dağıtımında karşılaşılan **Araç Rotalama Problemini (Vehicle Routing Problem - VRP)** ve **Gezgin Satıcı Problemini (Traveling Salesperson Problem - TSP)** 2 boyutlu Öklid düzleminde çözen modern ve etkileşimli bir mühendislik simülasyonudur.

Başlangıçta temel bir açgözlü algoritma prototipi olarak geliştirilen proje; v2.0 ile modüler mimari, **2-Opt yerel arama sezgiseli (local search)**, kapasite kısıtları (CVRP), canlı filo metrikleri ve tıklanabilir etkileşimli harita arayüzü ile yeniden tasarlanmıştır.

```
+-------------------------------------------------------------------------+
| [ OPTIMAL SHIPMENT ]                     [ Lojistik Haritası ]          |
| Filo Rota Optimizasyonu (VRP & TSP)                                     |
+------------------------------------+------------------------------------+
| SENARYOLAR                         |   D1 [Depo]                        |
| [ Standart Şehir (3 Depo, 30 M.) ] |     \                              |
|                                    |      --> C1 (1)                    |
| FİLO PARAMETRELERİ                 |            \                       |
| Depo Sayısı:      [ 3  ]           |             --> C2 (2)             |
| Müşteri Sayısı:   [ 30 ]           |                   \                |
| Rastgele Tohum:   [ 42 ]           |                    --> D1 [Dönüş]  |
|                                    |                                    |
| [x] 2-Opt İyileştirmesini Uygula   |  * Düzlemsel yön okları ve sıra    |
|                                    |    numaraları ile görsel rota      |
| [ ROTAYI OPTİMİZE ET ]             |                                    |
+------------------------------------+------------------------------------+
| FİLO METRİKLERİ                    |                                    |
| [ 342.5 km ]   [ %18.4 ]           | [ 3 Araç ]     [ 114.2 km/araç ]   |
| Toplam Mesafe  2-Opt Tasarrufu     | Aktif Filo     Ortalama Rota       |
+------------------------------------+------------------------------------+
```

---

## Matematiksel ve Algoritmik Temeller

### 1. Düzlemsel Öklid Mesafesi
İki teslimat noktası $P_1(x_1, y_1)$ ve $P_2(x_2, y_2)$ arasındaki kuş uçuşu mesafe Pisagor teoremine dayanır:

$$d(P_1, P_2) = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

### 2. Aşama 1: Açgözlü En Yakın Komşu (Greedy Nearest Neighbor)
* Depo $D$'den başlar.
* Henüz ziyaret edilmemiş müşteriler kümesi $U$ içinden mevcut konuma en yakın olan seçilir:
  $$C^* = \arg\min_{C \in U} d(\text{mevcut}, C)$$
* $O(N^2)$ zaman karmaşıklığında hızlı bir ilk tur oluşturur. Ancak açgözlü mantık son adımlarda uzun geri dönüşlere ve birbirini kesen yollara (çapraz çizgiler) neden olur.

### 3. Aşama 2: 2-Opt Yerel Arama Sezgisi (Crossing Elimination)
* Rotada kesişen iki kenar $(v_i, v_{i+1})$ ve $(v_j, v_{j+1})$ seçilir.
* Kenarların çapraz takas farkı hesaplanır:
  $$\Delta d = \left[ d(v_i, v_j) + d(v_{i+1}, v_{j+1}) \right] - \left[ d(v_i, v_{i+1}) + d(v_j, v_{j+1}) \right]$$
* Eğer $\Delta d < 0$ ise, bu iki kenarın bağlanması toplam mesafeyi kısaltır. Aradaki alt rota ($v_{i+1} \dots v_j$) ters çevrilir.
* Üçgen eşitsizliği gereği düzlemde kesişmeyen yollar daima daha kısa olduğundan, 2-Opt rotaları **%15 ile %25 arasında deterministik olarak optimize eder**.

---

## Öne Çıkan Yetenekler

- **Çok Depolu Kümeleme (Multi-Depot Voronoi):** Müşterileri coğrafi olarak en yakın depoya atar, filo yükünü dengeler.
- **Kapasiteli Filo Desteği (CVRP):** İsteğe bağlı kamyon yük kapasitesi tanımlandığında rotaları araç sınırlarına göre alt turlara böler.
- **Etkileşimli Harita:** Harita üzerinde serbest gezinme veya farenin sol tuşuyla doğrudan haritaya tıklayarak yeni **Müşteri** ya da **Depo** ekleyebilme.
- **Canlı Metrik Paneli:** Toplam mesafe, 2-Opt ile kurtarılan km ve yüzde tasarruf, aktif araç sayısı ve ortalama rota uzunluğu.
- **Hazır Senaryolar:** Tek tıkla küçük şehir, merkez depo veya geniş filo dağıtımı yükleme.

---

## Proje Yapısı

```
optimal-shipment/
├── core/                       # Saf matematiksel ve mantıksal motor
│   ├── models.py               # Point, Customer, Depot, Route, Solution modelleri
│   ├── distance.py             # Öklid mesafesi ve matris fonksiyonları
│   ├── clustering.py           # En yakın depo kümelemesi ve filo dağıtımı
│   ├── solver.py               # Nearest Neighbor & 2-Opt yerel arama çözücüsü
│   └── metrics.py              # Lojistik tasarruf ve verimlilik metrikleri
├── ui/                         # Kullanıcı arayüzü ve görselleştirme
│   ├── theme.py                # Pastel renk paleti ve tipografi
│   ├── canvas_view.py          # Etkileşimli Matplotlib haritası
│   └── app_window.py           # Kontrol paneli, metrik kartları ve olay yöneticisi
├── tests/                      # Birim testleri (pytest)
│   ├── test_distance.py        # Geometrik mesafe testleri
│   ├── test_clustering.py      # Kümeleme ve kapasite testleri
│   ├── test_solver.py          # 2-Opt kesişim giderme ve monotonluk testleri
│   ├── test_metrics.py         # Metrik hesaplama testleri
│   └── test_ui.py              # Arayüz başlatma ve entegrasyon testi
├── main.py                     # Uygulama başlangıç noktası
├── requirements.txt            # Sabit Python bağımlılıkları
├── LICENSE                     # MIT Lisansı
└── README.md                   # Proje dokümantasyonu
```

---

## Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükleme
```bash
pip install -r requirements.txt
```

### 2. Testleri Çalıştırma
```bash
python -m pytest tests/
```
*13 adet birim testi geometri, 2-opt iyileştirme garantisi ve arayüz başlatmasını doğrular.*

### 3. Uygulamayı Başlatma
```bash
python main.py
```

---

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile korunmaktadır.
