import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.spatial.distance import cdist

class DeliveryRouterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Shipment Routing System (VRP)")
        self.root.geometry("1200x800")

        # --- LEFT PANEL (INPUTS) ---
        panel = ttk.LabelFrame(root, text="Fleet Settings", padding=15)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)

        # Inputs
        ttk.Label(panel, text="Number of Depots (Trucks):").pack(anchor="w")
        self.ent_depots = ttk.Entry(panel)
        self.ent_depots.insert(0, "3")
        self.ent_depots.pack(fill=tk.X, pady=5)

        ttk.Label(panel, text="Number of Customers:").pack(anchor="w")
        self.ent_customers = ttk.Entry(panel)
        self.ent_customers.insert(0, "30")
        self.ent_customers.pack(fill=tk.X, pady=5)

        ttk.Label(panel, text="Random Seed:").pack(anchor="w")
        self.ent_seed = ttk.Entry(panel)
        self.ent_seed.insert(0, "10")
        self.ent_seed.pack(fill=tk.X, pady=5)

        ttk.Separator(panel, orient='horizontal').pack(fill=tk.X, pady=20)

        self.btn_run = ttk.Button(panel, text="GENERATE ROUTES", command=self.run_routing)
        self.btn_run.pack(fill=tk.X, pady=10)

        # Legend
        lbl_info = ttk.Label(panel, text="\nBlue Square = Warehouse\nRed Dot = Customer\nArrows = Drive Order", justify="left")
        lbl_info.pack(pady=10)

        # --- RIGHT PANEL (MAP) ---
        plot_frame = ttk.Frame(root)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def solve_tsp_nearest_neighbor(self, start_pos, points):
        """
        Simple Greedy Algorithm: Always go to the nearest unvisited neighbor.
        Returns: Ordered list of coordinates including start/end.
        """
        if len(points) == 0:
            return [start_pos]

        path = [start_pos]
        unvisited = list(range(len(points)))
        
        # Current location is the Depot (start_pos)
        curr_loc = start_pos
        
        while unvisited:
            # Find closest point in 'points' that hasn't been visited
            # Calculate dist from curr_loc to all unvisited points
            candidates = points[unvisited]
            dists = cdist([curr_loc], candidates)[0]
            
            nearest_idx_in_candidates = np.argmin(dists)
            real_idx = unvisited[nearest_idx_in_candidates]
            
            # Move there
            next_loc = points[real_idx]
            path.append(next_loc)
            curr_loc = next_loc
            unvisited.pop(nearest_idx_in_candidates) # Remove from list

        # Return to Depot? (Usually yes for delivery trucks)
        path.append(start_pos)
        return np.array(path)

    def run_routing(self):
        try:
            # 1. Get Inputs
            n_depots = int(self.ent_depots.get())
            n_cust = int(self.ent_customers.get())
            seed = int(self.ent_seed.get())
            np.random.seed(seed)

            # 2. Generate Locations
            depot_locs = np.random.rand(n_depots, 2) * 100
            cust_locs = np.random.rand(n_cust, 2) * 100

            # 3. Assign Customers to Nearest Depot (Clustering)
            # Distance matrix: [Customer i] to [Depot j]
            dists = cdist(cust_locs, depot_locs, metric='euclidean')
            # For each customer, find index of closest depot
            assignments = np.argmin(dists, axis=1)

            # 4. Visualization Setup
            self.ax.clear()
            self.ax.set_title("Optimized Delivery Routes")
            
            colors = plt.cm.tab10(np.linspace(0, 1, n_depots)) # Different color for each truck

            # 5. Process Each Depot (Truck)
            for i in range(n_depots):
                # Get customers assigned to this depot
                my_customers = cust_locs[assignments == i]
                depot_pos = depot_locs[i]

                # Plot Depot
                self.ax.scatter(depot_pos[0], depot_pos[1], c=[colors[i]], s=200, marker='s', edgecolors='black', zorder=10, label=f"Warehouse {i+1}")

                if len(my_customers) > 0:
                    # Plot Customers (Small dots)
                    self.ax.scatter(my_customers[:, 0], my_customers[:, 1], c=[colors[i]], s=50, edgecolors='black', zorder=5)

                    # SOLVE ROUTE (TSP)
                    route_path = self.solve_tsp_nearest_neighbor(depot_pos, my_customers)

                    # DRAW ARROWS
                    # We iterate through the path to draw arrows for direction
                    for k in range(len(route_path) - 1):
                        p1 = route_path[k]
                        p2 = route_path[k+1]
                        self.ax.annotate("", xy=p2, xytext=p1,
                                         arrowprops=dict(arrowstyle="->", color=colors[i], lw=1.5, alpha=0.8))
                        
                        # Add order number to customer
                        # p2 is a customer (unless it's the return trip to depot)
                        if k < len(route_path) - 2: # Don't label the return to depot
                             self.ax.text(p2[0], p2[1]+2, str(k+1), fontsize=9, fontweight='bold', color='black', ha='center')

            # self.ax.legend() # Legend might get too big
            self.ax.grid(True, alpha=0.3)
            self.canvas.draw()

        except ValueError:
            messagebox.showerror("Error", "Please enter integers.")

if __name__ == "__main__":
    root = tk.Tk()
    app = DeliveryRouterApp(root)
    root.mainloop()