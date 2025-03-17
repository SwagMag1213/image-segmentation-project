import random
import numpy as np
import matplotlib.pyplot as plt


class PlateLayout:
    def __init__(self):
        self.rows = "ABCDEFGH"
        self.cols = range(1, 13)
        self.plate_positions = [f"{row}{col}" for row in self.rows for col in self.cols]
        self.layout = {pos: "empty" for pos in self.plate_positions}
        self.seed = 28

        if self.seed is not None:
            random.seed(self.seed)  # Set random seed for reproducibility

        self.position_labels = {
            # 'A1': 'corner', 'A12': 'corner', 'H1': 'corner', 'H12': 'corner',
            'A1': 'outer_well', 'A12': 'outer_well', 'H1': 'outer_well', 'H12': 'outer_well',
            'A2': 'outer_well', 'A3': 'outer_well', 'A4': 'outer_well', 'A5': 'outer_well', 'A6': 'outer_well', 'A7': 'outer_well', 'A8': 'outer_well', 'A9': 'outer_well', 'A10': 'outer_well', 'A11': 'outer_well',
            'B1': 'outer_well', 'C1': 'outer_well', 'D1': 'outer_well', 'E1': 'outer_well', 'F1': 'outer_well', 'G1': 'outer_well',
            'H2': 'outer_well', 'H3': 'outer_well', 'H4': 'outer_well', 'H5': 'outer_well', 'H6': 'outer_well', 'H7': 'outer_well', 'H8': 'outer_well', 'H9': 'outer_well', 'H10': 'outer_well', 'H11': 'outer_well',
            'B12': 'outer_well', 'C12': 'outer_well', 'D12': 'outer_well', 'E12': 'outer_well', 'F12': 'outer_well', 'G12': 'outer_well',
            'B2': '2nd_row', 'B3': '2nd_row', 'B4': '2nd_row', 'B5': '2nd_row', 'B6': '2nd_row', 'B7': '2nd_row', 'B8': '2nd_row', 'B9': '2nd_row', 'B10': '2nd_row', 'B11': '2nd_row',
            'C2': '2nd_row', 'D2': '2nd_row', 'E2': '2nd_row', 'F2': '2nd_row', 'G2': '2nd_row',
            'C11': '2nd_row', 'D11': '2nd_row', 'E11': '2nd_row', 'F11': '2nd_row',
            'G3': '2nd_row', 'G4': '2nd_row', 'G5': '2nd_row', 'G6': '2nd_row', 'G7': '2nd_row', 'G8': '2nd_row', 'G9': '2nd_row', 'G10': '2nd_row', 'G11': '2nd_row',
            'C3': '3rd_row', 'C4': '3rd_row', 'C5': '3rd_row', 'C6': '3rd_row', 'C7': '3rd_row', 'C8': '3rd_row', 'C9': '3rd_row', 'C10': '3rd_row',
            'D3': '3rd_row', 'E3': '3rd_row', 'F3': '3rd_row', 'D10': '3rd_row', 'E10': '3rd_row',
            'F4': '3rd_row', 'F5': '3rd_row', 'F6': '3rd_row', 'F7': '3rd_row', 'F8': '3rd_row', 'F9': '3rd_row', 'F10': '3rd_row',
            'D4': 'center', 'D5': 'center', 'E4': 'center', 'E5': 'center',
            'D6': 'center', 'D7': 'center', 'D8': 'center', 'D9': 'center', 'E6': 'center', 'E7': 'center', 'E8': 'center', 'E9': 'center'
        }
        self.controls = "control"
        self.concentrations = [f"C{i}" for i in range(1, 5)]
        self.treatments = [f"T{i}" for i in range(1, 4)]

    def assign_conditions(self):
        self._assign_to_row("outer_well", 12) # 36 outer wells, 24 treatments, 8 controls 
        self._assign_to_row("2nd_row", 4) # 28 wells, 
        self._assign_to_row("3rd_row", 8) # 20 wells, 
        self._assign_to_center()

    def _assign_to_row(self, row_label, num_controls):
        wells = [pos for pos, label in self.position_labels.items() if label == row_label]
        random.shuffle(wells)

        # Assign controls
        for well in wells[:num_controls]:
            self.layout[well] = self.controls

        # Assign remaining wells
        remaining = wells[num_controls:]
        conditions = [f"{c}_{t}" for c in self.concentrations for t in self.treatments]
        
        # Randomly assign conditions to remaining wells
        for i, well in enumerate(remaining):
            self.layout[well] = conditions[i % len(conditions)]

    def _assign_to_center(self):
        center_wells = [pos for pos, label in self.position_labels.items() if label == "center"]
        conditions = [f"{c}_{t}" for c in self.concentrations for t in self.treatments]
        random.shuffle(conditions)
        for i, well in enumerate(center_wells):
            self.layout[well] = conditions[i % len(conditions)]

    def count_combinations(self):
        combination_count = {}

        # Count combinations in the layout
        for well, label in self.layout.items():
            combination_count[label] = combination_count.get(label, 0) + 1
        print("Combination count:")
        sorted_combination_count = dict(sorted(combination_count.items()))
        return sorted_combination_count

    def plot_layout(self):
        # Convert layout dictionary to 2D numpy array
        plate_layout = np.array(
            [[self.layout[f"{row}{col}"] for col in self.cols] for row in self.rows]
        )

        # Plot the layout
        self._plot_plate_layout(plate_layout)

    def _plot_plate_layout(self, plate_layout):
        rows, cols = plate_layout.shape
        plt.figure(figsize=(12, 6), dpi = 200)

        # Create a color map
        unique_labels = np.unique(plate_layout)
        color_map = {label: plt.cm.tab20(i / len(unique_labels)) for i, label in enumerate(unique_labels)}

        # Plot each well
        for i in range(rows):
            for j in range(cols):
                color = color_map[plate_layout[i, j]]
                plt.gca().add_patch(
                    plt.Rectangle((j, rows - i - 1), 1, 1, color=color, edgecolor='black')
                )
                plt.text(j + 0.5, rows - i - 1 + 0.5, plate_layout[i, j],
                         ha="center", va="center", fontsize=8)

        # Set axis limits to ensure all wells are visible
        plt.xlim(0, cols)
        plt.ylim(0, rows)

        # Add row and column labels
        plt.xticks(range(cols), range(1, cols + 1))
        plt.yticks(range(rows), [chr(65 + i) for i in range(rows)])
        plt.gca().set_aspect("equal")

        # Title and invert y-axis to mimic a top-down view
        plt.title("96-Well Plate Layout")
        plt.gca().invert_yaxis()

        plt.show()

# Instantiate and use the class
plate = PlateLayout()
plate.assign_conditions()
plate.plot_layout()
plate.count_combinations()