# DigitSTEM cell image analysis

See the [shared OneDrive folder](https://dtudk-my.sharepoint.com/:f:/r/personal/alpav_dtu_dk/Documents/projects/2024/cell_images?csf=1&web=1&e=8iK1h4) for meeting notes, presentations, and other documents related to this project.

## Contents

### **Code Files**
- **`cell_main.py`**: The main Python script for running the cell analysis. 
  
- **`cell_tools.py`**: Contains utility functions and helper methods used by `cell_main.py`. Includes functions for data preprocessing, feature extraction, and visualization.

- **`well_plate_layout_design.py`**: A Python script that generate a design of a well plate layout, including randomization or arrangement of samples for experiments.

- **`stat_test.R`**: An R script for statistical analysis on the experimental data. It includes different tests and visualizations.

### **Other files**
The remaining files can be loaded into the `cell_main.py` to avoid having to reproduce results which can be time-consuming. 
- `cancer_cell_images_df.csv`
- `NK_cell_images_df.csv`
- `area_results_df.csv`
- `iou_results_df.csv`
- `cluster_results.csv` (Could not be uploaded to GitLab since the file exceeds 100 MiB)
- `cancer_cell_mask_intensities.csv`
- `requirements.txt` (List of Python packages required for running `cell_main.py`)
                             
      