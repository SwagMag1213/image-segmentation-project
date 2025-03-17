# %% ------------------ Import packages ------------------
import numpy as np
import pandas as pd
import os
import cv2
import seaborn as sns
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter

#%% ------------------ Import functions from cell_tools.py ------------------
from cell_tools import compute_local_density, compute_density_statistics, plot_column_vs_hour, process_and_plot_thresholds, \
    plot_example_image, plot_time_series_images, find_cell_cluster_centers, find_NKcell_cluster_centers

#%% ------------------ Path to the folder containing the images ------------------
image_folder = 'C:/Users/mjohj/OneDrive - Danmarks Tekniske Universitet/ai - Lasse Ebdrup Pedersens filer/Xcelligence_data'

# List all .tif files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('B.tif')]
n_images = len(image_files)

# Create an empty DataFrame to store the information
data = []

treatment_dct = {
    '1': 'DU145',
    '2': 'DU145',
    '3': 'DU145',
    '4': 'AtezolizumabNG',
    '5': 'AtezolizumabWT',
    '6': 'AtezolizumabFut8',
    '7': 'AtezolizumabNG',
    '8': 'AtezolizumabWT',
    '9': 'AtezolizumabFut8',
    '10': 'AtezolizumabNG',
    '11': 'AtezolizumabWT',
    '12': 'AtezolizumabFut8',
}

concentration_dct = { # Concentration in ng/ml
    'A': 10000,
    'B': 2500,
    'C': 625,
    'D': 156,
    'E': 39.06,
    'F': 9.77,
    'G': 2.44,
    'H': 0.61,
}

plate_position_dct = {
    'A1': 'corner', 'A12': 'corner', 'H1': 'corner', 'H12': 'corner',
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

# Set global plotting parameters
plt.rcParams.update({
    'font.size': 14,          # Default text size
    'axes.titlesize': 16,     # Title font size
    'axes.labelsize': 14,     # X and Y label font size
    'xtick.labelsize': 14,    # X tick label font size
    'ytick.labelsize': 14,    # Y tick label font size
    'legend.fontsize': 14,    # Legend font size
    'figure.titlesize': 18    # Figure title font size
})

#%% ------------------ Load data from CSV files ------------------
# The dataframes contain information about the images of every well:
# {filename, mean_intensity, min_intensity, max_intensity, well_ID, treatment,
# concentration, timestamp, hour, minutes, seconds, plate_position, filename_short}
# See the last section "Legacy Code" for the code that generated these dataframes

df_B = pd.read_csv('NK_cell_images_df.csv') # Fluorescent images
df_W = pd.read_csv('cancer_cell_images_df.csv') # Broadband images

print(f"{df_B.shape[0]} B images and {df_W.shape[0]} W images")

# Filter the dataframes to include only rows where hour <= 99
df_B = df_B[df_B['hour'] <= 99]
df_W = df_W[df_W['hour'] <= 99]

# Sort the DataFrame by hour
df_B = df_B.sort_values(by='hour')
plot_column_vs_hour(df_B, group_by_column='treatment', plot_column='mean_intensity')

# Sort the DataFrame by hour
df_W = df_W.sort_values(by='hour')
plot_column_vs_hour(df_W, group_by_column='treatment', plot_column='mean_intensity')


#%% ------------------ Import ground truths ------------------
labels_folder = "C:/Users/mjohj/OneDrive - Danmarks Tekniske Universitet\Xcelligence_cells/images_for_labeling/segmentation_output/"
data = []

# Load the ground truth labels
# List all .tif files in the folder
label_files = [f for f in os.listdir(labels_folder) if f.endswith('GT.tif')]
n_images = len(label_files)

for image_file in label_files[:n_images]:
    # Extract information from the filename
    parts = image_file.split('_')
    timestamp = parts[1]  
    hour = int(timestamp.split('h')[0])
    minutes = int(timestamp.split('h')[1][:2])
    seconds = int(timestamp.split('h')[1][3:5])
    well_ID = parts[2]    
    treatment = treatment_dct[well_ID[1:]]
    img_type = parts[3][1]

    # Load the image using OpenCV
    image_path = os.path.join(labels_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # # Get image properties
    mean_intensity = image.mean()
    min_intensity = image.min()
    max_intensity = image.max()
    sum_intensity = image.sum()
    
    # Append the information to the DataFrame
    data.append({
        'filename': image_file,
        'mean_intensity': mean_intensity,
        'min_intensity': min_intensity,
        'max_intensity': max_intensity,
        'sum_intensity': sum_intensity,
        'well_ID': well_ID,
        'treatment': treatment,
        'img_type': img_type,
        'concentration': concentration_dct[well_ID[0]] if int(well_ID[1:]) > 3 else 0,
        'timestamp': timestamp,
        'hour': hour,
        'minutes': minutes,
        'seconds': seconds
    })

df_GT = pd.DataFrame(data)

# Update treatment for specific well_IDs
df_GT.loc[df_GT['well_ID'].isin(['A1', 'A2', 'A3']), 'treatment'] = 'media'
df_GT.loc[df_GT['well_ID'].isin(['B1', 'B2', 'B3', 'C1', 'C2', 'C3']), 'treatment'] = 'media_w_NKcells'
df_GT.loc[df_GT['well_ID'].isin(['D1', 'D2', 'D3', 'H1', 'H2', 'H3']), 'treatment'] = 'DU145_w_NKcells'
df_GT.loc[df_GT['well_ID'].isin(['E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'G1', 'G2', 'G3']), 'treatment'] = 'DU145'

#%% ------------------ Test DU145 cell segmentations ------------------
# Compute TP, FP, TN, FN, and IoU
results = []

label_files = [f for f in os.listdir(labels_folder) if f.endswith('GT.tif')]
n_images = len(label_files)

for filename in label_files[:n_images]:
    # load GT image
    image_path = os.path.join(labels_folder, filename)
    parts = filename.split('_')
    timestamp = parts[1]  
    hour = int(timestamp.split('h')[0])
    labels = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # load original image
    original_image_path = os.path.join(image_folder, filename[:-7] + '.tif')
    image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)

    img_type = filename[-8]
    if img_type == 'W':
        contour_image, _, cell_coordinates, _ = find_cell_cluster_centers(image, plot_steps=False)
    else:
        contour_image, _, cell_coordinates, _, _, _, _ = find_NKcell_cluster_centers(image, plot_steps=False)

    # Binarize the ground truth labels and cell coordinates
    gt_binary = (labels > 0).astype(np.uint8)
    pred_binary = (cell_coordinates > 0).astype(np.uint8)

    # Calculate the total number of elements
    total = image.shape[0] * image.shape[1]

    # Compute TP, FP, TN, FN
    TP = round((np.sum((gt_binary == 1) & (pred_binary == 1)) / total) * 100, 2)
    FP = round((np.sum((gt_binary == 0) & (pred_binary == 1)) / total) * 100, 2)
    TN = round((np.sum((gt_binary == 0) & (pred_binary == 0)) / total) * 100, 2)
    FN = round((np.sum((gt_binary == 1) & (pred_binary == 0)) / total) * 100, 2)

    # Compute IoU
    IoU = TP / (TP + FP + FN)

    # Append results to the list
    results.append({
        'filename': filename,
        'hour': hour,
        'img_type': img_type,
        'TP': TP,
        'FP': FP,
        'TN': TN,
        'FN': FN,
        'IoU': IoU
    })

    # Plot the prediction on top of the ground truth
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f'Image {filename.split('_')[2]}')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(labels, cmap='gray')
    plt.imshow(cell_coordinates, alpha=0.5)#, cmap='jet')
    plt.title('Prediction, IoU: {:.2f}'.format(IoU))
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Convert results to a DataFrame
results_df = pd.DataFrame(results)
print(results_df)

#%% Visualize the IoU values for the 'B' and 'W' image types
# Filter the DataFrame for 'B' and 'W' image types
iou_B = results_df[(results_df.img_type == 'B') & (results_df.IoU != 0)]['IoU']
iou_W = results_df[(results_df.img_type == 'W') & (results_df.IoU != 0)]['IoU']

# Create a figure with two subplots sharing the y-axis
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(8, 6), sharey=True, dpi = 200)

# Boxplot for 'B' image type
axes[0].boxplot(iou_B)
axes[0].set_title('IoU for fluorescent images')
axes[0].set_ylabel('IoU')
axes[0].set_xticklabels([''])
axes[0].grid(True, axis='y')

# Boxplot for 'W' image type
axes[1].boxplot(iou_W)
axes[1].set_title('IoU for broadband images')
axes[1].set_ylabel('')
axes[1].set_xticklabels([''])
axes[1].grid(True)

# Adjust layout
plt.tight_layout()

# Show the plot
plt.show()

#%% ------------------ Combine the dataframes ------------------
# Sort the DataFrame by hour
df_B = df_B.sort_values(by='hour')
df_W = df_W.sort_values(by='hour')

df_B['filename_short'] = [filename[:-5] for filename in df_B['filename']]
df_W['filename_short'] = [filename[:-5] for filename in df_W['filename']]

df_combined = pd.merge(df_B[['filename', 'well_ID', 'treatment', 'concentration', 'timestamp', 'hour', 'minutes', 'seconds', 'filename_short']], df_W[['filename', 'filename_short']], on = 'filename_short')
df_combined.drop(['filename_short'], axis = 1, inplace=True)
df_combined.rename(columns={
    'filename_x': 'B_image_filename',
    'filename_y': 'W_image_filename'
}, inplace=True)

df_combined.head()

#%% ------------------ Plot mean intensity ------------------
fig = plt.figure(figsize = (6,6), dpi = 200)

df_B[df_B['hour'] < 28]['mean_intensity'].plot(kind='hist', bins=35, color = 'gray', edgecolor = 'k', alpha = 0.7)

plt.grid(True)
plt.xlim([40, 160])
plt.xlabel('Mean Pixel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Mean Intensities of W images, $t < 28$')
plt.show()

#%% ------------------ Plot example image ------------------
# Plot an example of a fluorescent image
plot_example_image(image_folder, df_B, well_ID='C12', hour=30, concentration=625.0, treatment='DU145_w_NK_cells')

# Plot an example of a broadband image
plot_example_image(image_folder, df_W, well_ID='C12', hour=30, concentration=None)

#%% ------------------ Plot time series of images with histograms for a specific well ------------------
well_ID = 'G10'
W_images = df_W[df_W['well_ID'] == well_ID].sort_values(['hour', 'minutes', 'seconds'])[['filename', 'timestamp']]
B_images = df_B[df_B['well_ID'] == well_ID].sort_values(['hour', 'minutes', 'seconds'])[['filename', 'timestamp']]

# Create a figure with 4x5 subplots
fig, axes = plt.subplots(4, 5, figsize=(20, 16), dpi=200)

# Plot the raw images in the first row and their histograms in the second row
for i, row in enumerate(W_images[25:30].itertuples()): # Plot images between hours 26-30
    W_filename = row.filename
    W_image_path = os.path.join(image_folder, W_filename)
    W_image = cv2.imread(W_image_path, cv2.IMREAD_GRAYSCALE)

    B_filename = row.filename[:-5] + 'B.tif'
    B_image_path = os.path.join(image_folder, B_filename)
    B_image = cv2.imread(B_image_path, cv2.IMREAD_GRAYSCALE)

    # Determine the subplot location for images (first row)
    col_idx = i % 5

    # Plot the raw image
    row_idx_image = 0
    axes[row_idx_image, col_idx].imshow(W_image, cmap='gray')
    axes[row_idx_image, col_idx].axis('off')
    axes[row_idx_image, col_idx].set_title(f'Broadband, {row.timestamp}')

    # Plot the histogram of the image for each RGB channel
    row_idx_hist = 1
    hist = cv2.calcHist([W_image], [0], None, [256], [0, 256])
    axes[row_idx_hist, col_idx].grid(True)
    axes[row_idx_hist, col_idx].plot(hist, color='k')
    axes[row_idx_hist, col_idx].set_xlim([0, 256])
    axes[row_idx_hist, col_idx].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[row_idx_hist, col_idx].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[row_idx_hist, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    # axes[row_idx_image, col_idx].set_xlabel('Pixel intensities')

    # Plot the raw image
    row_idx_image = 2
    axes[row_idx_image, col_idx].imshow(B_image, cmap='gray')
    axes[row_idx_image, col_idx].axis('off')
    axes[row_idx_image, col_idx].set_title(f'Fluorescent, {row.timestamp}')

    # Plot the histogram of the image
    row_idx_hist = 3
    hist = cv2.calcHist([B_image], [0], None, [256], [0, 256])
    axes[row_idx_hist, col_idx].grid(True)
    axes[row_idx_hist, col_idx].plot(hist, color='k')
    axes[row_idx_hist, col_idx].set_xlim([0, 256])
    axes[row_idx_hist, col_idx].yaxis.set_major_locator(MaxNLocator(integer=True))
    axes[row_idx_hist, col_idx].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    axes[row_idx_hist, col_idx].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

plt.tight_layout()
plt.show()

#%% ------------------ Thresholding example ------------------
# Example of processing an image with different thresholds
image = cv2.imread(os.path.join(image_folder, '240305134144P2_01h45m25s_A10_1W.tif'), cv2.IMREAD_GRAYSCALE)
process_and_plot_thresholds(image, alpha=2, beta=0)

#%% ------------------ Example of processing an image ------------------
image = cv2.imread(os.path.join(image_folder, '240305134144P2_01h45m25s_A10_1W.tif'), cv2.IMREAD_GRAYSCALE)

def process_image(image_path, alpha=4, beta=10, small_object_th=0, plot_steps=False):
    """
    Process the image with contrast and brightness adjustment, Gaussian blur, adaptive thresholding,
    Canny edge detection, and dilation. Optionally plot the intermediate steps.

    Parameters:
    - image_path (str): Path to the input image.
    - alpha (float): Contrast control (default is 4).
    - beta (int): Brightness control (default is 10).
    - plot_steps (bool): Whether to plot the intermediate steps (default is False).

    Returns:
    - processed_image (numpy.ndarray): The final processed image with contours drawn.
    """
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # # Apply the transformation
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Apply adaptive contrast and brightness adjustment
    blur = cv2.GaussianBlur(adjusted_image, (3, 3), 0)

    # Apply adaptive thresholding
    # th = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 25, 1)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # th = cv2.cvtColor(th, cv2.COLOR_GRAY2RGB)

    # Apply Canny edge detection
    img_canny = cv2.Canny(th, 20, 75)

    # Apply dilation
    # kernel = np.ones((4, 4))
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=4)

    # Plot intermediate steps if required
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image, cmap='gray')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Original Image')

    axes[0, 1].imshow(adjusted_image, cmap='gray')
    axes[0, 1].axis('off')
    axes[0, 1].set_title('Increased contrast and brightness')

    axes[0, 2].imshow(blur, cmap='gray')
    axes[0, 2].axis('off')
    axes[0, 2].set_title('Gaussian Blur')

    axes[1, 0].imshow(th, cmap='gray')
    axes[1, 0].axis('off')
    axes[1, 0].set_title('Threshold')

    axes[1, 1].imshow(img_canny, cmap='gray')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Canny Edge Detection')

    axes[1, 2].imshow(img_dilate, cmap='gray')
    axes[1, 2].axis('off')
    axes[1, 2].set_title('Dilation')

    plt.tight_layout()
    plt.show()

    # Find contours
    contours, _ = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Loop through contours and remove small noise
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > small_object_th:
            cv2.drawContours(th, cont, -1, (0, 255, 0), 2)

    # Compute average area of objects
    avg_area = 0
    for cont in contours:
        avg_area += cv2.contourArea(cont)

    # Plot the final image with contours if required
    if plot_steps:
        plt.figure(figsize=(5, 5))
        plt.imshow(th, cmap='gray')
        plt.axis('off')
        plt.title('Cancer Cell Image with Contours')
        plt.show()

    return avg_area, th

# Example usage:
processed_image = process_image(os.path.join(image_folder, '240305134144P2_01h45m25s_A4_1W.tif'), alpha = 1.5, beta = 10, small_object_th = 0, plot_steps=True)

#%% ------------------ Plot time series of images for a specific well ------------------
# plot 25 images for a specific well
plot_time_series_images(image_folder, df_W, well_ID='G5')

# start timestep, end timestep, and step size can be specified as well as whether to show contours
plot_time_series_images(image_folder, df_W, well_ID='G5', start_idx=0, end_idx=25, step=1, show_contours = True)

# start timestep, end timestep, and step size can be specified as well as whether to show contours
plot_time_series_images(image_folder, df_B, well_ID='A4', show_contours = True)

#%% ------------------ Cancer cell (DU145) cluster detection in W images ------------------
image = cv2.imread(os.path.join(image_folder, '240305134144P2_25h46m54s_G12_1W.tif'), cv2.IMREAD_GRAYSCALE)

find_cell_cluster_centers(image, plot_steps=True)

#%% ------------------ TAKES A WHILE TO RUN: Investigate how the mean area of DU145 cell clusters changes over time ------------------
### ------------------ IMPORT BELOW INSTEAD ------------------
counter = 0

# Initialize a list to store the results
area_results = []

# Outliers are commented out!!
for well_ID in ['A4', 'A5', 'A7', 'A8', 'A9', # 'A1', 'A2', 'A3', 'A10', 'A11', 'A12',
                'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B10', 'B11', 'B12', # 'B1', 'B2', 'B3',
                'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', # 'C1', 'C2', 'C3',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
                'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', # 'E1', 'E2', 'E3',
                'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', # 'F1', 'F2', 'F3',
                'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', # 'G1', 'G2', 'G3',
                'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10']: # 'H11', 'H12'
    
    df_W_sub28 = df_W[(df_W['hour'] < 28) & (df_W['well_ID'] == well_ID)]

    # Iterate through the images in df_W_sub28 pairwise
    for i in range(len(df_W_sub28) - 1):
        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} images")

        # Get the current and next image filenames
        image_filename = df_W_sub28.iloc[i]['filename']
        image_path = os.path.join(image_folder, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Find cell cluster centers
        _, _, _, contour_areas = find_cell_cluster_centers(image, plot_steps=False)

        # Compute the mean area of the cell clusters
        mean_area = np.mean(contour_areas)

        # Append the results to the list
        area_results.append({
            'image_filename': image_filename,
            'well_ID': well_ID,
            'treatment': df_W_sub28.iloc[i]['treatment'],
            'hour': df_W_sub28.iloc[i]['hour'],
            'mean_area': mean_area
        })

# Convert the results to a DataFrame
area_results_df = pd.DataFrame(area_results)

#%% ------------------ Import and plot DU145 cluster area results ------------------
area_results_df = pd.read_csv('area_results_df.csv')

plot_column_vs_hour(area_results_df, group_by_column='treatment', plot_column='mean_area')

#%% ------------------ TAKES A WHILE TO RUN: Investigate how much DU145 cells move around in timesteps < 28 ------------------
### ------------------ IMPORT BELOW INSTEAD ------------------
counter = 0

# Initialize a list to store the results
iou_results = []

# Outliers are commented out
for well_ID in ['A4', 'A5', 'A7', 'A8', 'A9', # 'A1', 'A2', 'A3', 'A10', 'A11', 'A12',
                'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B10', 'B11', 'B12', # 'B1', 'B2', 'B3',
                'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', # 'C1', 'C2', 'C3',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
                'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', # 'E1', 'E2', 'E3',
                'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', # 'F1', 'F2', 'F3',
                'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', # 'G1', 'G2', 'G3',
                'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10']: # 'H11', 'H12'
    
    df_W_sub28 = df_W[(df_W['hour'] < 28) & (df_W['well_ID'] == well_ID)]

    # Iterate through the images in df_W_sub28 pairwise
    for i in range(len(df_W_sub28) - 1):
        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} images")

        # Get the current and next image filenames
        current_image_filename = df_W_sub28.iloc[i]['filename']
        next_image_filename = df_W_sub28.iloc[i + 1]['filename']
        
        # Construct the full paths to the images
        current_image_path = os.path.join(image_folder, current_image_filename)
        next_image_path = os.path.join(image_folder, next_image_filename)
        
        # Read the images in grayscale
        current_image = cv2.imread(current_image_path, cv2.IMREAD_GRAYSCALE)
        next_image = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Extract the masks
        _, _, current_mask, _ = find_cell_cluster_centers(current_image, plot_steps=False)
        _, _, next_mask, _ = find_cell_cluster_centers(next_image, plot_steps=False)
        
        # Ensure the masks are the same size
        if current_mask.shape != next_mask.shape:
            continue

        # # Plot the B image with the mask and inverse mask
        # plt.figure(figsize=(5, 5))
        # plt.imshow(current_mask, cmap='gray')
        # plt.imshow(next_mask, alpha=0.2, cmap='Reds')
        # plt.title(f'Timesteps {df_W_sub28.iloc[i]['hour']} and {df_W_sub28.iloc[i + 1]['hour']}')
        # plt.axis('off')

        # plt.tight_layout()
        # plt.show()
        
        # Flatten the masks
        current_mask_flat = current_mask.flatten()
        next_mask_flat = next_mask.flatten()
        
        # Compute the IoU
        iou = jaccard_score(current_mask_flat, next_mask_flat, average='binary', pos_label=255)
        
        # Append the results to the list
        iou_results.append({
            'current_image_filename': current_image_filename,
            'next_image_filename': next_image_filename,
            'well_ID': well_ID,
            'treatment': df_W_sub28.iloc[i]['treatment'],
            'hour_current': df_W_sub28.iloc[i]['hour'],
            'iou': iou
        })

# Convert the results to a DataFrame
iou_results_df = pd.DataFrame(iou_results)

# Print the IoU results
print(iou_results_df)

#%% ------------------ Import and plot IoU results ------------------
iou_results_df = pd.read_csv('iou_results_df.csv')

plot_column_vs_hour(iou_results_df, group_by_column='plate_position', plot_column='iou')

#%% ------------------ TAKES A WHILE TO RUN: Cell zone coordinates ------------------
### ------------------ IMPORT BELOW INSTEAD ------------------
cell_zone_coords = []

# Iterate through unique well_IDs in the 'well_ID' column
# Outliers are commented out
for well_ID in ['A4', 'A5', 'A7', 'A8', 'A9', # 'A1', 'A2', 'A3', 'A10', 'A11', 'A12',
                'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B10', 'B11', 'B12', # 'B1', 'B2', 'B3',
                'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', # 'C1', 'C2', 'C3',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
                'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', # 'E1', 'E2', 'E3',
                'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', # 'F1', 'F2', 'F3',
                'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', # 'G1', 'G2', 'G3',
                'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10']: # 'H11', 'H12'

    # Filter the DataFrame to get the W image where 'hour' == 27 for the current well_ID
    df_W_hour_27 = df_combined[(df_combined['hour'] == 27) & (df_combined['well_ID'] == well_ID)]
    
    if df_W_hour_27.empty:
        continue
    
    # There should be only one W image for the given well_ID and timestep
    row_W = df_W_hour_27.iloc[0]
    
    # Construct the full path to the W image
    W_image_path = os.path.join(image_folder, row_W['W_image_filename'])
    
    # Read the W image in grayscale
    W_image = cv2.imread(W_image_path, cv2.IMREAD_GRAYSCALE)
    
    # Run the function to find cell cluster centers in the W image
    _, _, mask, _ = find_cell_cluster_centers(W_image, plot_steps=False)
    
    # Process each B image where 'hour' > 27 for the same well_ID
    df_B_hour_gt_27 = df_combined[(df_combined['hour'] > 27) & (df_combined['well_ID'] == well_ID)][:5]
    
    for index_b, row_b in df_B_hour_gt_27.iterrows():
        # Construct the full path to the B image
        B_image_path = os.path.join(image_folder, row_b['B_image_filename'])
        
        # Read the B image in grayscale
        B_image = cv2.imread(B_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Ensure the mask is the same size as the B image
        if mask.shape != B_image.shape:
            continue
        
        # Compute the mean intensity of pixels in the mask
        mean_intensity_mask = cv2.mean(B_image, mask=mask)[0] 
        
        # Compute the inverse mask
        inverse_mask = cv2.bitwise_not(mask)
        
        # Compute the mean intensity of pixels in the inverse mask
        mean_intensity_inverse_mask = cv2.mean(B_image, mask=inverse_mask)[0] 
        
        # Append the results to the list
        cell_zone_coords.append({
            'well_ID': well_ID,
            'W_image_filename': row_W['W_image_filename'],
            'B_image_filename': row_b['B_image_filename'],
            'treatment': row_b['treatment'],
            'hour': row_b['hour'],
            'mean_intensity_mask': mean_intensity_mask,
            'mean_intensity_inverse_mask': mean_intensity_inverse_mask
        })
        
        # Plot the B image with the mask and inverse mask
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 4, 1)
        plt.imshow(W_image, cmap='gray')
        plt.imshow(mask, alpha = 0.2)
        plt.title(f'W Image, {row_W['well_ID']}, t: {row_W['hour']}')
        plt.axis('off')
        
        plt.subplot(1, 4, 2)
        plt.imshow(B_image, cmap='gray')
        plt.imshow(mask, alpha = 0.2)#, cmap='Blues')
        plt.title(f'B Image, {row_W['well_ID']}, t: {row_b['hour']}')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

# Convert the results to a DataFrame
# mask_intensities = pd.DataFrame(cell_zone_coords)
# mask_intensities.to_csv('cancer_cell_mask_intensities.csv', index = False)

#%% ------------------ Import and plot mask and inverse mask intensities ------------------
mask_intensities_df = pd.read_csv('cancer_cell_mask_intensities.csv')

# add well plate row as an attribute
# mask_intensities_df['plate_row'] = [int(id[1]) for id in mask_intensities_df['well_ID']]

plot_column_vs_hour(mask_intensities_df, group_by_column='treatment', plot_column='mean_intensity_mask')

#%% ------------------ Plot boxplots of mean_intensity_mask and mean_intensity_inverse_mask ------------------
treatments = mask_intensities_df['treatment'].unique()

plt.figure(figsize=(14, 6), dpi=200)

for i, treatment in enumerate(treatments):
    treatment_data = mask_intensities_df[(mask_intensities_df['treatment'] == treatment) 
                                         & (mask_intensities_df['hour'] == 29)
                                         & (~mask_intensities_df['plate_row'].isin([1,2,3,10,11,12]))]
    plt.subplot(1, 4, i + 1)
    sns.boxplot(data=treatment_data[['mean_intensity_mask', 'mean_intensity_inverse_mask']])
    plt.title(f'{treatment}')
    plt.xticks([0, 1], ['', ''])
    plt.grid(True)
    plt.ylim(75, 95)
    if i == 0:
        plt.ylabel('Mean Intensity')
    else:
        plt.ylabel('')
    plt.xlabel('')

plt.legend(['DU145 cell region', 'Non-cell region'], frameon=False, loc='upper left', bbox_to_anchor=(1, 1))

plt.tight_layout()
plt.show()

#%% ------------------ Investigate relative change in pixel intensity (mask and inverse mask) ------------------
df_B_intensity_change = df_B[(df_B['well_ID'].isin(['A4', 'A5', 'A7', 'A8', 'A9', # 'A1', 'A2', 'A3', 'A10', 'A11', 'A12',
                'B4', 'B5', 'B6', 'B7', 'B8', 'B9','B10', 'B11', 'B12', # 'B1', 'B2', 'B3',
                'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', # 'C1', 'C2', 'C3',
                'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 
                'E4', 'E5', 'E6', 'E7', 'E8', 'E9', 'E10', 'E11', 'E12', # 'E1', 'E2', 'E3',
                'F4', 'F5', 'F6', 'F7', 'F8', 'F9', 'F10', 'F11', 'F12', # 'F1', 'F2', 'F3',
                'G4', 'G5', 'G6', 'G7', 'G8', 'G9', 'G10', 'G11', 'G12', # 'G1', 'G2', 'G3',
                'H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10'])) & (df_B['hour'] > 27)][['well_ID', 'hour', 'mean_intensity']]

# normalize raw fluorescence intensity values
df_B_hour_28_intensity = df_B_intensity_change[df_B_intensity_change['hour'] == 28].set_index('well_ID')['mean_intensity'].to_dict()
df_B_intensity_change['normalized_intensity'] = df_B_intensity_change.apply(lambda row: row['mean_intensity'] / df_B_hour_28_intensity[row['well_ID']], axis=1)

# Calculate the change in B image pixel intensity compared to hour = 28 for each well
df_B_intensity28_dict = df_B_intensity_change[df_B_intensity_change['hour'] == 28].set_index('well_ID')['normalized_intensity'].to_dict()
df_B_intensity_change['intensity_change'] = df_B_intensity_change['well_ID'].map(df_B_intensity28_dict)
df_B_intensity_change['intensity_change'] = df_B_intensity_change['normalized_intensity'] - df_B_intensity_change['intensity_change']

# normalize mask intensities
hour_28_intensity = mask_intensities_df[mask_intensities_df['hour'] == 28].set_index('well_ID')['mean_intensity_mask'].to_dict()
mask_intensities_df['normalized_intensity_mask'] = mask_intensities_df.apply(lambda row: row['mean_intensity_mask'] / hour_28_intensity[row['well_ID']], axis=1)
hour_28_intensity = mask_intensities_df[mask_intensities_df['hour'] == 28].set_index('well_ID')['mean_intensity_inverse_mask'].to_dict()
mask_intensities_df['normalized_intensity_inverse_mask'] = mask_intensities_df.apply(lambda row: row['mean_intensity_inverse_mask'] / hour_28_intensity[row['well_ID']], axis=1)

# Adjust the mean intensity values
df_B_intensity_change = df_B_intensity_change.sort_values(by=['well_ID', 'hour']).reset_index(drop = True)
mask_intensities_df = mask_intensities_df.sort_values(by=['well_ID', 'hour']).reset_index(drop = True)
mask_intensities_df['mean_intensity_mask_adjusted'] = mask_intensities_df['normalized_intensity_mask'] - df_B_intensity_change['intensity_change']
mask_intensities_df['mean_intensity_inverse_mask_adjusted'] = mask_intensities_df['normalized_intensity_inverse_mask'] - df_B_intensity_change['intensity_change']

# Define subset of mask intensities DataFrame
# mask_intensities_df_subset = mask_intensities_df[~mask_intensities_df['plate_row'].isin([1,2,3,10,11,12])]

# Plot the adjusted mean intensity values
plot_column_vs_hour(mask_intensities_df, group_by_column='treatment', plot_column='mean_intensity_mask_adjusted')
plot_column_vs_hour(mask_intensities_df, group_by_column='treatment', plot_column='mean_intensity_inverse_mask_adjusted')

#%% ------------------ NK cell cluster detection in B images ------------------
image = cv2.imread(os.path.join(image_folder, '240305134144P2_31h50m34s_C12_1B.tif'), cv2.IMREAD_GRAYSCALE)

find_NKcell_cluster_centers(image, min_contour_area=1500, plot_steps=True)

#%% ------------------ TAKES A WHILE TO RUN: Investigate how the mean area of NK cell clusters changes over time ------------------
### ------------------ IMPORT BELOW INSTEAD ------------------
counter = 0

# Initialize a list to store the results
area_results = [] # mean cluster area per image
cluster_results_all = [] # mean and summed cluster intensity per cluster in image

for well_ID in df_B['well_ID'].unique():
    
    df_B_after28 = df_B[(df_B['hour'] > 27) & (df_B['hour'] < 51) & (df_B['well_ID'] == well_ID)]
    
    # Iterate through the images in df_B_after28 pairwise
    for i in range(len(df_B_after28) - 1):
        counter += 1
        if counter % 100 == 0:
            print(f"Processed {counter} images")

        # Get the current and next image filenames
        image_filename = df_B_after28.iloc[i]['filename']
        image_path = os.path.join(image_folder, image_filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # print(f'image: {image_filename}, hour: {df_B_after28.iloc[i]["hour"]}')

        # Find cell cluster centers
        _, _, _, contour_areas, mean_cluster_intensities, summed_cluster_intensities, cluster_areas = find_NKcell_cluster_centers(image,  min_contour_area = 0, plot_steps=False)

        # Append the individual cluster results
        for cluster_area, mean_intensity, summed_intensity in zip(cluster_areas, mean_cluster_intensities, summed_cluster_intensities):
            cluster_results_all.append({
                'image_filename': image_filename,
                'cluster_areas': cluster_area,
                'mean_intensity': mean_intensity,
                'summed_intensity': summed_intensity,
                'treatment': df_B_after28.iloc[i]['treatment'],
                'well_ID': well_ID,
                'hour': df_B_after28.iloc[i]['hour']
            })

        # Compute the mean area of the cell clusters
        mean_area = np.mean(contour_areas)
        max_area = np.max(contour_areas)

        # Append the results to the list
        area_results.append({
            'image_filename': image_filename,
            'well_ID': well_ID,
            'treatment': df_B_after28.iloc[i]['treatment'],
            'hour': df_B_after28.iloc[i]['hour'],
            'mean_area': mean_area,
            'max_area': max_area
        })

# Convert the results to a DataFrame
area_results_df = pd.DataFrame(area_results)
cluster_results_df = pd.DataFrame(cluster_results_all)

#%% ------------------ Investigate NK cluster sizes ------------------
# Plot NK cluster sizes in bins and divide by total number of NK cell groups
cluster_results_df = pd.read_csv('cluster_results.csv')

# cluster_results_subset: clusters detected in NK cell images - filter images that contain NK cells
cluster_results_all_df = cluster_results_df[cluster_results_df['treatment'].isin(['AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'DU145_w_NKcells'])]

min_cluster_area = 0
max_cluster_area = 100 #max(cluster_results_all_df['cluster_areas'])

# cluster_results_subset_df: clusters detected in NK cell images with area > min_cluster_area and area < max_cluster_area
cluster_results_subset_df = cluster_results_all_df[(cluster_results_all_df['cluster_areas'] > min_cluster_area) & (cluster_results_all_df['cluster_areas'] < max_cluster_area)]

# large_clusters_df: number of large clusters detected in NK cell images grouped by treatment, well_ID (Well ID), and hour
all_clusters_df = cluster_results_all_df.groupby(['treatment', 'well_ID', 'hour']).count()['image_filename'].reset_index()

# repeat and filter by size
# cluster_results_size_filtered_df = cluster_results_subset_df[(cluster_results_subset_df['cluster_areas'] > min_cluster_area) & (cluster_results_df['cluster_areas'] < max_cluster_area)]

large_clusters_df = cluster_results_subset_df.groupby(['treatment', 'well_ID', 'hour']).count()['image_filename'].reset_index()
large_clusters_df['relative_cluster_counts'] = large_clusters_df['image_filename'] / all_clusters_df['image_filename']

# Rename the columns
large_clusters_df.rename(columns = {'image_filename': 'clusters_count'}, inplace = True)

# large_clusters_df = filter_outliers(large_clusters_df, 'relative_cluster_counts')

print(f'{len(cluster_results_subset_df) / len(cluster_results_all_df)*100:.3f}% of clusters are larger than {min_cluster_area} and smaller than {max_cluster_area}')

# Plot the average number of large clusters per well grouped by treatment 
plot_column_vs_hour(large_clusters_df, 'treatment', 'relative_cluster_counts', include_legend=True)




# %% ------------------ LEGACY CODE ------------------

# ------------------ Load and analyze B images ------------------
# Initialize a counter
counter = 0

for image_file in image_files[:n_images]:
    # Extract information from the filename
    parts = image_file.split('_')
    timestamp = parts[1]  
    hour = int(timestamp.split('h')[0])
    minutes = int(timestamp.split('h')[1][:2])
    seconds = int(timestamp.split('h')[1][3:5])
    well_ID = parts[2]  
    treatment = treatment_dct[well_ID[1:]]

    # Load the image using OpenCV
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  
    assert image is not None, "File could not be read, check with os.path.exists()"
    
    # Get image properties
    height, width = image.shape
    mean_intensity = image.mean()
    min_intensity = image.min()
    max_intensity = image.max()

    # Calculate the threshold for global thresholding
    threshold = np.percentile(image, 50)  # Lower IQR (25th percentile)

    # Gaussian filtering
    blur = cv2.GaussianBlur(image,(5,5),0) # 5x5 kernel, sigma = 0

    # Apply the global threshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
    
    # Calculate the number of white and black pixels
    white_pixels = np.sum(thresholded_image == 255)
    black_pixels = np.sum(thresholded_image == 0)
    
    # Calculate the ratio of white to black pixels
    if black_pixels > 0:
        ratio = white_pixels / black_pixels
    else:
        ratio = np.inf  # Handle the case where there are no black pixels

    # Compute the local density of white pixels and descriptive statistics for the density map
    density_map = compute_local_density(thresholded_image, window_size=16, step_size=16)
    density_stats1 = compute_density_statistics(density_map, window_size=16, step_size=16)
        
    density_map = compute_local_density(thresholded_image, window_size=32, step_size=32)
    density_stats2 = compute_density_statistics(density_map, window_size=32, step_size=32)

    density_map = compute_local_density(thresholded_image, window_size=64, step_size=64)
    density_stats3 = compute_density_statistics(density_map, window_size=64, step_size=64)
    
    # Append the information to the DataFrame
    data.append({
        'filename': image_file,
        'height': height,
        'width': width,
        'mean_intensity': mean_intensity,
        'min_intensity': min_intensity,
        'max_intensity': max_intensity,
        'well_ID': well_ID,
        'treatment': treatment,
        'concentration': concentration_dct[well_ID[0]] if int(well_ID[1:]) > 3 else 0,
        'timestamp': timestamp,
        'hour': hour,
        'minutes': minutes,
        'seconds': seconds,
        'white_pixels': white_pixels,
        'black_pixels': black_pixels,
        'w/b_pixel_ratio': ratio,
        **density_stats1,
        **density_stats2,
        **density_stats3
    })

    # Increment the counter
    counter += 1
    
    # Print status every 100 images
    if counter % 100 == 0:
        print(f"Processed {counter} images")

df = pd.DataFrame(data)

# Update treatment for specific well_IDs
df.loc[df['well_ID'].isin(['A1', 'A2', 'A3']), 'treatment'] = 'media'
df.loc[df['well_ID'].isin(['B1', 'B2', 'B3', 'C1', 'C2', 'C3']), 'treatment'] = 'media_w_NKcells'
df.loc[df['well_ID'].isin(['D1', 'D2', 'D3', 'H1', 'H2', 'H3']), 'treatment'] = 'DU145_w_NKcells'
df.loc[df['well_ID'].isin(['E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'G1', 'G2', 'G3']), 'treatment'] = 'DU145'
# Save the results to a CSV file
# df.to_csv('NK_cell_images_df.csv', index=False)

# ------------------ Load and analyze W images ------------------
# List all .tif files in the folder
image_files = [f for f in os.listdir(image_folder) if f.endswith('W.tif')]
n_images = len(image_files)

# Initialize a counter
counter = 0

for image_file in image_files[:n_images]:
    # Extract information from the filename
    parts = image_file.split('_')
    timestamp = parts[1]  
    hour = int(timestamp.split('h')[0])
    minutes = int(timestamp.split('h')[1][:2])
    seconds = int(timestamp.split('h')[1][3:5])
    well_ID = parts[2]    
    treatment = treatment_dct[well_ID[1:]]

    # Load the image using OpenCV
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
    
    # # Get image properties
    mean_intensity = image.mean()
    min_intensity = image.min()
    max_intensity = image.max()
    sum_intensity = image.sum()
    
    # Append the information to the DataFrame
    data.append({
        'filename': image_file,
        'mean_intensity': mean_intensity,
        'min_intensity': min_intensity,
        'max_intensity': max_intensity,
        'sum_intensity': sum_intensity,
        'well_ID': well_ID,
        'treatment': treatment,
        'concentration': concentration_dct[well_ID[0]] if int(well_ID[1:]) > 3 else 0,
        'timestamp': timestamp,
        'hour': hour,
        'minutes': minutes,
        'seconds': seconds
    })

    # Increment the counter
    counter += 1
    
    # Print status every 100 images
    if counter % 100 == 0:
        print(f"Processed {counter} images")

df_W = pd.DataFrame(data)

# Update treatment for specific well_IDs
df_W.loc[df_W['well_ID'].isin(['A1', 'A2', 'A3']), 'treatment'] = 'media'
df_W.loc[df_W['well_ID'].isin(['B1', 'B2', 'B3', 'C1', 'C2', 'C3']), 'treatment'] = 'media_w_NKcells'
df_W.loc[df_W['well_ID'].isin(['D1', 'D2', 'D3', 'H1', 'H2', 'H3']), 'treatment'] = 'DU145_w_NKcells'
df_W.loc[df_W['well_ID'].isin(['E1', 'E2', 'E3', 'F1', 'F2', 'F3', 'G1', 'G2', 'G3']), 'treatment'] = 'DU145'

# Save the results to a CSV file
# df_W.to_csv('cancer_cell_images_df.csv', index=False)

# ------------------ Cell detection in W images ------------------
# Load the grayscale image
image = cv2.imread(os.path.join(image_folder, '240305134144P2_25h46m54s_A4_1W.tif'), cv2.IMREAD_GRAYSCALE)

def find_cell_centers(image, plot_steps=False):
    """
    Process the image and optionally plot all steps.

    Parameters:
    - image (numpy.ndarray): The input grayscale image.
    - plot_steps (bool): Whether to plot all steps (default is False).

    Returns:
    - contour_image (numpy.ndarray): The final image with contours and centers marked.
    - centers (list): List of centers of the detected contours.
    """
    # Step 1: Display the original grayscale image
    if plot_steps:
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Grayscale Image")
        plt.axis('off')

    # Step 2: Apply Gaussian Blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    if plot_steps:
        plt.subplot(2, 3, 2)
        plt.imshow(blur, cmap='gray')
        plt.title("Gaussian Blurred Image")
        plt.axis('off')

    # Step 3: Adaptive Thresholding
    th7 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 35, 4)
    if plot_steps:
        plt.subplot(2, 3, 3)
        plt.imshow(th7, cmap='gray')
        plt.title("Adaptive Thresholding")
        plt.axis('off')

    # Step 4: Initial Contour Detection and Removal of Small Contours
    cnts = cv2.findContours(th7, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 12000:
            cv2.drawContours(th7, [c], -1, (0, 0, 0), -1)
    if plot_steps:
        plt.subplot(2, 3, 4)
        plt.imshow(th7, cmap='gray')
        plt.title("Removed Small Contours")
        plt.axis('off')

    # Step 5: Morphological Closing and Image Inversion
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    close = 255 - cv2.morphologyEx(th7, cv2.MORPH_CLOSE, kernel)
    if plot_steps:
        plt.subplot(2, 3, 5)
        plt.imshow(close, cmap='gray')
        plt.title("Morphological Closing & Inversion")
        plt.axis('off')

    # Step 6: Final Contour Detection and Display on Original Image
    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    if plot_steps:
        plt.subplot(2, 3, 6)
        plt.imshow(contour_image)
        plt.title("Final Contours on Original Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Find and mark contour centers
    centers = []
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            # Draw a red cross at each center
            cv2.drawMarker(contour_image, (cX, cY), (255, 0, 0), markerType=cv2.MARKER_CROSS, 
                           markerSize=10, thickness=2)

    # Plot the final result with centers marked
    if plot_steps:
        plt.figure(figsize=(10, 10))
        plt.imshow(contour_image)
        plt.title("Final Contours with Cell Centers")
        plt.axis('off')
        plt.show()

    return contour_image, centers

find_cell_centers(image, plot_steps=True)

# ------------------ Crop and plot image for a specific well and timestamp ------------------
example_image = df_B[(df_B['well_ID'] == 'H4') & (df_B['hour'] == 28)][['filename', 'timestamp']]

# Create a figure with 5x5 subplots
fig, axes = plt.subplots(5, 5, figsize=(20, 20), dpi = 200)

# Load the example image
image_path = os.path.join(image_folder, example_image.iloc[0]['filename'])
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Define the size of the subset
subset_size = 64

# Loop through the 5x5 grid and plot subsets of the image
for i in range(5):
    for j in range(5):
        # Calculate the starting and ending indices for the subset
        start_row = i * subset_size
        end_row = start_row + subset_size
        start_col = j * subset_size
        end_col = start_col + subset_size
        
        # Extract the subset
        subset = image[start_row:end_row, start_col:end_col]
        
        # Plot the subset
        axes[i, j].imshow(subset, cmap='gray')
        axes[i, j].axis('off')
        axes[i, j].set_title(f'Subset ({start_row}, {start_col})')

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# ------------------ Interactive cell counting and intensity statistics ------------------
def plot_and_count_cells(image, subset_size=64):
    """
    Plot subsections of the image and prompt the user to input the number of cells in each subsection.
    Save the intensity descriptive statistics along with the cell count.

    Parameters:
    - image (numpy.ndarray): The input grayscale image.
    - subset_size (int): The size of each subsection (default is 64).

    Returns:
    - results (list): A list of dictionaries containing the cell count and intensity statistics for each subsection.
    """
    # results = []

    # Loop through the image in steps of subset_size
    for i in range(0, image.shape[0], subset_size):
        for j in range(0, image.shape[1], subset_size):
            # Extract the subset
            subset = image[i:i + subset_size, j:j + subset_size]

            # Plot the subset
            plt.figure(figsize=(4, 4))
            plt.imshow(subset,vmin=50, vmax=200, cmap='gray')
            plt.title(f'Subset ({i}, {j})')
            plt.axis('off')
            plt.show()

            # Prompt the user to input the number of cells
            cell_count = input(f"Enter the number of cells in subset ({i}, {j}): ")

            # Calculate intensity descriptive statistics
            mean_intensity = subset.mean()
            min_intensity = subset.min()
            max_intensity = subset.max()
            std_intensity = subset.std()
            sum_intensity = subset.sum()

            # Save the results
            results.append({
                'subset_coords': (i, j),
                'cell_count': int(cell_count),
                'mean_intensity': mean_intensity,
                'min_intensity': min_intensity,
                'max_intensity': max_intensity,
                'std_intensity': std_intensity,
                'sum_intensity': sum_intensity
            })

    return results

# Example usage
example_image = df_B[(df_B['well_ID'] == 'A4') & (df_B['hour'] == 29)][['filename', 'timestamp']].iloc[0]
image_path = os.path.join(image_folder, example_image['filename'])
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

results = plot_and_count_cells(image, 99)

# Convert the results to a DataFrame and save to CSV
results_df = pd.DataFrame(results)
results_df.to_csv('cell_count_intensity_stats.csv', index=False)

# ------------------ Dunno ------------------

color_mapping = {
    'DU145_alone': 'maroon',
    'DU145_w_NKcells': 'gray',
    'AtezolizumabNG': 'green',
    'AtezolizumabWT': 'blue',
    'AtezolizumabFut8': 'purple'
}

cluster_results_subset_df = cluster_results_df[cluster_results_df['treatment'].isin(['AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8', 'DU145_w_NKcells']) \
                                               & (cluster_results_df['cluster_areas'] < 2000)]

# Function to calculate weights for each treatment group
def calculate_weights(df, group_column):
    group_counts = df[group_column].value_counts()
    weights = df[group_column].map(lambda x: 1 / group_counts[x])
    return weights

# Calculate weights for each treatment group
weights = calculate_weights(cluster_results_subset_df, 'treatment')

# Function to filter outliers based on IQRla
def filter_outliers(df, column):
    Q1 = df[column].quantile(0.05)
    Q3 = df[column].quantile(0.95)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Get the unique hours
unique_hours = cluster_results_subset_df['hour'].unique()[0:40:3]

# Determine the number of rows and columns for the subplots
n_hours = len(unique_hours)
n_cols = 4  # Number of columns for the subplots
n_rows = (n_hours + n_cols - 1) // n_cols  # Calculate the number of rows needed

# Create a figure with subplots
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5), dpi=200)

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Iterate through each unique hour and plot the KDEs
for i, hour in enumerate(unique_hours):
    # Filter the DataFrame for the current hour
    df_hour = cluster_results_subset_df[cluster_results_subset_df['hour'] == hour]
    
    # Filter out outliers
    # df_hour_filtered = filter_outliers(df_hour, 'cluster_areas')

    hue_order = ['DU145_w_NKcells', 'AtezolizumabNG', 'AtezolizumabWT', 'AtezolizumabFut8']
    
    # Plot KDE of the summed_intensity column, grouped by treatment
    if i == 0:
        # sns.kdeplot(data=df_hour_filtered, x='cluster_areas', hue='treatment', ax=axes[i], fill=True, alpha=0.5, common_norm=False)
        sns.histplot(data=df_hour, bins = 100, x='cluster_areas', hue='treatment', weights=weights, ax=axes[i], fill=True, alpha=0.5, hue_order=hue_order, stat='proportion')#, palette=color_mapping)
        # sns.boxplot(data=df_hour_filtered, y='cluster_areas', x='treatment', hue='treatment', ax=axes[i], showfliers=False)
    else:
        # sns.kdeplot(data=df_hour_filtered, x='cluster_areas', hue='treatment', ax=axes[i], fill=True, common_norm=False, alpha=0.5, legend=False)
        sns.histplot(data=df_hour, bins = 100, x='cluster_areas', hue='treatment', weights=weights, ax=axes[i], fill=True, alpha=0.5, legend=False, hue_order=hue_order, stat='proportion')#, palette=color_mapping)
        # sns.boxplot(data=df_hour_filtered, y='cluster_areas', x='treatment', hue='treatment', ax=axes[i], showfliers=False, legend=False)

    # Set the title and labels
    axes[i].set_title(f'Hour {hour}')
    axes[i].set_xlabel('Cluster Areas')
    axes[i].set_ylabel('')
    axes[i].grid(True)
    axes[i].set_xlim([0, 800])
    axes[i].set_ylim([0, 0.07])

# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# Adjust layout
plt.tight_layout()
plt.show()