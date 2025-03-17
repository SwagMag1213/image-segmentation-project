import numpy as np
import pandas as pd
import os
import random
import cv2
import seaborn as sns
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt

def plot_example_image(image_folder, df, well_ID, hour, treatment=None, concentration=None):
    """
    Plot an example image based on specified or randomly chosen parameters.

    Parameters:
    - image_folder (str): Path to the folder containing the images.
    - df (pd.DataFrame): df_W for broadband or df_B for fluorescent images.
    - well_ID (str): Well ID.
    - hour (int): Time point in hours.
    - treatment (str, optional): Treatment. Randomly chosen if not provided.
    - concentration (float, optional): Treatment concentration. Randomly chosen if not provided.
    """
        
    # Filter the DataFrame based on the provided or randomly chosen parameters
    filtered_df = df[(df['well_ID'] == well_ID) & (df['hour'] == hour)]
    
    if treatment is None:
        treatment = random.choice(filtered_df['treatment'].unique())
        filtered_df = filtered_df[(filtered_df['treatment'] == treatment)]
    if concentration is None:
        concentration = random.choice(filtered_df['concentration'].unique())
        filtered_df = filtered_df[(filtered_df['treatment'] == treatment)]

    if filtered_df.empty:
        print("No image found with the specified parameters.")
        return
    
    # Get the filename of the first matching image
    image_filename = filtered_df.iloc[0]['filename']
    image_path = os.path.join(image_folder, image_filename)
    
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    if image is None:
        print("Image could not be loaded.")
        return
    
    # Plot the image
    plt.figure(figsize=(6, 6), dpi=200)
    plt.imshow(image, cmap='gray')
    if hour > 27:
        plt.title(f'{well_ID} ({treatment}), {hour}h, {concentration} ng/ml')
    else:
        plt.title(f'{well_ID}, {hour}h')
        
    plt.show()


def process_and_plot_thresholds(image, alpha=2, beta=0, th_percentile = 50):
    """
    Process the image with various thresholding techniques and plot the results.
    # Inspired by https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html
    
    Parameters:
    - image (numpy.ndarray): The input grayscale image.
    - alpha (float): Contrast control (default is 2).
    - beta (int): Brightness control (default is 0).
    """
    # Apply the transformation
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    threshold = np.percentile(image, th_percentile)
    
    # Gaussian filtering
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    adjusted_image_blur = cv2.GaussianBlur(adjusted_image, (3, 3), 0)
    threshold_adjusted = np.percentile(adjusted_image, th_percentile)

    # Global thresholding
    ret, th1 = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

    # Adaptive mean thresholding
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Adaptive Gaussian thresholding
    th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Otsu's thresholding
    _, th4 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Global thresholding after Gaussian filtering
    _, th5 = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY)

    # Adaptive mean thresholding after Gaussian filtering
    th6 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Adaptive Gaussian thresholding after Gaussian filtering
    th7 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Otsu's thresholding after Gaussian filtering
    _, th8 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Increased contrast and brightness
    # Global thresholding
    _, th9 = cv2.threshold(adjusted_image_blur, threshold_adjusted, 255, cv2.THRESH_BINARY)

    # Adaptive mean thresholding after Gaussian filtering
    th10 = cv2.adaptiveThreshold(adjusted_image_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Adaptive Gaussian thresholding after Gaussian filtering
    th11 = cv2.adaptiveThreshold(adjusted_image_blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Otsu's thresholding after Gaussian filtering
    _, th12 = cv2.threshold(adjusted_image_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = [
        'Original Image', f'Global Thresholding (v = {round(threshold, 1)})', 'Adaptive Mean', 'Adaptive Gaussian', 'Otsu Thresholding',
        'Gaussian filtered (GF)', f'GF Global Threshold', 'GF Adaptive Mean', 'GF Adaptive Gaussian', 'GF Otsu Threshold',
        'Increased contrast (IC)', f'IC Global Thresholding (v = {round(threshold_adjusted, 1)})', 'IC Adaptive Mean', 'IC Adaptive Gaussian', 'IC Otsu Thresholding'
    ]
    images = [image, th1, th2, th3, th4, blur, th5, th6, th7, th8, adjusted_image_blur, th9, th10, th11, th12]

    plt.figure(figsize=(20, 12), dpi=200)
    for i in range(15):
        plt.subplot(3, 5, i + 1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def plot_column_vs_hour(df, group_by_column, plot_column, hour_column='hour', include_legend=True):
    """
    Plot the specified column vs hours for each group in the specified column with error bars.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - group_by_column (str): The column name to group by (e.g., 'treatment').
    - plot_column (str): The column name to plot (e.g., 'mean_intensity').
    - hour_column (str): The column name for the hour values (default is 'hour').
    - color_mapping (dict): A dictionary mapping group names to colors (default is None).
    """
    color_mapping = { # Only applies if group_by_column is 'treatment'
    'DU145': 'maroon',
    'DU145_w_NKcells': 'gray',
    'AtezolizumabNG': 'green',
    'AtezolizumabWT': 'blue',
    'AtezolizumabFut8': 'purple',
    '2nd_row': '#1f77b4',
    '3rd_row': '#ff7f0e',
    'outer_well': '#9467bd',
    'corner': '#d62728',
    'center': '#2ca02c',
    }

    # Get the unique group labels
    labels = df[group_by_column]

    # Plot the specified column vs hours for each group
    plt.figure(figsize=(10, 6), dpi=200)

    for group in sorted(labels.unique()):
        group_data = df[df[group_by_column] == group]
        plt.scatter(group_data[hour_column], group_data[plot_column], label=group, color=color_mapping.get(group, None), alpha=0.50)

    plt.xlabel('Hour')
    plt.ylabel(plot_column.replace('_', ' ').title())
    plt.title(f'{plot_column.replace("_", " ").title()} vs Hour Grouped by {group_by_column.title()}')
    if include_legend:
        plt.legend(title=group_by_column.title(), bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.grid(True)
    plt.xlim([min(group_data[hour_column]), max(group_data[hour_column])])
    plt.xticks(range(min(group_data[hour_column]), max(group_data[hour_column]), 5))  
    plt.tight_layout()
    # plt.ylim(65, 95)
    plt.show()

    # Group by the specified column and hour, then compute the mean and standard deviation
    grouped = df.groupby([group_by_column, hour_column])[plot_column].agg(['mean', 'std']).reset_index()

    # Plot the mean of the specified column vs hours for each group with error bars
    plt.figure(figsize=(10, 6), dpi=200)

    for group in grouped[group_by_column].unique():
        group_data = grouped[grouped[group_by_column] == group]
        plt.errorbar(group_data[hour_column], group_data['mean'], yerr=group_data['std'], label=group, color=color_mapping.get(group, None), fmt='--o',alpha=0.50)

    plt.xlabel('Hour')
    plt.ylabel(f'{plot_column.replace("_", " ").title()}')
    plt.title(f'{plot_column.replace("_", " ").title()} vs Hour Grouped by {group_by_column.title()}')
    if include_legend:
        plt.legend(title=group_by_column.title(), bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False)
    plt.grid(True)
    plt.xlim([min(group_data[hour_column]), max(group_data[hour_column])])
    plt.xticks(range(min(group_data[hour_column]), max(group_data[hour_column]), 5)) 
    plt.tight_layout()
    # plt.ylim(65, 95)
    plt.show()

def plot_time_series_images(image_folder, df, well_ID, start_idx=27, end_idx=52, step=1, show_contours = False):
    """
    Plot a time series of images for a specific well.

    Parameters:
    - image_folder (str): Path to the folder containing images.
    - df (pd.DataFrame): DataFrame containing image metadata.
    - well_ID (str): Well identifier.
    - start_idx (int, optional): Starting index for the images to plot (default is 27).
    - end_idx (int, optional): Ending index for the images to plot (default is 52).
    - step (int, optional): Step size for the images to plot (default is 1).
    - show_contours (bool, optional): Whether to show cell contours (default is False).
    """    
    well_images = df[df['well_ID'] == well_ID].sort_values(['hour', 'minutes', 'seconds'])[['filename', 'timestamp']]

    # If W image and show_contours is True, plot the first 25 images
    if show_contours and list(df['filename'])[0][-5] == 'W': 
        start_idx, end_idx, step = 0, 25, 1

    # Create a figure with 5x5 subplots
    fig, axes = plt.subplots(5, 5, figsize=(20, 20), dpi=200)

    # Loop through the images and plot each one
    for i, row in enumerate(well_images[start_idx:end_idx:step].itertuples()):
        image_path = os.path.join(image_folder, row.filename)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Load the next image in the sequence
        # next_image_path = os.path.join(image_folder, well_images.iloc[i + 1]['filename'])
        # next_image = cv2.imread(next_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Determine the subplot location
        row_idx = i // 5
        col_idx = i % 5

        if show_contours and list(df['filename'])[0][-5] == 'W':
            # Find cell centers for W images
            cell_image, _, _, _ = find_cell_cluster_centers(image)
            # _, _, mask, _ = find_cell_cluster_centers(image)
            # _, _, next_mask, _ = find_cell_cluster_centers(next_image)
            axes[row_idx, col_idx].imshow(cell_image, cmap='gray')

        elif show_contours and list(df['filename'])[0][-5] == 'B':
            # Find cell centers for W images
            cell_image, _, _, _, _, _, _ = find_NKcell_cluster_centers(image)
            # _, _, mask, _ = find_cell_cluster_centers(image)
            # _, _, next_mask, _ = find_cell_cluster_centers(next_image)
            axes[row_idx, col_idx].imshow(cell_image, cmap='gray')
        
        else:
            # Plot the processed image
            axes[row_idx, col_idx].imshow(image, cmap='gray')

        # axes[row_idx, col_idx].imshow(next_image, alpha=0.5, cmap='gray')
        axes[row_idx, col_idx].axis('off')
        axes[row_idx, col_idx].set_title(f'{row.timestamp}')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

# Cell cluster detection in W images
def find_cell_cluster_centers(image, plot_steps=False):
    """
    Process the image and optionally plot all steps.

    Parameters:
    - image (numpy.ndarray): The input grayscale image.
    - plot_steps (bool): Whether to plot all steps (default is False).

    Returns:
    - contour_image (numpy.ndarray): The final image with contours and centers marked.
    - centers (list): List of centers of the detected contours.
    - cell_coordinates (numpy.ndarray): Array with coordinates of all pixels within the smoothed contours.
    - contour_areas (list): List of areas of the smoothed contours.
    """
    # Step 1: Display the original grayscale image
    if plot_steps:
        plt.figure(figsize=(10, 10))
        plt.subplot(2, 3, 1)
        plt.imshow(image, cmap='gray')
        plt.title("Original Grayscale Image")
        plt.axis('off')

    # Step 2: Apply Gaussian Blur
    adjusted_image = cv2.convertScaleAbs(image, alpha=2, beta=1)
    blur = cv2.GaussianBlur(adjusted_image, (5, 5), 0)
    if plot_steps:
        plt.subplot(2, 3, 2)
        plt.imshow(blur, cmap='gray')
        plt.title("Gaussian Blurred Image")
        plt.axis('off')

    # Step 3: Otsu's thresholding
    _, th7 = cv2.threshold(image,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    if plot_steps:
        plt.subplot(2, 3, 3)
        plt.imshow(th7, cmap='gray')
        plt.title("Adaptive Thresholding")
        plt.axis('off')

    # Step 4: Apply Canny edge detection
    img_canny = cv2.Canny(th7, 20, 75)
    if plot_steps:
        plt.subplot(2, 3, 4)
        plt.imshow(img_canny, cmap='gray')
        plt.title("Canny Edge Detection")
        plt.axis('off')

    # thicken edge
    img_canny = cv2.GaussianBlur(img_canny, (7, 7), 0)

    # Step 5: Apply dilation
    kernel = np.ones((3, 3)) 
    img_dilate = cv2.dilate(img_canny, kernel, iterations=4)
    if plot_steps:
        plt.subplot(2, 3, 5)
        plt.imshow(img_dilate, cmap='gray')
        plt.title("Dilation")
        plt.axis('off')

    # Step 6: Final Contour Detection and Display on Original Image
    contours, _ = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours with small area
    min_contour_area = 400  # Define a minimum contour area threshold
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]

    # Draw filtered contours on the original image
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Smooth the contours using approxPolyDP
    smoothed_contours = []
    for contour in filtered_contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        smoothed_contours.append(smoothed_contour)

    # Compute areas of the smoothed contours
    contour_areas = [cv2.contourArea(contour) for contour in smoothed_contours]

    # Extract coordinates of all pixels within the smoothed contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, smoothed_contours, -1, (255), thickness=cv2.FILLED)
    cell_coordinates = mask 

    cv2.drawContours(contour_image, smoothed_contours, -1, (0, 255, 0), 2)
    if plot_steps:
        plt.subplot(2, 3, 6)
        plt.imshow(contour_image)
        plt.title("Final Contours on Original Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Find and mark contour centers
    centers = []
    for contour in filtered_contours:
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
        plt.figure(figsize=(10, 10), dpi = 200)
        plt.imshow(contour_image)
        plt.title("Final Contours with Cell Centers")
        plt.axis('off')
        plt.show()

    return contour_image, centers, cell_coordinates, contour_areas

def find_NKcell_cluster_centers(image, min_contour_area = 0, plot_steps=False):
    """
    Process the image and optionally plot all steps.

    Parameters:
    - image (numpy.ndarray): The input grayscale image.
    - plot_steps (bool): Whether to plot all steps (default is False).
    - min_contour_area (int): Minimum contour area to filter out small contours (default is 0).

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

    # Step 2: Increase contrast and brightness
    adjusted_image = cv2.convertScaleAbs(image, alpha=2, beta=0)
    if plot_steps:
        plt.subplot(2, 3, 2)
        plt.imshow(adjusted_image, cmap='gray')
        plt.title("Increase image contrast")
        plt.axis('off')

    # Step 3: Apply Gaussian Blur
    blur = cv2.GaussianBlur(adjusted_image, (3, 3), 0)
    if plot_steps:
        plt.subplot(2, 3, 3)
        plt.imshow(blur, cmap='gray')
        plt.title("Gaussian Blurred Image")
        plt.axis('off')

    # Step 4: Otsu's thresholding
    _, th = cv2.threshold(blur, 40, 150 ,cv2.THRESH_BINARY+cv2.THRESH_OTSU) 
    if plot_steps:
        plt.subplot(2, 3, 4)
        plt.imshow(th, cmap='gray')
        plt.title("Adaptive Thresholding")
        plt.axis('off')

    ## Step: Initial Contour Detection and Removal of Small Contours
    ## Step: Morphological Closing and Image Inversion

    # Step 5: Apply dilation
    # kernel = np.ones((4, 4))
    kernel = np.ones((2, 2))
    img_dilate = cv2.dilate(th, kernel, iterations=2)
    if plot_steps:
        plt.subplot(2, 3, 5)
        plt.imshow(img_dilate, cmap='gray')
        plt.title("Dilation")
        plt.axis('off')

    # Step 6: Final Contour Detection and Display on Original Image
    contours, _ = cv2.findContours(img_dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Filter out contours with small area
    filtered_contours = [contour for contour in contours if cv2.contourArea(contour) >= min_contour_area]

    # # Draw filtered contours on the original image
    # contour_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Smooth the contours using approxPolyDP
    smoothed_contours = []
    for contour in filtered_contours:
        epsilon = 0.005 * cv2.arcLength(contour, True)
        smoothed_contour = cv2.approxPolyDP(contour, epsilon, True)
        smoothed_contours.append(smoothed_contour)

    # Compute areas of the smoothed contours
    contour_areas = [cv2.contourArea(contour) for contour in smoothed_contours]

    # Extract coordinates of all pixels within the smoothed contours
    mask = np.zeros_like(image)
    cv2.drawContours(mask, smoothed_contours, -1, (255), thickness=cv2.FILLED)
    cell_coordinates = mask #np.column_stack(np.where(mask == 255))

    # Extract pixel intensities for pixels within each of the smoothed contours
    mean_cluster_intensities = []
    summed_cluster_intensities = []
    cluster_areas = []
    for contour in smoothed_contours:
        mask = np.zeros_like(image)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)
        pixel_values = image[mask == 255]

        # Compute area in number of pixels
        cluster_area = len(pixel_values)

        # Calculate mean and summed intensities
        mean_intensity = (pixel_values).mean()
        summed_intensity = (pixel_values).sum()

        mean_cluster_intensities.append(mean_intensity)
        summed_cluster_intensities.append(summed_intensity)
        cluster_areas.append(cluster_area)

    cv2.drawContours(contour_image, smoothed_contours, -1, (0, 255, 0), 1)
    if plot_steps:
        plt.subplot(2, 3, 6)
        plt.imshow(contour_image)
        plt.title("Final Contours on Original Image")
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    # Find and mark contour centers
    centers = []
    for contour in filtered_contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            centers.append((cX, cY))
            # Draw a red cross at each center
            cv2.drawMarker(contour_image, (cX, cY), (255, 0, 0), markerType=cv2.MARKER_CROSS, 
                           markerSize=8, thickness=1)

    if plot_steps:
        plt.figure(figsize=(5, 5), dpi = 200)
        plt.imshow(contour_image)
        plt.title(f"NK cell contours, min area = {min_contour_area}")
        plt.axis('off')
        plt.show()

    return contour_image, centers, cell_coordinates, contour_areas, \
        mean_cluster_intensities, summed_cluster_intensities, cluster_areas

def compute_local_density(segmented_image, window_size=128, step_size=64):
    """
    Compute the local density of white pixels in a segmented image using an overlapping sliding window approach.

    Parameters:
    - segmented_image (numpy.ndarray): The segmented image (binary image with white and black pixels).
    - window_size (int): The size of the sliding window (default is 50).
    - step_size (int): The step size for the sliding window (default is 25).

    Returns:
    - density_map (numpy.ndarray): The density map showing the local density of white pixels.
    """
    # Get the dimensions of the image
    height, width = segmented_image.shape

    # Initialize the density map
    density_map = np.zeros((height, width))

    # Slide the window across the image with overlapping regions
    for y in range(0, height - window_size + 1, step_size):
        for x in range(0, width - window_size + 1, step_size):
            # Define the window
            window = segmented_image[y:y+window_size, x:x+window_size]
            
            # Calculate the density of white pixels in the window
            white_pixels = np.sum(window == 255)
            total_pixels = window.size
            density = white_pixels / total_pixels
            
            # Assign the density value to the corresponding region in the density map
            density_map[y:y+window_size, x:x+window_size] += density

    # Normalize the density map by the number of times each pixel was covered
    normalization_map = np.zeros((height, width))
    for y in range(0, height - window_size + 1, step_size):
        for x in range(0, width - window_size + 1, step_size):
            normalization_map[y:y+window_size, x:x+window_size] += 1

    with np.errstate(divide='ignore', invalid='ignore'):
        density_map /= normalization_map
        density_map[normalization_map == 0] = 0

    return density_map

def compute_density_statistics(density_map, window_size=128, step_size=64):
    """
    Compute descriptive statistics for the density map.

    Parameters:
    - density_map (numpy.ndarray): The density map showing the local density of white pixels.

    Returns:
    - stats (dict): A dictionary containing the mean, standard deviation, minimum, and maximum density values.
    """
    stats = {
        f'mean_density_w{window_size}_s{step_size}': np.mean(density_map),
        f'std_density_w{window_size}_s{step_size}': np.std(density_map),
        f'min_density_w{window_size}_s{step_size}': np.min(density_map),
        f'max_density_w{window_size}_s{step_size}': np.max(density_map)
    }
    return stats

