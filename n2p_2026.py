#!/home/vic/.venvs/n2p_2026/bin/python
#import pdb; pdb.set_trace()
import os
import glob
import argparse
import subprocess
import numpy as np
from sklearn.linear_model import TheilSenRegressor
from scipy.cluster.hierarchy import linkage, fcluster
import concurrent.futures
import imageio.v3 as iio
import matplotlib.pyplot as plt
import rawpy
import tifffile

# ---------------------------------------------------------
# USER SETTINGS
# ---------------------------------------------------------

LOW_PCT = 1.0      # black point percentile
HIGH_PCT = 99.0    # white point percentile

WB_FILTER_LOW_PCT = 5.0    # Low percentile for luminance-based WB pixel filtering
WB_FILTER_HIGH_PCT = 95.0  # High percentile for luminance-based WB pixel filtering

GRID_BINS = 500            # Resolution of the chromaticity grid
GRID_RANGE = 2.0           # Range of the grid in log space (+/-)
BUCKET_THRESHOLD_PCT = 0.0001 # Minimum percentage of total pixels to keep a bucket (0.01 = 1%)

IGNORE_BORDER = 300  # ignore a border of 300 pixels when calculating black/white points

INTERMEDIATE_DIR = "intermediate"
OUTPUT_DIR = "final"

TARGET_MIN = 0.0
TARGET_MAX = 1.0   # we work in normalized float space

GAMMA = 0.7        # change later if desired
BRIGHTNESS = 0.0   # additive
CONTRAST = 1.0     # multiplicative

VIGNETTING_REFERENCE_IMAGE = "/home/vic/Pictures/vignetting-single-channel.tiff"
# requirements: 16 bit grayscale tiff with exact same size as raw images

RAW_EXPORT = False  # save the raw images as 16 bit tiff (useful for creating a vignetting image)

# ---------------------------------------------------------
# RAW IMPORT STUB (YOU IMPLEMENT THIS)
# ---------------------------------------------------------

def load_raw_channel(filename: str, channel: int) -> np.ndarray:
    """
    Load a RAW image file and return a 2D numpy array
    of dtype uint16 corresponding to the requested channel.

    channel: 0 = R, 1 = G, 2 = B

    You will replace this with your own implementation.
    """
    with rawpy.imread(filename) as raw: # tries to read as raw file
        # https://letmaik.github.io/rawpy/api/rawpy.Params.html#rawpy.Params
        raw_image = raw.postprocess(
            output_bps = 16, # output 16-bit image
            use_camera_wb = True, # Screws up the colours if not used
            user_wb = [1, 1, 1, 1], # wb multipliers
            demosaic_algorithm = rawpy.DemosaicAlgorithm(0),
            output_color = rawpy.ColorSpace(0),
            gamma = (1.0,1.0),
            auto_bright_thr = 0, # no clipping of highlights
            exp_preserve_highlights = 1,
            exp_shift = 1,  # was: 2 ** 3
            half_size = False # take the average of 4 pixels to reduce resolution and computational requirements (because no interpolation is done)
            )
        raw_channel = raw_image[:, :, channel]
    return raw_channel

# ---------------------------------------------------------
# UTILITY FUNCTIONS
# ---------------------------------------------------------

def convert_to_float(img_u16: np.ndarray) -> np.ndarray:
    """Convert a 16-bit image to float."""
    img = img_u16.astype(np.float32)
    return img


def correct_vignetting(img_float: np.ndarray) -> np.ndarray:
    """Apply correction for lens used during digitizing, based on reference flat field image."""
    img_float = vignetting_high * img_float / vignetting_reference_image[..., None]
    return img_float


def compute_black_white(img: np.ndarray):
    """Compute black/white points using percentiles."""
    black = np.percentile(img, LOW_PCT)
    white = np.percentile(img, HIGH_PCT)
    return black, white


def apply_levels(img, black, white):
    """Apply linear black/white scaling."""
    out = (img - black) / (white - black)
    return np.clip(out, 0.0, 1.0)


def apply_bc_gamma(img):
    """Apply brightness/contrast/gamma equally to all channels."""
    out = img * CONTRAST + BRIGHTNESS
    out = np.clip(out, 0.0, 1.0)
    if GAMMA != 1.0:
        out = out ** (1.0 / GAMMA)
    return out


def calculate_luminance(img: np.ndarray) -> np.ndarray:
    """Computes the luminance of an RGB image using Rec. 709 coefficients."""
    return (
        0.2126 * img[..., 0] +
        0.7152 * img[..., 1] +
        0.0722 * img[..., 2]
    )


def compute_auto_bc_from_luminance(img):
    """
    Compute optimal brightness and contrast for an image
    based on luminance percentiles.
    """
    L = calculate_luminance(img)

    L_black = np.percentile(L, LOW_PCT)
    L_white = np.percentile(L, HIGH_PCT)

    if L_white <= L_black:
        return 0.0, 1.0  # no-op safety

    contrast = 1.0 / (L_white - L_black)
    brightness = -L_black * contrast

    return brightness, contrast


def get_wb_luminance_range(img):
    """
    Compute a luminance range from the darkest pixels of an image.
    This can be used to identify pixels in unexposed areas (film base)
    for white/black balance.
    """
    L = calculate_luminance(img)

    # Use very low percentiles to isolate the darkest parts of the image
    L_min = np.percentile(L, WB_FILTER_LOW_PCT)
    L_max = np.percentile(L, WB_FILTER_HIGH_PCT)

    return L_min, L_max


# Load vignetting reference globally so it is available to worker processes
vignetting_reference_image = tifffile.imread(VIGNETTING_REFERENCE_IMAGE)
_, vignetting_high = compute_black_white(vignetting_reference_image)

# -----------------------------------------------------
# WORKER FUNCTIONS (Defined at top level to be picklable)
# -----------------------------------------------------

def process_raw_group_task(args):
    """Worker for Phase 1: Raw -> Intermediate"""
    i, fR, fG, fB, half_frame = args
    print(f"Processing raw group {i//3 + 1}...")

    # 3.1 Import RAW channels
    R = load_raw_channel(fR, 0)
    G = load_raw_channel(fG, 1)
    B = load_raw_channel(fB, 2)

    # 3.2 Recombine into RGB image
    img = np.stack([R, G, B], axis=-1)

    # 3.2 bis Save raw 16 bit tiff image 
    if RAW_EXPORT:
        out_name = os.path.join(INTERMEDIATE_DIR, f"raw_{i//3:04d}.tif")
        iio.imwrite(out_name, img)

    # 3.3 Convert to float, correct for vignetting
    img = convert_to_float(img)
    img = correct_vignetting(img)

    sub_images_data = []
    if half_frame:
        h_img, w_img, _ = img.shape
        mid = w_img // 2
        cut_border = 40
        sub_images_data.append((img[:, :mid-cut_border, :], "_left"))
        sub_images_data.append((img[:, mid+cut_border:, :], "_right"))
    else:
        sub_images_data.append((img, ""))

    results = []
    for sub_img, suffix in sub_images_data:
        # Calculate min/max luminances
        L_min, L_max = get_wb_luminance_range(sub_img[IGNORE_BORDER:-IGNORE_BORDER, IGNORE_BORDER:-IGNORE_BORDER, :])
        
        # Save intermediate
        out_name = os.path.join(INTERMEDIATE_DIR, f"frame_{i//3:04d}{suffix}.npy")
        np.save(out_name, sub_img)
        results.append((out_name, L_min, L_max))
    return results

def analyze_frame_task(args):
    """Worker for Phase 2: Analysis"""
    idx, fname, L_min_bound, L_max_bound, n, u, v = args
    print(f"Analyzing frame {idx + 1}...")
    
    local_chroma_grid = np.zeros((GRID_BINS, GRID_BINS), dtype=np.int32)
    img = np.load(fname)
    
    luminance = calculate_luminance(img)
    mask_lum = (luminance >= L_min_bound) & (luminance <= L_max_bound)
    pixels_after_lum = img[mask_lum]

    median_result = None
    
    if pixels_after_lum.shape[0] > 0:
        pixels_subset = pixels_after_lum[::100]
        pixels_subset = np.clip(pixels_subset, 1e-9, None)
        density_pixels = -np.log(pixels_subset)
        dot_products = (density_pixels @ n).reshape(-1, 1)
        projections = density_pixels - dot_products * n
        
        u_coords = projections @ u
        v_coords = projections @ v
        
        u_indices = ((u_coords + GRID_RANGE) / (2 * GRID_RANGE) * GRID_BINS).astype(int)
        v_indices = ((v_coords + GRID_RANGE) / (2 * GRID_RANGE) * GRID_BINS).astype(int)
        
        valid_mask = (u_indices >= 0) & (u_indices < GRID_BINS) & \
                        (v_indices >= 0) & (v_indices < GRID_BINS)
        
        np.add.at(local_chroma_grid, (v_indices[valid_mask], u_indices[valid_mask]), 1)

    # Calculate median
    total_local_pixels = np.sum(local_chroma_grid)
    if total_local_pixels > 0:
        local_threshold = total_local_pixels * BUCKET_THRESHOLD_PCT
        local_active_mask = local_chroma_grid > local_threshold
        if np.any(local_active_mask):
            v_inds, u_inds = np.nonzero(local_active_mask)
            u_vals = (u_inds + 0.5) / GRID_BINS * (2 * GRID_RANGE) - GRID_RANGE
            v_vals = (v_inds + 0.5) / GRID_BINS * (2 * GRID_RANGE) - GRID_RANGE
            median_result = (np.median(u_vals), np.median(v_vals))

    return (local_chroma_grid, median_result, idx)

def render_frame_task(args):
    """Worker for Phase 3: Rendering"""
    idx, fname, cluster_id, direction = args
    print(f"Rendering final image {idx + 1}...")
    
    img = np.load(fname)
    
    # Apply direction
    out = img / direction
    
    # Auto BC
    auto_brightness, auto_contrast = compute_auto_bc_from_luminance(out[IGNORE_BORDER:-IGNORE_BORDER, IGNORE_BORDER:-IGNORE_BORDER, :])
    
    out = 1.0 - (out * auto_contrast + auto_brightness)
    out = np.clip(out, 0.0, 1.0)
    
    if GAMMA != 1.0:
        out = out ** (1.0 / GAMMA)
        
    out_u8 = (out * 255.0).astype(np.uint8)
    out_name = os.path.join(OUTPUT_DIR, f"frame_{idx:04d}_c{cluster_id}.jpg")
    iio.imwrite(out_name, out_u8)

# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Process negative scans", add_help=False)
    parser.add_argument("--help", action="help", help="show this help message and exit")
    parser.add_argument("-h", "--half-frame", action="store_true", help="Split frame in half (left/right) with border cut")
    args, _ = parser.parse_known_args()

    base_dir = os.getcwd()

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # -----------------------------------------------------
    # 2. Prepare lists for black/white points
    # -----------------------------------------------------

    min_luminances = []
    max_luminances = []

    # -----------------------------------------------------
    # 3. Iterate files in groups of three
    # -----------------------------------------------------

    files = sorted(glob.glob(os.path.join(base_dir, "*.ARW")))

    if len(files) % 3 != 0:
        raise RuntimeError("Number of files is not divisible by 3")

    intermediate_files = []

    total_groups = len(files) // 3
    raw_tasks = []
    for i in range(0, len(files), 3):
        fR, fG, fB = files[i:i+3]
        raw_tasks.append((i, fR, fG, fB, args.half_frame))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(f"Starting parallel processing of {total_groups} raw groups...")
        for result_list in executor.map(process_raw_group_task, raw_tasks):
            for (out_name, L_min, L_max) in result_list:
                intermediate_files.append(out_name)
                min_luminances.append(L_min)
                max_luminances.append(L_max)


        # -----------------------------------------------------
        # 4. Calculate Direction Vector from film base
        # -----------------------------------------------------

        print("Calculating direction vector from all pixels...")
        
        # Setup Grid
        chroma_grid = np.zeros((GRID_BINS, GRID_BINS), dtype=np.int32)
        image_medians = []
        valid_file_indices = []
        
        n = np.ones(3) / np.sqrt(3)
        # Define basis vectors u, v orthogonal to n
        u = np.array([1.0, -1.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(n, u)

        analyze_tasks = []
        for idx, fname in enumerate(intermediate_files):
            analyze_tasks.append((idx, fname, min_luminances[idx], max_luminances[idx], n, u, v))

        print("Starting parallel analysis...")
        for (local_grid, median_res, idx) in executor.map(analyze_frame_task, analyze_tasks):
            chroma_grid += local_grid
            if median_res is not None:
                image_medians.append(median_res)
                valid_file_indices.append(idx)

        if not image_medians:
            raise ValueError("No valid medians found. Try adjusting WB_FILTER_LOW_PCT and WB_FILTER_HIGH_PCT.")

        image_medians = np.array(image_medians)
        
        # -----------------------------------------------------
        # Clustering Logic (Ward Linkage + Largest Jump)
        # -----------------------------------------------------
        print("Clustering image medians...")
        n_samples = len(image_medians)
        
        if n_samples < 2:
            num_clusters = 1
            labels = np.ones(n_samples, dtype=int)
        else:
            # Ward linkage
            Z = linkage(image_medians, method='ward')
            distances = Z[:, 2]
            
            # Find optimal k (1 to 5) based on largest distance jump
            # We look at the last few merges
            rev_dist = distances[::-1]
            jumps = []
            
            # Check jumps for k=2 to 5
            max_check = min(5, len(rev_dist))
            
            # Calculate jumps: jump[i] corresponds to transition from i+2 clusters to i+1
            # We want to find the split that maximizes the distance gap
            for i in range(max_check - 1):
                jump = rev_dist[i] - rev_dist[i+1]
                # Weight the jump by the number of clusters to favor more clusters
                jumps.append((jump * (i + 2), i + 2))
                
            if not jumps:
                num_clusters = 1
                labels = np.ones(n_samples, dtype=int)
            else:
                # Find k with max jump
                best_jump, best_k = max(jumps, key=lambda x: x[0])
                num_clusters = best_k
                labels = fcluster(Z, num_clusters, criterion='maxclust')
                print(f"Auto-detected {num_clusters} clusters (max jump: {best_jump:.4f}).")

        # Calculate centroids (Median of Medians) and Direction Vectors for each cluster
        cluster_centroids = {}   # cluster_id -> (u, v)
        cluster_directions = {}  # cluster_id -> direction_vector (RGB)
        
        unique_labels = np.unique(labels)
        for lab in unique_labels:
            mask = labels == lab
            cluster_points = image_medians[mask]
            med_u = np.median(cluster_points[:, 0])
            med_v = np.median(cluster_points[:, 1])
            cluster_centroids[lab] = (med_u, med_v)
            
            # Calculate direction vector for this cluster
            c_median = med_u * u + med_v * v
            d_final = c_median + n
            cluster_directions[lab] = np.exp(-d_final)

        # Visualization of the chromaticity plane
        axes = np.eye(3)
        axes_dots = axes @ n
        axes_proj = axes - np.outer(axes_dots, n)
        axes_x = axes_proj @ u
        axes_y = axes_proj @ v

        # Determine scale for visualization
        scale = GRID_RANGE * 0.8

        plt.figure(figsize=(10, 8))
        extent = [-GRID_RANGE, GRID_RANGE, -GRID_RANGE, GRID_RANGE]
        plt.imshow(np.log1p(chroma_grid), extent=extent, origin='lower', cmap='inferno')
        plt.colorbar(label='Log Pixel Count')
        
        # Plot image medians colored by cluster
        scatter = plt.scatter(image_medians[:, 0], image_medians[:, 1], c=labels, cmap='tab10', s=40, marker='o', label='Image Medians', zorder=12, alpha=0.9, edgecolors='black')
        
        # Plot cluster centroids
        centroid_coords = np.array(list(cluster_centroids.values()))
        plt.scatter(centroid_coords[:, 0], centroid_coords[:, 1], c='cyan', s=200, marker='X', label='Cluster Centroids', zorder=11, edgecolors='black')
        
        plt.legend(*scatter.legend_elements(), title="Clusters")

        # Plot R, G, B axes
        axis_labels = ['R', 'G', 'B']
        colors = ['red', 'green', 'blue']
        for i in range(3):
            plt.arrow(0, 0, axes_x[i] * scale, axes_y[i] * scale, 
                      color=colors[i], width=0.01*scale, head_width=0.05*scale, 
                      length_includes_head=True, zorder=11)
            plt.text(axes_x[i] * scale * 1.15, axes_y[i] * scale * 1.15, axis_labels[i], 
                     color=colors[i], fontsize=12, fontweight='bold', ha='center', va='center')

        plt.title('Projected Chromaticity (Plane $\perp$ [1,1,1])')
        plt.xlabel('U axis (approx. Red - Green)')
        plt.ylabel('V axis (approx. Yellow - Blue)')
        plt.axis('equal')

        graph_path = os.path.join(INTERMEDIATE_DIR, "chromaticity_graph.png")
        plt.savefig(graph_path)
        plt.close()

        # Display graph using system default viewer without blocking
        subprocess.Popen(['xdg-open', graph_path])
        
        # Map file index to cluster ID for easy lookup
        file_to_cluster = {}
        # Default to the most populous cluster for any files that might have been skipped (though unlikely with current logic)
        most_populous_cluster = np.bincount(labels).argmax()
        
        for i, file_idx in enumerate(valid_file_indices):
            file_to_cluster[file_idx] = labels[i]
            
        # Print cluster info
        for lab, direction in cluster_directions.items():
            inv_dir = 1.0 / direction
            scaled_inv = (inv_dir / np.max(inv_dir)) * 100
            print(f"Cluster {lab} WB: {int(round(scaled_inv[0]))} {int(round(scaled_inv[1]))} {int(round(scaled_inv[2]))}")
            
        # -----------------------------------------------------
        # 5. Apply global scaling + per-image auto BC + gamma
        # -----------------------------------------------------

        render_tasks = []
        for idx, fname in enumerate(intermediate_files):
            cluster_id = file_to_cluster.get(idx, most_populous_cluster)
            direction = cluster_directions[cluster_id]
            render_tasks.append((idx, fname, cluster_id, direction))

        print("Starting parallel rendering...")
        list(executor.map(render_frame_task, render_tasks))


# ---------------------------------------------------------
# ENTRY POINT
# ---------------------------------------------------------

if __name__ == "__main__":
    import multiprocessing
    main()
