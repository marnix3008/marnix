import cv2
import numpy as np
from pathlib import Path


def blue_hsv_bounds_from_samples() -> tuple[tuple[int, int, int], tuple[int, int, int]]:
	# User-provided RGBA samples (alpha ignored).
	rgb_samples = [
		(54, 115, 196),
		(72, 116, 205),
		(25, 117, 200),
		(27, 115, 212),
		(27, 115, 212),
		(25, 115, 212),
	]

	# Convert RGB samples to OpenCV BGR image for HSV conversion.
	bgr_samples = np.array([[[b, g, r] for (r, g, b) in rgb_samples]], dtype=np.uint8)
	hsv_samples = cv2.cvtColor(bgr_samples, cv2.COLOR_BGR2HSV)[0]

	h_vals = [int(px[0]) for px in hsv_samples]
	s_vals = [int(px[1]) for px in hsv_samples]
	v_vals = [int(px[2]) for px in hsv_samples]

	# Small padding keeps filter tight but tolerant to camera noise.
	h_pad, s_pad, v_pad = 5, 25, 25
	lower = (
		max(0, min(h_vals) - h_pad),
		max(0, min(s_vals) - s_pad),
		max(0, min(v_vals) - v_pad),
	)
	upper = (
		min(179, max(h_vals) + h_pad),
		min(255, max(s_vals) + s_pad),
		min(255, max(v_vals) + v_pad),
	)

	return lower, upper


def keep_tall_regions(mask: np.ndarray, min_height_width_ratio: float, min_area: int) -> tuple[np.ndarray, int]:
	contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	tall_mask = np.zeros_like(mask)
	kept_regions = 0

	for contour in contours:
		x, y, w, h = cv2.boundingRect(contour)
		area = cv2.contourArea(contour)
		if w == 0:
			continue

		ratio = h / float(w)
		if ratio >= min_height_width_ratio and area >= min_area:
			cv2.drawContours(tall_mask, [contour], contourIdx=-1, color=255, thickness=-1)
			kept_regions += 1

	return tall_mask, kept_regions


def cluster_blobs(mask: np.ndarray, dilation_kernel_size: int = 10) -> tuple[np.ndarray, int, list[dict]]:
	"""
	Use connected component analysis to find and label separate blobs.
	Applies morphological dilation to close small gaps between blob parts.
	
	Args:
		- mask: Binary mask of detected regions
		- dilation_kernel_size: Size of the dilation kernel to close gaps. Larger = less strict clustering.
	
	Returns:
		- labeled_mask: Image where each blob has a unique label (0 = background)
		- num_blobs: Number of blobs found
		- blob_info: List of dicts with stats for each blob (area, centroid, bounding box)
	"""
	# Dilate to close small gaps between blob parts
	kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilation_kernel_size, dilation_kernel_size))
	dilated_mask = cv2.dilate(mask, kernel, iterations=1)
	
	# Perform connected component analysis on dilated mask
	num_labels, labeled_mask = cv2.connectedComponents(dilated_mask, connectivity=8)
	
	blob_info = []
	# num_labels includes background (label 0), so actual blobs are 1 to num_labels-1
	for label in range(1, num_labels):
		blob_mask = (labeled_mask == label).astype(np.uint8) * 255
		area = cv2.countNonZero(blob_mask)
		
		# Find centroid
		moments = cv2.moments(blob_mask)
		if moments["m00"] > 0:
			cx = int(moments["m10"] / moments["m00"])
			cy = int(moments["m01"] / moments["m00"])
		else:
			cx, cy = 0, 0
		
		# Find bounding box
		x, y, w, h = cv2.boundingRect(blob_mask)
		
		blob_info.append({
			"label": label,
			"area": area,
			"centroid": (cx, cy),
			"bbox": (x, y, w, h),
		})
	
	return labeled_mask.astype(np.uint8), num_labels - 1, blob_info


def colorize_blobs(labeled_mask: np.ndarray, num_blobs: int) -> np.ndarray:
	"""
	Colorize each blob with a different shade of blue.
	
	Returns:
		- colored_image: RGB image with each blob colored differently
	"""
	# Create an empty RGB image
	colored_image = np.zeros((labeled_mask.shape[0], labeled_mask.shape[1], 3), dtype=np.uint8)
	
	# Define a palette of blue shades (in BGR format for OpenCV)
	# Each blob gets progressively darker or lighter blue
	colors = []
	for i in range(1, num_blobs + 1):
		# Create variations of blue: hue stays around blue, vary saturation/value
		ratio = i / (num_blobs + 1)
		# Light to dark blue shades
		blue = int(255 * (0.5 + 0.5 * ratio))
		green = int(100 * ratio)
		red = int(50 * ratio)
		colors.append((blue, green, red))
	
	# Color each blob
	for label in range(1, num_blobs + 1):
		blob_mask = (labeled_mask == label)
		colored_image[blob_mask] = colors[label - 1]
	
	return colored_image


def compute_gate_bounding_box(blob_info: list[dict]) -> dict:
	"""
	Compute a gate trapezoid from left and right blobs.
	Uses the actual corner positions of each blob to create a trapezoid shape:
	- Top-left: top-left corner of left blob
	- Top-right: top-right corner of right blob
	- Bottom-left: bottom-left corner of left blob
	- Bottom-right: bottom-right corner of right blob
	
	Args:
		- blob_info: List of blob dictionaries with bbox info
	
	Returns:
		- gate_bbox: Dict with trapezoid 'corners' as a list of (x, y) points in order
	"""
	if len(blob_info) < 2:
		return None  # Need at least 2 blobs for a gate
	
	# Sort blobs by x position (center of bounding box)
	sorted_blobs = sorted(blob_info, key=lambda b: b["bbox"][0] + b["bbox"][2] / 2)
	
	left_blob = sorted_blobs[0]
	right_blob = sorted_blobs[-1]
	
	left_x, left_y, left_w, left_h = left_blob["bbox"]
	right_x, right_y, right_w, right_h = right_blob["bbox"]
	
	# Trapezoid corners using actual blob positions
	top_left = (left_x, left_y)
	top_right = (right_x + right_w, right_y)
	bottom_left = (left_x, left_y + left_h)
	bottom_right = (right_x + right_w, right_y + right_h)
	
	# List of corners in order (for polygon drawing)
	corners_list = [top_left, top_right, bottom_right, bottom_left]
	
	return {
		"corners": corners_list,
		"points": {
			"top_left": top_left,
			"top_right": top_right,
			"bottom_left": bottom_left,
			"bottom_right": bottom_right,
		},
	}


def draw_gate_bounding_box(image: np.ndarray, gate_bbox: dict, color: tuple = (0, 255, 0), thickness: int = 2) -> np.ndarray:
	"""
	Draw the gate trapezoid on the image.
	
	Args:
		- image: Input image (BGR format)
		- gate_bbox: Gate bounding box dict from compute_gate_bounding_box
		- color: Color in BGR format (default green)
		- thickness: Line thickness
	
	Returns:
		- image_with_bbox: Image with trapezoid drawn
	"""
	if gate_bbox is None:
		return image
	
	result = image.copy()
	corners_list = gate_bbox["corners"]
	
	# Draw trapezoid polygon
	corners_array = np.array(corners_list, dtype=np.int32)
	cv2.polylines(result, [corners_array], isClosed=True, color=color, thickness=thickness)
	
	# Draw corner circles for clarity
	for corner_pos in corners_list:
		cv2.circle(result, corner_pos, radius=5, color=color, thickness=-1)
	
	return result


def create_comparison_image(original: np.ndarray, processed: np.ndarray, label_original: str = "Original", label_processed: str = "Processed") -> np.ndarray:
	"""
	Stack original and processed images vertically for easy comparison.
	
	Args:
		- original: Original rotated image
		- processed: Processed image with gate bounding box
		- label_original: Text label for original image
		- label_processed: Text label for processed image
	
	Returns:
		- comparison_image: Stacked image with both original and processed
	"""
	# Ensure both images have the same width (resize if needed)
	h1, w1 = original.shape[:2]
	h2, w2 = processed.shape[:2]
	
	if w1 != w2:
		target_w = min(w1, w2)
		original = cv2.resize(original, (target_w, int(h1 * target_w / w1)))
		processed = cv2.resize(processed, (target_w, int(h2 * target_w / w2)))
	
	# Add labels to images
	font = cv2.FONT_HERSHEY_SIMPLEX
	font_scale = 1.0
	font_thickness = 2
	text_color = (0, 255, 0)
	
	original_labeled = original.copy()
	processed_labeled = processed.copy()
	
	cv2.putText(original_labeled, label_original, (10, 30), font, font_scale, text_color, font_thickness)
	cv2.putText(processed_labeled, label_processed, (10, 30), font, font_scale, text_color, font_thickness)
	
	# Stack vertically
	comparison = np.vstack([original_labeled, processed_labeled])
	
	return comparison


def rotate_filter_blue_and_save(
	input_path: str,
	output_path: str,
	min_height_width_ratio: float = 1.8,
	min_area: int = 80,
) -> tuple[int, int, int, list[dict], np.ndarray, dict]:
	image = cv2.imread(input_path)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {input_path}")

	rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	# Tight blue filter from provided sample colors.
	hsv = cv2.cvtColor(rotated, cv2.COLOR_BGR2HSV)
	lower_blue, upper_blue = blue_hsv_bounds_from_samples()
	blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
	tall_blue_mask, kept_regions = keep_tall_regions(
		blue_mask,
		min_height_width_ratio=min_height_width_ratio,
		min_area=min_area,
	)
	
	# Cluster blobs to differentiate them
	labeled_mask, num_blobs, blob_info = cluster_blobs(tall_blue_mask)
	
	# Colorize blobs
	colored_blobs = colorize_blobs(labeled_mask, num_blobs)
	
	# Compute gate bounding box (if we have left and right blobs)
	gate_bbox = compute_gate_bounding_box(blob_info) if num_blobs >= 2 else None
	
	# Draw gate bounding box on colored image
	if gate_bbox is not None:
		colored_blobs = draw_gate_bounding_box(colored_blobs, gate_bbox, color=(0, 255, 0), thickness=2)
		original_with_gate = draw_gate_bounding_box(rotated, gate_bbox, color=(0, 255, 0), thickness=2)
	else:
		original_with_gate = rotated.copy()
	
	# Create comparison image (original + processed)
	comparison_image = create_comparison_image(
		original_with_gate,
		colored_blobs,
		label_original="Original Rotated + Gate",
		label_processed="Detected Blobs & Gate",
	)
	
	# Save comparison visualization
	saved = cv2.imwrite(output_path, comparison_image)
	if not saved:
		raise RuntimeError(f"Failed to save image: {output_path}")

	blue_pixel_count = int(cv2.countNonZero(tall_blue_mask))

	return blue_pixel_count, kept_regions, num_blobs, blob_info, colored_blobs, gate_bbox


if __name__ == "__main__":
	script_dir = Path(__file__).resolve().parent
	input_folder = script_dir / "Images/drive-download-20260318T090553Z-3-001"
	output_dir = script_dir / "output"
	output_dir.mkdir(parents=True, exist_ok=True)

	min_ratio = 1.8
	min_area = 80

	# Get all image files (jpg, png, etc.)
	image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
	image_files = sorted([
		f for f in input_folder.iterdir()
		if f.is_file() and f.suffix.lower() in image_extensions
	])

	if not image_files:
		print(f"No image files found in {input_folder}")
	else:
		print(f"Found {len(image_files)} image(s) to process\n")

		for image_path in image_files:
			output_filename = f"{image_path.stem}_rotated_left_blue_tall_filter.jpg"
			output_path = output_dir / output_filename

			try:
				blue_pixels, kept_regions, num_blobs, blob_info, colored_image, gate_bbox = rotate_filter_blue_and_save(
					str(image_path),
					str(output_path),
					min_height_width_ratio=min_ratio,
					min_area=min_area,
				)
				print(f"✓ Processed: {image_path.name}")
				print(f"  Saved to: {output_path}")
				print(f"  Detected blue pixels (tall only): {blue_pixels}")
				print(f"  Detected tall blue regions: {kept_regions}")
				print(f"  Clustered blobs found: {num_blobs}")
				for i, blob in enumerate(blob_info, 1):
					x, y, w, h = blob["bbox"]
					cx, cy = blob["centroid"]
					print(f"    Blob {i}: Area={blob['area']}px, Centroid=({cx}, {cy}), BBox=x={x} y={y} w={w} h={h}")
				
				if gate_bbox is not None:
					points = gate_bbox.get("points", {})
					print(f"  Gate Trapezoid:")
					print(f"    Top-Left: {points.get('top_left', 'N/A')}")
					print(f"    Top-Right: {points.get('top_right', 'N/A')}")
					print(f"    Bottom-Left: {points.get('bottom_left', 'N/A')}")
					print(f"    Bottom-Right: {points.get('bottom_right', 'N/A')}")
				else:
					print(f"  Gate Bounding Box: Not computed (need at least 2 blobs)")
				print()
			except Exception as e:
				print(f"✗ Error processing {image_path.name}: {e}\n")

		print(f"Processing complete. All images saved to: {output_dir}")
		print(f"Used height/width ratio threshold: {min_ratio}")
		print(f"Used minimum region area: {min_area}")
		print(f"Using HSV bounds: {blue_hsv_bounds_from_samples()}")
