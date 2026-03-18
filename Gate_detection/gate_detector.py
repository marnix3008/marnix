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


def rotate_filter_blue_and_save(
	input_path: str,
	output_path: str,
	min_height_width_ratio: float = 1.8,
	min_area: int = 80,
) -> tuple[int, int]:
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
	filtered = cv2.bitwise_and(rotated, rotated, mask=tall_blue_mask)

	blue_pixel_count = int(cv2.countNonZero(tall_blue_mask))
	saved = cv2.imwrite(output_path, filtered)
	if not saved:
		raise RuntimeError(f"Failed to save image: {output_path}")

	return blue_pixel_count, kept_regions


if __name__ == "__main__":
	script_dir = Path(__file__).resolve().parent
	input_image = script_dir / "Images/drive-download-20260318T090553Z-3-001/240819070.jpg"
	output_dir = script_dir / "output"
	output_dir.mkdir(parents=True, exist_ok=True)
	output_image = output_dir / "240819070_rotated_left_blue_tall_filter.jpg"

	min_ratio = 1.8
	min_area = 80

	blue_pixels, kept_regions = rotate_filter_blue_and_save(
		str(input_image),
		str(output_image),
		min_height_width_ratio=min_ratio,
		min_area=min_area,
	)
	print(f"Saved tall-blue filtered image to: {output_image}")
	print(f"Detected blue pixels (tall only): {blue_pixels}")
	print(f"Detected tall blue regions: {kept_regions}")
	print(f"Used height/width ratio threshold: {min_ratio}")
	print(f"Used minimum region area: {min_area}")
	print(f"Using HSV bounds: {blue_hsv_bounds_from_samples()}")
