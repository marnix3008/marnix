import cv2
from pathlib import Path
import shutil
import subprocess

try:
	import matplotlib.pyplot as plt
except ImportError:
	plt = None


def show_rotated_image_left(input_path: str) -> None:
	image = cv2.imread(input_path)
	if image is None:
		raise FileNotFoundError(f"Could not read image: {input_path}")

	rotated = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

	if plt is not None:
		rotated_rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)
		plt.imshow(rotated_rgb)
		plt.title("Rotated Left (90 degrees)")
		plt.axis("off")
		plt.show()
	else:
		# Fallback for headless OpenCV: save and open with system image viewer.
		output_path = Path(input_path).with_name("237952445_rotated_left_preview.jpg")
		cv2.imwrite(str(output_path), rotated)

		xdg_open = shutil.which("xdg-open")
		if xdg_open:
			subprocess.run([xdg_open, str(output_path)], check=False)
			print(f"Opened preview with system viewer: {output_path}")
		else:
			print(f"Preview saved at: {output_path}")


if __name__ == "__main__":
	script_dir = Path(__file__).resolve().parent
	input_image = script_dir / "Images/drive-download-20260318T090553Z-3-001/237952445.jpg"

	show_rotated_image_left(str(input_image))
