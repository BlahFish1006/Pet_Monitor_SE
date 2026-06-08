"""YOLO-World pet (dog/cat) vision detector — vision core of Pet Edge Tracking System.

Converted from YOLO_World_Customize_Model_SE_project.ipynb (originally a Google
Colab notebook) into a runnable local module. Colab-specific pieces have been
adapted for local use:

  * `from google.colab import drive` / `drive.mount(...)`  -> removed
  * `!pip install -q ultralytics`                          -> see requirements.txt
  * `from google.colab.patches import cv2_imshow`          -> cv2.imwrite / cv2.imshow
  * hardcoded /content/drive/MyDrive paths                 -> CLI args / constants

The detection and analysis logic (model customization + the four analysis
functions) is preserved verbatim from the notebook.

Usage examples:
  # Build a custom open-vocabulary model and save it locally
  python yolo_world_detector.py --customize --classes dog,cat --save-model dogandcat.pt

  # Run detection on an image or a directory, writing annotated copies to out/
  python yolo_world_detector.py --model dogandcat.pt --predict path/to/img_or_dir --save-dir out

  # Analyse detection confidence statistics for a class over a directory
  python yolo_world_detector.py --model dogandcat.pt --analyze path/to/dir --class dog
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from ultralytics import YOLO

# OpenCV is used to save / display annotated frames locally (replacing cv2_imshow).
try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False

# matplotlib with a non-interactive backend so histograms can be saved headless.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False


DEFAULT_BASE_MODEL = "yolov8l-world.pt"
DEFAULT_CLASSES = ["dog", "cat"]
DEFAULT_CONF = 0.25
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".gif")


# --------------------------------------------------------------------------- #
# Model build / load  (notebook cells 3 & 4)
# --------------------------------------------------------------------------- #
def build_custom_model(
    classes: list[str] | None = None,
    base_model: str = DEFAULT_BASE_MODEL,
    save_path: str | None = None,
) -> YOLO:
    """Load a YOLO-World base model, set its open vocabulary, optionally save.

    Port of notebook cell 3. Downloads `base_model` from the Ultralytics
    release assets on first use if it is not present locally.
    """
    classes = classes or list(DEFAULT_CLASSES)

    # Initialize a YOLO-World model
    model = YOLO(base_model)

    # Define custom classes (open-vocabulary via CLIP embeddings)
    model.set_classes(classes)

    # Save the model with the defined offline vocabulary
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        model.save(save_path)
        print(f"[*] Customized YOLO-World model saved at {save_path}")

    return model


def load_model(model_path: str) -> YOLO:
    """Reload a previously customized YOLO-World model. Port of cell 4."""
    model = YOLO(model_path)
    print(f"[*] Model loaded from {model_path}")
    return model


# --------------------------------------------------------------------------- #
# Inference on images / directories  (notebook cell 4)
# --------------------------------------------------------------------------- #
def predict_path(
    model: YOLO,
    source: str,
    conf: float = DEFAULT_CONF,
    save_dir: str | None = None,
    show: bool = False,
) -> list:
    """Run inference on an image or directory and annotate the results.

    Replaces the notebook's Colab `cv2_imshow(im_array)` with local equivalents:
    annotated frames are written to `save_dir` (cv2.imwrite) and/or shown in a
    desktop window (cv2.imshow) when `show=True`.
    """
    print(f"[*] Running inference on: {source}")
    results = model.predict(source, conf=conf)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for idx, r in enumerate(results):
        im_array = r.plot()  # BGR numpy array with boxes drawn

        if save_dir and _HAS_CV2:
            src = getattr(r, "path", None)
            name = os.path.basename(src) if src else f"detection_{idx}.jpg"
            out_path = os.path.join(save_dir, name)
            cv2.imwrite(out_path, im_array)
            print(f"  saved annotated image -> {out_path}")

        if show and _HAS_CV2:
            cv2.imshow("YOLO-World detection", im_array)
            cv2.waitKey(0)

    if show and _HAS_CV2:
        cv2.destroyAllWindows()

    print("[*] Inference complete.")
    return results


# --------------------------------------------------------------------------- #
# Analysis helpers  (notebook cells 5, 7, 11, 12 — preserved verbatim)
# --------------------------------------------------------------------------- #
def analyze_detection_statistics(model, directory_path, target_class_name):
    """
    Performs inference on all images in a directory and calculates statistics
    for the confidence scores of a specific target class.

    Args:
        model: The loaded YOLO model.
        directory_path (str): Path to the directory containing images.
        target_class_name (str): The name of the class to analyze (e.g., 'dog', 'cat').

    Returns:
        dict: A dictionary containing the mean and standard deviation of confidence scores,
              or None if no detections for the target class were found.
    """
    print(f"\n[*] Analyzing detections in: {directory_path} for class: {target_class_name}")
    confidence_scores = []
    found_images = False

    if not os.path.isdir(directory_path):
        print(f"WARNING: Directory not found: {directory_path}")
        return None

    image_extensions = IMAGE_EXTENSIONS

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(image_extensions):
            found_images = True
            image_path = os.path.join(directory_path, filename)
            try:
                results = model.predict(image_path, conf=0.25, verbose=False)  # Set verbose to False to suppress per-image output
                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        # Get class name using model.names
                        detected_class_name = model.names[class_id]
                        if detected_class_name == target_class_name:
                            confidence_scores.append(conf)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    if not found_images:
        print(f"WARNING: No supported image files found in directory: {directory_path}")
        return None

    if confidence_scores:
        mean_conf = np.mean(confidence_scores)
        std_conf = np.std(confidence_scores)
        print(f"  Total {target_class_name} detections: {len(confidence_scores)}")
        print(f"  Mean confidence for {target_class_name}: {mean_conf:.4f}")
        print(f"  Standard deviation of confidence for {target_class_name}: {std_conf:.4f}")
        return {"mean": mean_conf, "std": std_conf, "count": len(confidence_scores)}
    else:
        print(f"  No {target_class_name} detections found in {directory_path}")
        return None


def get_confidence_scores(model, directory_path, target_class_name):
    """
    Extracts confidence scores for a target class from images in a directory.
    """
    confidence_scores = []
    if not os.path.isdir(directory_path):
        print(f"WARNING: Directory not found: {directory_path}")
        return []

    image_extensions = IMAGE_EXTENSIONS

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(image_extensions):
            image_path = os.path.join(directory_path, filename)
            try:
                results = model.predict(image_path, conf=0.25, verbose=False)
                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        detected_class_name = model.names[class_id]
                        if detected_class_name == target_class_name:
                            confidence_scores.append(conf)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
    return confidence_scores


def plot_confidence_histograms(
    model,
    cat_images_dir,
    dog_images_dir,
    save_path="confidence_histograms.png",
):
    """Plot cat vs dog confidence-score histograms. Port of cell 7.

    The notebook used `plt.show()` (inline Colab display); here the figure is
    saved to `save_path` so it works in a headless / local environment.
    """
    if not _HAS_PLT:
        print("WARNING: matplotlib not available; skipping histogram plot.")
        return

    # Get confidence scores for 'cat' and 'dog'
    cat_confidences = get_confidence_scores(model, cat_images_dir, 'cat')
    dog_confidences = get_confidence_scores(model, dog_images_dir, 'dog')

    # Plotting histograms
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot
    if cat_confidences:
        plt.hist(cat_confidences, bins=10, color='skyblue', edgecolor='black')
        plt.title('Distribution of Cat Detection Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'No cat detections to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title('Distribution of Cat Detection Confidence Scores')

    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot
    if dog_confidences:
        plt.hist(dog_confidences, bins=10, color='lightcoral', edgecolor='black')
        plt.title('Distribution of Dog Detection Confidence Scores')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
    else:
        plt.text(0.5, 0.5, 'No dog detections to plot', horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)
        plt.title('Distribution of Dog Detection Confidence Scores')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[*] Histogram saved -> {save_path}")


def analyze_no_detection_statistics(model, directory_path):
    """
    Performs inference on all images in a directory and counts how many images
    result in zero detections.

    Args:
        model: The loaded YOLO model.
        directory_path (str): Path to the directory containing images.

    Returns:
        int: The count of images with no detections, or None if the directory is not found.
    """
    print(f"\n[*] Analyzing images with no detections in: {directory_path}")
    no_detection_count = 0
    total_images_processed = 0

    if not os.path.isdir(directory_path):
        print(f"WARNING: Directory not found: {directory_path}")
        return None

    image_extensions = IMAGE_EXTENSIONS

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(image_extensions):
            total_images_processed += 1
            image_path = os.path.join(directory_path, filename)
            try:
                results = model.predict(image_path, conf=0.25, verbose=False)
                # Check if any bounding boxes were detected in the current image's results
                detected = False
                for r in results:
                    if len(r.boxes) > 0:
                        detected = True
                        break
                if not detected:
                    no_detection_count += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    print(f"  Total images processed: {total_images_processed}")
    print(f"  Images with NO detections: {no_detection_count}")
    return no_detection_count


def analyze_class_miss_detections(model, directory_path, target_class_name):
    """
    Counts images in a directory where the target class was not detected.

    Args:
        model: The loaded YOLO model.
        directory_path (str): Path to the directory containing images.
        target_class_name (str): The name of the class expected to be detected.

    Returns:
        tuple: A tuple containing (miss_detection_count, total_images_in_dir).
    """
    print(f"\n[*] Analyzing images for miss detections of '{target_class_name}' in: {directory_path}")
    miss_detection_count = 0
    total_images_in_dir = 0

    if not os.path.isdir(directory_path):
        print(f"WARNING: Directory not found: {directory_path}")
        return 0, 0  # Return 0 for count and total if directory not found

    image_extensions = IMAGE_EXTENSIONS

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(image_extensions):
            total_images_in_dir += 1
            image_path = os.path.join(directory_path, filename)
            try:
                results = model.predict(image_path, conf=0.25, verbose=False)
                detected_target_class = False
                for r in results:
                    for box in r.boxes:
                        class_id = int(box.cls[0])
                        detected_class_name = model.names[class_id]
                        if detected_class_name == target_class_name:
                            detected_target_class = True
                            break  # Found target class in this image
                    if detected_target_class:
                        break  # Found target class in this image's results

                if not detected_target_class:
                    miss_detection_count += 1
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    print(f"  Total images in directory: {total_images_in_dir}")
    print(f"  Images where '{target_class_name}' was NOT detected: {miss_detection_count}")
    return miss_detection_count, total_images_in_dir


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLO-World pet (dog/cat) vision detector (converted from notebook)"
    )
    parser.add_argument(
        "--customize", action="store_true",
        help="Build a custom open-vocabulary model from the base YOLO-World weights",
    )
    parser.add_argument(
        "--classes", type=str, default=",".join(DEFAULT_CLASSES),
        help=f"Comma-separated class vocabulary (default '{','.join(DEFAULT_CLASSES)}')",
    )
    parser.add_argument(
        "--base-model", type=str, default=DEFAULT_BASE_MODEL,
        help=f"Base YOLO-World weights (default {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--save-model", type=str, default=None,
        help="Path to save the customized model (e.g. dogandcat.pt)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to a (previously saved) model to load for predict/analyze",
    )
    parser.add_argument(
        "--predict", type=str, default=None,
        help="Image file or directory to run detection on",
    )
    parser.add_argument(
        "--save-dir", type=str, default=None,
        help="Directory to write annotated images into (used with --predict)",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="Show annotated detections in a desktop window (used with --predict)",
    )
    parser.add_argument(
        "--analyze", type=str, default=None,
        help="Directory of images to compute detection statistics over",
    )
    parser.add_argument(
        "--class", dest="target_class", type=str, default=None,
        help="Target class name for --analyze (e.g. dog)",
    )
    parser.add_argument(
        "--conf", type=float, default=DEFAULT_CONF,
        help=f"Confidence threshold (default {DEFAULT_CONF})",
    )
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]

    # Obtain a model: either build a custom one, or load an existing one.
    model = None
    if args.customize:
        model = build_custom_model(classes, args.base_model, args.save_model)
    elif args.model:
        model = load_model(args.model)

    if args.predict:
        if model is None:
            model = build_custom_model(classes, args.base_model)
        predict_path(model, args.predict, conf=args.conf,
                     save_dir=args.save_dir, show=args.show)

    if args.analyze:
        if model is None:
            model = build_custom_model(classes, args.base_model)
        if not args.target_class:
            parser.error("--analyze requires --class <name>")
        analyze_detection_statistics(model, args.analyze, args.target_class)

    if not (args.customize or args.predict or args.analyze):
        parser.print_help()


if __name__ == "__main__":
    main()
