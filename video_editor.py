"""
Post-Generation Video Editing Pipeline
Frame-by-frame editing, effects, and composition.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional


@dataclass
class EditOperation:
    operation_type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_frame: int = 0
    end_frame: int = -1
    blend_mode: str = "normal"


class FrameEditor:
    """
    Applies a sequence of EditOperation objects to a video file and exports a new one.
    """

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.operations: List[EditOperation] = []

    def add_operation(self, operation: EditOperation) -> None:
        self.operations.append(operation)

    # ===== Effects Implementations =====

    @staticmethod
    def adjust_brightness(frame: np.ndarray, value: float) -> np.ndarray:
        frame = frame.astype(np.float32)
        frame = frame + float(value)
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)

    @staticmethod
    def adjust_contrast(frame: np.ndarray, value: float) -> np.ndarray:
        # value > 1 → more contrast, 0-1 → less
        frame = frame.astype(np.float32)
        mean = np.mean(frame)
        frame = (frame - mean) * float(value) + mean
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)

    @staticmethod
    def adjust_saturation(frame: np.ndarray, value: float) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] *= float(value)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return result

    @staticmethod
    def apply_grayscale(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def apply_gaussian_blur(frame: np.ndarray, kernel_size: int) -> np.ndarray:
        k = max(1, kernel_size | 1)  # force odd
        return cv2.GaussianBlur(frame, (k, k), 0)

    @staticmethod
    def apply_motion_blur(frame: np.ndarray, kernel_size: int, angle: float) -> np.ndarray:
        k = max(1, kernel_size)
        kernel = np.zeros((k, k), dtype=np.float32)
        kernel[k // 2, :] = 1.0 / k
        # rotate kernel
        center = (k // 2, k // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (k, k))
        kernel /= np.sum(kernel) if np.sum(kernel) != 0 else 1.0
        return cv2.filter2D(frame, -1, kernel)

    @staticmethod
    def apply_box_blur(frame: np.ndarray, kernel_size: int) -> np.ndarray:
        k = max(1, kernel_size | 1)
        return cv2.blur(frame, (k, k))

    @staticmethod
    def apply_vignette(frame: np.ndarray, intensity: float) -> np.ndarray:
        rows, cols = frame.shape[:2]
        # Create vignette mask
        kernel_x = cv2.getGaussianKernel(cols, cols * intensity)
        kernel_y = cv2.getGaussianKernel(rows, rows * intensity)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        vignette = np.empty_like(frame)
        for i in range(3):
            vignette[:, :, i] = frame[:, :, i] * mask
        return vignette

    @staticmethod
    def apply_color_grade(frame: np.ndarray, color: str, intensity: float) -> np.ndarray:
        overlay = np.zeros_like(frame, dtype=np.float32)
        color = color.lower()
        if color == "warm":
            overlay[:, :] = (0, 50, 80)  # BGR
        elif color == "cool":
            overlay[:, :] = (80, 50, 0)
        elif color == "cinematic":
            overlay[:, :] = (20, 40, 80)
        else:
            overlay[:, :] = (0, 0, 0)

        frame_f = frame.astype(np.float32)
        result = frame_f * (1 - intensity) + overlay * intensity
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result

    @staticmethod
    def apply_film_grain(frame: np.ndarray, intensity: float) -> np.ndarray:
        noise = np.random.randn(*frame.shape) * (255.0 * intensity)
        frame_f = frame.astype(np.float32) + noise
        frame_f = np.clip(frame_f, 0, 255)
        return frame_f.astype(np.uint8)

    # ===== Effect Router =====

    def apply_frame_edits(
        self, frame: np.ndarray, frame_number: int
    ) -> np.ndarray:
        result = frame.copy()
        for op in self.operations:
            if op.start_frame <= frame_number <= (op.end_frame if op.end_frame >= 0 else frame_number):
                t = op.operation_type.lower()
                params = op.parameters
                if t == "brightness":
                    result = self.adjust_brightness(result, params.get("value", 0))
                elif t == "contrast":
                    result = self.adjust_contrast(result, params.get("value", 1.0))
                elif t == "saturation":
                    result = self.adjust_saturation(result, params.get("value", 1.0))
                elif t == "grayscale":
                    result = self.apply_grayscale(result)
                elif t == "gaussian_blur":
                    result = self.apply_gaussian_blur(result, params.get("kernel_size", 5))
                elif t == "box_blur":
                    result = self.apply_box_blur(result, params.get("kernel_size", 5))
                elif t == "vignette":
                    result = self.apply_vignette(result, params.get("intensity", 0.5))
                elif t == "warm_grade":
                    result = self.apply_color_grade(result, "warm", params.get("intensity", 0.3))
                elif t == "cool_grade":
                    result = self.apply_color_grade(result, "cool", params.get("intensity", 0.3))
                elif t == "cinematic_grade":
                    result = self.apply_color_grade(result, "cinematic", params.get("intensity", 0.3))
                elif t == "film_grain":
                    result = self.apply_film_grain(result, params.get("intensity", 0.05))
                # ... extend with more effects
        return result

    # ===== Export Pipeline =====

    def export_edited_video(
        self,
        output_path: str,
        show_progress: bool = True,
    ) -> None:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps if fps > 0 else 30, (width, height))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            edited = self.apply_frame_edits(frame, frame_idx)
            out.write(edited)

            frame_idx += 1
            if show_progress and frame_count > 0 and frame_idx % max(1, frame_count // 10) == 0:
                pct = int(frame_idx / frame_count * 100)
                print(f"[Edit] Progress: {pct}%")

        cap.release()
        out.release()
        if show_progress:
            print(f"✓ Edited video saved to: {output_path}")


if __name__ == "__main__":
    # Simple demo (requires 'demo_basic.mp4' from animation_engine demo)
    input_video = "demo_basic.mp4"
    output_video = "demo_basic_edited.mp4"

    editor = FrameEditor(input_video)
    editor.add_operation(
        EditOperation(
            operation_type="brightness",
            parameters={"value": 20},
            start_frame=0,
            end_frame=50,
        )
    )
    editor.add_operation(
        EditOperation(
            operation_type="vignette",
            parameters={"intensity": 0.5},
            start_frame=0,
            end_frame=-1,
        )
    )
    editor.add_operation(
        EditOperation(
            operation_type="film_grain",
            parameters={"intensity": 0.03},
            start_frame=0,
            end_frame=-1,
        )
    )

    editor.export_edited_video(output_video, show_progress=True)
