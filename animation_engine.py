"""
AI-Powered Animated Video Generation Engine
Frame-by-frame animation rendering with character support, keyframes,
interpolation, basic effects, and video composition via OpenCV/FFmpeg.
"""

import os
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Optional, Any


# =========================
# Data Classes
# =========================

@dataclass
class CharacterConfig:
    """Configuration for a character in the scene."""
    name: str
    image_path: Optional[str] = None
    position: Tuple[int, int] = (0, 0)
    scale: float = 1.0
    rotation: float = 0.0
    opacity: float = 1.0
    expressions: Dict[str, float] = field(default_factory=dict)


@dataclass
class SceneConfig:
    """Configuration for the scene/background."""
    name: str = "Untitled Scene"
    background_path: Optional[str] = None
    width: int = 1280
    height: int = 720
    lighting: float = 1.0
    effects: List[str] = field(default_factory=list)


@dataclass
class AnimationKeyframe:
    """A single keyframe describing a character state at a given frame."""
    frame_number: int
    character_name: str
    position: Tuple[int, int]
    rotation: float = 0.0
    scale: float = 1.0
    expression: str = "neutral"
    action: str = "idle"
    duration: float = 0.0  # seconds or relative notion


@dataclass
class EditOperation:
    """Post-process video edit operation."""
    operation_type: str
    parameters: Dict[str, Any]
    start_frame: int = 0
    end_frame: int = -1
    blend_mode: str = "normal"


# =========================
# Interpolation / Utilities
# =========================

class KeyframeInterpolator:
    """Handles interpolation between keyframes."""

    @staticmethod
    def linear_interpolate(start: float, end: float, progress: float) -> float:
        """Linear interpolation between two float values."""
        return start + (end - start) * progress

    @staticmethod
    def ease_in_out(progress: float) -> float:
        """Simple ease-in-out (smoothstep)."""
        return progress * progress * (3 - 2 * progress)

    @staticmethod
    def interpolate_position(
        start: Tuple[int, int], end: Tuple[int, int], progress: float
    ) -> Tuple[int, int]:
        """Interpolate between two 2D positions."""
        x = KeyframeInterpolator.linear_interpolate(float(start[0]), float(end[0]), progress)
        y = KeyframeInterpolator.linear_interpolate(float(start[1]), float(end[1]), progress)
        return int(x), int(y)

    @staticmethod
    def interpolate_state(
        state1: Dict[str, Any],
        state2: Dict[str, Any],
        progress: float,
    ) -> Dict[str, Any]:
        """Interpolate between two arbitrary state dictionaries."""
        result = {}
        for key in state1.keys() | state2.keys():
            v1 = state1.get(key)
            v2 = state2.get(key)
            
            # Handle numeric types
            if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
                result[key] = KeyframeInterpolator.linear_interpolate(float(v1), float(v2), progress)
            # Handle 2D tuples (positions)
            elif isinstance(v1, tuple) and isinstance(v2, tuple) and len(v1) == 2 and len(v2) == 2:
                result[key] = KeyframeInterpolator.interpolate_position(v1, v2, progress)
            else:
                # Non-numeric or mismatch → pick based on progress
                result[key] = v2 if progress >= 0.5 else v1
        return result



# =========================
# Frame Generator
# =========================

class FrameGenerator:
    """
    Generates individual frames given scene config and character states.
    """

    def __init__(self, scene_config: SceneConfig):
        self.scene_config = scene_config
        self.characters: Dict[str, CharacterConfig] = {}
        self._background = self._load_background()

    def register_character(self, character: CharacterConfig) -> None:
        self.characters[character.name] = character

    def _load_background(self) -> np.ndarray:
        if self.scene_config.background_path and os.path.exists(
            self.scene_config.background_path
        ):
            img = cv2.imread(self.scene_config.background_path, cv2.IMREAD_COLOR)
            if img is None:
                bg = np.zeros(
                    (self.scene_config.height, self.scene_config.width, 3),
                    dtype=np.uint8,
                )
            else:
                bg = cv2.resize(
                    img, (self.scene_config.width, self.scene_config.height)
                )
        else:
            bg = np.zeros(
                (self.scene_config.height, self.scene_config.width, 3),
                dtype=np.uint8,
            )
        if self.scene_config.lighting != 1.0:
            bg = self.apply_lighting(bg, self.scene_config.lighting)
        return bg

    @staticmethod
    def apply_lighting(frame: np.ndarray, lighting: float) -> np.ndarray:
        frame = frame.astype(np.float32) * float(lighting)
        frame = np.clip(frame, 0, 255)
        return frame.astype(np.uint8)

    def _load_character_image(self, character: CharacterConfig) -> Optional[np.ndarray]:
        if not character.image_path:
            return None
        if not os.path.exists(character.image_path):
            return None
        img = cv2.imread(character.image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        return img

    def _transform_character(
        self, img: np.ndarray, scale: float, rotation: float
    ) -> np.ndarray:
        # scale
        h, w = img.shape[:2]
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w <= 0 or new_h <= 0:
            return img
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        # rotate around center
        center = (new_w // 2, new_h // 2)
        M = cv2.getRotationMatrix2D(center, rotation, 1.0)
        rotated = cv2.warpAffine(
            img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT
        )
        return rotated

    def _blend_rgba(
        self, base: np.ndarray, overlay: np.ndarray, x: int, y: int, opacity: float
    ) -> np.ndarray:
        """
        Blend RGBA (or RGB) overlay into BGR base at (x, y) with given opacity.
        If overlay has alpha channel, combine with opacity.
        """
        oh, ow = overlay.shape[:2]
        bh, bw = base.shape[:2]

        if x >= bw or y >= bh:
            return base

        x_end = min(x + ow, bw)
        y_end = min(y + oh, bh)
        overlay_region = overlay[0 : y_end - y, 0 : x_end - x]

        if overlay_region.shape == 4:
            # overlay is BGRA or RGBA; assume OpenCV -> BGRA
            overlay_rgb = overlay_region[:, :, :3].astype(np.float32)
            alpha_channel = overlay_region[:, :, 3].astype(np.float32) / 255.0
        else:
            overlay_rgb = overlay_region.astype(np.float32)
            alpha_channel = np.ones((overlay_region.shape, overlay_region.shape), dtype=np.float32)

        alpha = np.clip(alpha_channel * float(opacity), 0.0, 1.0)

        base_region = base[y:y_end, x:x_end].astype(np.float32)

        for c in range(3):
            base_region[:, :, c] = (
                overlay_rgb[:, :, c] * alpha + base_region[:, :, c] * (1 - alpha)
            )

        base[y:y_end, x:x_end] = np.clip(base_region, 0, 255).astype(np.uint8)
        return base

    def generate_frame(
        self, frame_number: int, keyframe_state: Dict[str, Dict[str, Any]]
    ) -> np.ndarray:
        """
        Generate a single frame based on keyframe_state:
        keyframe_state: {
          'char_name': {
             'position': (x, y),
             'scale': float,
             'rotation': float,
             'opacity': float,
             'expression': 'happy'  # not fully implemented visually
          },
          ...
        }
        """
        frame = self._background.copy()

        # For each character, draw them with given properties
        for char_name, state in keyframe_state.items():
            cfg = self.characters.get(char_name)
            if not cfg:
                continue
            img = self._load_character_image(cfg)
            if img is None:
                # no sprite; optionally draw a simple circle
                pos = state.get("position", cfg.position)
                cv2.circle(frame, pos, 30, (0, 255, 0), -1)
                cv2.putText(
                    frame,
                    char_name,
                    (int(pos[0] - 20), int(pos[1] - 40)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                continue

            scale = float(state.get("scale", cfg.scale))
            rotation = float(state.get("rotation", cfg.rotation))
            opacity = float(state.get("opacity", cfg.opacity))

            transformed = self._transform_character(img, scale, rotation)
            pos = state.get("position", cfg.position)
            x = int(pos - transformed.shape // 2)
            y = int(pos - transformed.shape // 2)

            frame = self._blend_rgba(frame, transformed, x, y, opacity)

        return frame


# =========================
# Animation Controller
# =========================

class AnimationController:
    """High-level controller that manages keyframes and rendering."""

    def __init__(self, scene_config: SceneConfig, fps: int = 30):
        self.scene_config = scene_config
        self.fps = fps
        self.frame_gen = FrameGenerator(scene_config)
        # keyframes[character_name] = list[AnimationKeyframe]
        self.keyframes: Dict[str, List[AnimationKeyframe]] = {}
        self.characters: Dict[str, CharacterConfig] = {}
        self.frame_buffer: List[np.ndarray] = []

    def register_character(self, character: CharacterConfig) -> None:
        self.characters[character.name] = character
        self.frame_gen.register_character(character)

    def add_keyframe(self, keyframe: AnimationKeyframe) -> None:
        if keyframe.character_name not in self.keyframes:
            self.keyframes[keyframe.character_name] = []
        self.keyframes[keyframe.character_name].append(keyframe)
        # Ensure keyframes are sorted by frame_number
        self.keyframes[keyframe.character_name].sort(key=lambda kf: kf.frame_number)

    def _get_neighbor_keyframes(
        self, character_name: str, frame_number: int
    ) -> Tuple[Optional[AnimationKeyframe], Optional[AnimationKeyframe]]:
        char_kfs = self.keyframes.get(character_name, [])
        if not char_kfs:
            return None, None

        prev_kf = None
        next_kf = None
        for kf in char_kfs:
            if kf.frame_number <= frame_number:
                prev_kf = kf
            if kf.frame_number >= frame_number and next_kf is None:
                next_kf = kf
        return prev_kf, next_kf

    def get_interpolated_state(self, frame_number: int) -> Dict[str, Dict[str, Any]]:
        """
        Compute the character states for the given frame_number
        based on keyframes and interpolation.
        Returns a dict of character_name -> state dict.
        """
        result: Dict[str, Dict[str, Any]] = {}

        for char_name, cfg in self.characters.items():
            prev_kf, next_kf = self._get_neighbor_keyframes(char_name, frame_number)

            if prev_kf is None and next_kf is None:
                # No keyframes → default config state
                result[char_name] = {
                    "position": cfg.position,
                    "rotation": cfg.rotation,
                    "scale": cfg.scale,
                    "expression": "neutral",
                    "opacity": cfg.opacity,
                }
                continue

            if prev_kf is not None and next_kf is None:
                # Use prev keyframe state
                result[char_name] = {
                    "position": prev_kf.position,
                    "rotation": prev_kf.rotation,
                    "scale": prev_kf.scale,
                    "expression": prev_kf.expression,
                    "opacity": cfg.opacity,
                }
                continue

            if prev_kf is None and next_kf is not None:
                # Use next kf as constant until that frame
                result[char_name] = {
                    "position": next_kf.position,
                    "rotation": next_kf.rotation,
                    "scale": next_kf.scale,
                    "expression": next_kf.expression,
                    "opacity": cfg.opacity,
                }
                continue

            # Interpolate between prev_kf and next_kf
            if prev_kf.frame_number == next_kf.frame_number:
                progress = 0.0
            else:
                progress = (frame_number - prev_kf.frame_number) / float(
                    next_kf.frame_number - prev_kf.frame_number
                )
                progress = np.clip(progress, 0.0, 1.0)

            progress_eased = KeyframeInterpolator.ease_in_out(progress)

            state1 = {
                "position": prev_kf.position,
                "rotation": prev_kf.rotation,
                "scale": prev_kf.scale,
                "expression": prev_kf.expression,
                "opacity": cfg.opacity,
            }
            state2 = {
                "position": next_kf.position,
                "rotation": next_kf.rotation,
                "scale": next_kf.scale,
                "expression": next_kf.expression,
                "opacity": cfg.opacity,
            }

            interp = KeyframeInterpolator.interpolate_state(
                state1, state2, progress_eased
            )
            result[char_name] = interp

        return result

    def render_animation(
        self,
        total_frames: int,
        output_video_path: str,
        show_progress: bool = True,
        temp_frame_dir: str = "frames_temp",
    ) -> None:
        """
        Render the animation to an MP4 video using OpenCV VideoWriter.
        """
        os.makedirs(temp_frame_dir, exist_ok=True)
        self.frame_buffer.clear()

        for frame_idx in range(total_frames):
            state = self.get_interpolated_state(frame_idx)
            frame = self.frame_gen.generate_frame(frame_idx, state)
            self.frame_buffer.append(frame)

            if show_progress and (frame_idx + 1) % max(1, total_frames // 10) == 0:
                pct = int((frame_idx + 1) / total_frames * 100)
                print(f"[Render] Progress: {pct}%")

        # Write to video
        height, width = self.scene_config.height, self.scene_config.width
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, self.fps, (width, height))

        for idx, frame in enumerate(self.frame_buffer):
            out.write(frame)
        out.release()

        if show_progress:
            print(f"✓ Animation complete: {output_video_path}")


# =========================
# Video Composer (if frames on disk)
# =========================

class VideoComposer:
    """
    Compose a video from saved frames on disk.
    Useful if you stream frames instead of holding them in memory.
    """

    def __init__(self, width: int, height: int, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps

    def compose_video(self, frame_dir: str, output_path: str) -> None:
        frames = sorted(
            [
                os.path.join(frame_dir, f)
                for f in os.listdir(frame_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg"))
            ]
        )
        if not frames:
            raise ValueError(f"No frames found in directory: {frame_dir}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        for i, fpath in enumerate(frames):
            img = cv2.imread(fpath, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = cv2.resize(img, (self.width, self.height))
            out.write(img)
        out.release()
        print(f"✓ Composed video from frames in {frame_dir} → {output_path}")


# =========================
# Simple Demo
# =========================

def _demo_basic():
    scene = SceneConfig(name="Demo Scene", width=1280, height=720)
    anim = AnimationController(scene, fps=30)

    char = CharacterConfig(name="actor", position=(640, 360))
    anim.register_character(char)

    anim.add_keyframe(
        AnimationKeyframe(
            frame_number=0,
            character_name="actor",
            position=(640, 360),
            expression="neutral",
        )
    )
    anim.add_keyframe(
        AnimationKeyframe(
            frame_number=60,
            character_name="actor",
            position=(640, 500),
            expression="happy",
        )
    )

    anim.render_animation(120, "demo_basic.mp4", show_progress=True)


if __name__ == "__main__":
    _demo_basic()
