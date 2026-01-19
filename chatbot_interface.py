"""
Natural Language Chatbot Interface for AI Animation System.
Parses simple English-like commands into animation operations.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

from animation_engine import (
    SceneConfig,
    CharacterConfig,
    AnimationKeyframe,
    AnimationController,
    FrameGenerator,
)


# =========================
# Command Parser
# =========================

@dataclass
class ParsedCommand:
    intent: str
    params: Dict[str, Any] = field(default_factory=dict)


class CommandParser:
    """
    Very lightweight regex-based parser for animation-related commands.
    This is NOT a full NLP system, but covers common patterns like:
      - create scene
      - add character named John
      - move John to 640, 360
      - John looks happy
      - render animation
    """

    def __init__(self):
        self.patterns = [
            # Scene creation
            (r"^\s*create scene(?: called (?P<name>.+))?\s*$", "create_scene"),
            # Add character
            (
                r"^\s*add character named (?P<name>[A-Za-z0-9_]+)(?: at (?P<x>\d+)\s*,\s*(?P<y>\d+))?\s*$",
                "add_character",
            ),
            # Set position
            (
                r"^\s*move (?P<name>[A-Za-z0-9_]+) to (?P<x>\d+)\s*,\s*(?P<y>\d+)\s*$",
                "set_position",
            ),
            # Expression
            (
                r"^\s*(?P<name>[A-Za-z0-9_]+) looks (?P<expression>[A-Za-z_]+)\s*$",
                "set_expression",
            ),
            # Action
            (
                r"^\s*(?P<name>[A-Za-z0-9_]+) is (?P<action>[A-Za-z_]+)\s*$",
                "set_action",
            ),
            # Generic keyframe at frame
            (
                r"^\s*keyframe (?P<name>[A-Za-z0-9_]+) at frame (?P<frame>\d+)\s*$",
                "add_keyframe",
            ),
            # Duration / total frames
            (
                r"^\s*set total frames to (?P<frames>\d+)\s*$",
                "set_duration",
            ),
            # FPS
            (
                r"^\s*set fps to (?P<fps>\d+)\s*$",
                "set_speed",
            ),
            # Background
            (
                r"^\s*set background to (?P<path>.+)\s*$",
                "set_background",
            ),
            # Effects
            (
                r"^\s*apply effect (?P<effect>[A-Za-z0-9_]+)\s*$",
                "set_animation_effect",
            ),
            # Render
            (r"^\s*render(?: animation)?(?: to (?P<path>.+))?\s*$", "render"),
            # Reset
            (r"^\s*clear project\s*$", "clear"),
            # Help
            (r"^\s*help\s*$", "help"),
            # Info
            (r"^\s*status\s*$", "status"),
        ]

    def parse_command(self, text: str) -> Optional[ParsedCommand]:
        text = text.strip()
        if not text:
            return None
        for pattern, intent in self.patterns:
            m = re.match(pattern, text, re.IGNORECASE)
            if m:
                return ParsedCommand(intent=intent, params={k: v for k, v in m.groupdict().items() if v is not None})
        # Fallback: unknown
        return ParsedCommand(intent="unknown", params={"raw": text})


# =========================
# Animation State Manager
# =========================

class AnimationStateManager:
    """Holds the current scene, characters, keyframes, etc."""

    def __init__(self):
        self.scene_config: Optional[SceneConfig] = None
        self.controller: Optional[AnimationController] = None
        self.total_frames: int = 120  # default
        self.characters: Dict[str, CharacterConfig] = {}

    def create_scene(self, name: Optional[str] = None) -> str:
        if not name:
            name = "Untitled Scene"
        self.scene_config = SceneConfig(name=name, width=1280, height=720)
        self.controller = AnimationController(self.scene_config, fps=30)
        self.characters.clear()
        return f"Scene '{name}' created (1280x720)"

    def add_character(
        self,
        name: str,
        image_path: Optional[str] = None,
        position: Optional[tuple] = None,
    ) -> str:
        if not self.controller:
            return "No scene yet. Use 'create scene' first."
        if name in self.characters:
            return f"Character '{name}' already exists."
        if position is None:
            position = (100, 100)
        cfg = CharacterConfig(name=name, image_path=image_path, position=position)
        self.characters[name] = cfg
        self.controller.register_character(cfg)
        return f"Character '{name}' added at position {position}"

    def set_position(self, name: str, x: int, y: int) -> str:
        if name not in self.characters:
            return f"No character named '{name}'."
        pos = (int(x), int(y))
        # Add a keyframe at next frame or last?
        frame = self.total_frames // 2
        kf = AnimationKeyframe(
            frame_number=frame,
            character_name=name,
            position=pos,
        )
        if self.controller:
            self.controller.add_keyframe(kf)
        return f"Keyframe added: {name} at frame {frame} → position {pos}"

    def set_expression(self, name: str, expression: str) -> str:
        if name not in self.characters:
            return f"No character named '{name}'."
        frame = self.total_frames // 2
        cfg = self.characters[name]
        kf = AnimationKeyframe(
            frame_number=frame,
            character_name=name,
            position=cfg.position,
            expression=expression,
        )
        if self.controller:
            self.controller.add_keyframe(kf)
        return f"Character '{name}' will expression='{expression}' at frame {frame}"

    def set_action(self, name: str, action: str) -> str:
        if name not in self.characters:
            return f"No character named '{name}'."
        frame = self.total_frames // 2
        cfg = self.characters[name]
        kf = AnimationKeyframe(
            frame_number=frame,
            character_name=name,
            position=cfg.position,
            action=action,
        )
        if self.controller:
            self.controller.add_keyframe(kf)
        return f"Character '{name}' will action='{action}' at frame {frame}"

    def add_keyframe(
        self,
        name: str,
        frame: int,
    ) -> str:
        if name not in self.characters:
            return f"No character named '{name}'."
        cfg = self.characters[name]
        kf = AnimationKeyframe(
            frame_number=int(frame),
            character_name=name,
            position=cfg.position,
            expression="neutral",
        )
        if self.controller:
            self.controller.add_keyframe(kf)
        return f"Keyframe added for '{name}' at frame {frame}"

    def set_duration(self, frames: int) -> str:
        self.total_frames = int(frames)
        return f"Total frames set to {self.total_frames}"

    def set_speed(self, fps: int) -> str:
        if self.controller:
            self.controller.fps = int(fps)
            return f"FPS set to {fps}"
        return "No scene yet. Use 'create scene' first."

    def set_background(self, path: str) -> str:
        if not self.scene_config:
            return "No scene yet. Use 'create scene' first."
        self.scene_config.background_path = path.strip()
        # re-init FrameGenerator with new background
        if self.controller:
            self.controller.frame_gen = FrameGenerator(self.scene_config)
        return f"Background set to '{path.strip()}'"

    def set_animation_effect(self, effect: str) -> str:
        if not self.scene_config:
            return "No scene yet. Use 'create scene' first."
        if effect not in self.scene_config.effects:
            self.scene_config.effects.append(effect)
        return f"Effect '{effect}' added to scene effects list."

    def render(self, output_path: Optional[str] = None) -> str:
        if not self.controller:
            return "No scene yet. Use 'create scene' first."
        if output_path is None:
            output_path = "output_animation.mp4"
        self.controller.render_animation(
            total_frames=self.total_frames,
            output_video_path=output_path,
            show_progress=True,
        )
        return f"Animation rendered to {output_path}"

    def clear_project(self) -> str:
        self.scene_config = None
        self.controller = None
        self.characters.clear()
        self.total_frames = 120
        return "Project cleared."

    def get_status(self) -> str:
        if not self.scene_config:
            return "No scene created yet."
        s = [
            f"Scene: {self.scene_config.name} ({self.scene_config.width}x{self.scene_config.height})",
            f"Total frames: {self.total_frames}",
            f"Characters: {', '.join(self.characters.keys()) if self.characters else 'None'}",
        ]
        return "\n".join(s)


# =========================
# Chatbot Orchestrator
# =========================

class AnimationChatbot:
    """Simple text-based chatbot for controlling the animation system."""

    def __init__(self):
        self.parser = CommandParser()
        self.state = AnimationStateManager()

    def process_input(self, user_input: str) -> str:
        parsed = self.parser.parse_command(user_input)
        if not parsed:
            return "I didn't understand. Type 'help' for commands."

        intent = parsed.intent
        p = parsed.params

        if intent == "create_scene":
            return self.state.create_scene(name=p.get("name"))

        if intent == "add_character":
            name = p.get("name")
            x = p.get("x")
            y = p.get("y")
            pos = (int(x), int(y)) if x and y else None
            return self.state.add_character(name=name, position=pos)

        if intent == "set_position":
            return self.state.set_position(
                name=p["name"], x=int(p["x"]), y=int(p["y"])
            )

        if intent == "set_expression":
            return self.state.set_expression(
                name=p["name"], expression=p["expression"]
            )

        if intent == "set_action":
            return self.state.set_action(name=p["name"], action=p["action"])

        if intent == "add_keyframe":
            return self.state.add_keyframe(
                name=p["name"], frame=int(p["frame"])
            )

        if intent == "set_duration":
            return self.state.set_duration(frames=int(p["frames"]))

        if intent == "set_speed":
            return self.state.set_speed(fps=int(p["fps"]))

        if intent == "set_background":
            return self.state.set_background(path=p["path"])

        if intent == "set_animation_effect":
            return self.state.set_animation_effect(effect=p["effect"])

        if intent == "render":
            return self.state.render(output_path=p.get("path"))

        if intent == "clear":
            return self.state.clear_project()

        if intent == "help":
            return self._help_text()

        if intent == "status":
            return self.state.get_status()

        if intent == "unknown":
            return f"Unknown command: '{p.get('raw', '')}'. Type 'help' for examples."

        return "Unhandled intent."

    def _help_text(self) -> str:
        return (
            "Supported commands:\n"
            "  - create scene\n"
            "  - add character named John\n"
            "  - add character named John at 200, 300\n"
            "  - move John to 640, 360\n"
            "  - John looks happy\n"
            "  - John is walking\n"
            "  - keyframe John at frame 60\n"
            "  - set total frames to 120\n"
            "  - set fps to 24\n"
            "  - set background to path/to/image.png\n"
            "  - apply effect vignette\n"
            "  - render animation\n"
            "  - clear project\n"
            "  - status\n"
            "  - help\n"
        )

    def interactive_chat(self) -> None:
        print("🎬 Animation Chatbot - type 'help' for commands, 'quit' to exit.")
        while True:
            try:
                user_input = input("> ")
            except EOFError:
                break
            if user_input.strip().lower() in {"quit", "exit"}:
                print("Goodbye!")
                break
            response = self.process_input(user_input)
            print(response)


if __name__ == "__main__":
    bot = AnimationChatbot()
    bot.interactive_chat()
