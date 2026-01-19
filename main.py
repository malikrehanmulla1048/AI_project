"""
Main Entry Point for AI Animation Generation System
Combines chatbot, animation engine, and video editor.
"""

import argparse

from animation_engine import SceneConfig, AnimationController, CharacterConfig, AnimationKeyframe
from chatbot_interface import AnimationChatbot
from video_editor import FrameEditor, EditOperation


def run_demo_1():
    """Demo 1: Simple single-character animation."""
    scene = SceneConfig(name="Demo 1", width=1280, height=720)
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

    anim.render_animation(120, "demo1_basic.mp4", show_progress=True)


def run_demo_2():
    """Demo 2: Two characters interacting."""
    scene = SceneConfig(name="Demo 2", width=1280, height=720)
    anim = AnimationController(scene, fps=30)

    alice = CharacterConfig(name="alice", position=(300, 360))
    bob = CharacterConfig(name="bob", position=(900, 360))
    anim.register_character(alice)
    anim.register_character(bob)

    # Alice moves towards Bob
    anim.add_keyframe(AnimationKeyframe(0, "alice", (300, 360), expression="neutral"))
    anim.add_keyframe(AnimationKeyframe(60, "alice", (600, 360), expression="happy"))

    # Bob reacts
    anim.add_keyframe(AnimationKeyframe(0, "bob", (900, 360), expression="neutral"))
    anim.add_keyframe(AnimationKeyframe(60, "bob", (900, 360), expression="happy"))

    anim.render_animation(120, "demo2_interaction.mp4", show_progress=True)


def run_demo_3():
    """Demo 3: Apply editing effects to Demo 1 output."""
    input_video = "demo1_basic.mp4"
    output_video = "demo1_basic_edited.mp4"

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


def run_chatbot():
    """Run interactive chatbot interface."""
    bot = AnimationChatbot()
    bot.interactive_chat()


def main_menu():
    while True:
        print("\n🎬 AI Animation System")
        print("1) Interactive Chatbot")
        print("2) Demo 1 - Basic Animation")
        print("3) Demo 2 - Multi-Character Interaction")
        print("4) Demo 3 - Edit Demo 1 Video")
        print("5) About")
        print("0) Exit")
        choice = input("Select an option: ").strip()

        if choice == "1":
            run_chatbot()
        elif choice == "2":
            run_demo_1()
        elif choice == "3":
            run_demo_2()
        elif choice == "4":
            run_demo_3()
        elif choice == "5":
            print(
                "This system renders frame-by-frame animations with a chatbot interface\n"
                "and a post-processing video editor. See README.md and docs for details."
            )
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Try again.")


def main():
    parser = argparse.ArgumentParser(description="AI Animation Generation System")
    parser.add_argument("--demo", type=int, help="Run a specific demo (1, 2, or 3)")
    parser.add_argument("--chatbot", action="store_true", help="Start chatbot directly")
    parser.add_argument("--info", action="store_true", help="Show system info")
    args = parser.parse_args()

    if args.info:
        print(
            "AI Animation System\n"
            "- Frame-by-frame animation engine\n"
            "- Natural language chatbot\n"
            "- Video editor with effects\n"
            "Run without arguments for interactive menu."
        )
        return

    if args.chatbot:
        run_chatbot()
        return

    if args.demo == 1:
        run_demo_1()
        return
    if args.demo == 2:
        run_demo_2()
        return
    if args.demo == 3:
        run_demo_3()
        return

    # Default: show menu
    main_menu()


if __name__ == "__main__":
    main()
