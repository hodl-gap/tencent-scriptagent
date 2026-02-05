#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical pipeline executor for the ScriptAgent system.

Usage:
    python code/pipeline.py \\
        --dialogue "dialogue text or path to .txt file" \\
        --openai_api_key KEY \\
        --gemini_api_key KEY \\
        --output_dir ./output

Flow:
    1. ScriptAgent  - generate shooting script from dialogue
    2. DirectorAgent - parse script and generate video
    3. CriticAgent (script) - evaluate script quality
    4. CriticAgent (video)  - evaluate video quality
"""

import argparse
import json
import logging
import os

try:
    from dotenv import load_dotenv

    load_dotenv()  # loads .env from cwd (project root)
except ImportError:
    pass  # python-dotenv not installed, rely on shell env vars

LOGGER = logging.getLogger(__name__)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ScriptAgent pipeline: dialogue -> script -> video -> evaluation"
    )

    parser.add_argument(
        "--dialogue",
        type=str,
        required=True,
        help="Input dialogue text, or path to a .txt file containing the dialogue",
    )
    parser.add_argument(
        "--openai_api_key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", ""),
        help="OpenAI API key (for Sora video gen + script evaluation). "
        "Falls back to OPENAI_API_KEY env var / .env file.",
    )
    parser.add_argument(
        "--gemini_api_key",
        type=str,
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Google Gemini API key (for Veo video gen + video evaluation). "
        "Falls back to GEMINI_API_KEY env var / .env file.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./models/ScriptAgent",
        help="Path to ScriptAgent model weights (default: ./models/ScriptAgent)",
    )
    parser.add_argument(
        "--script_path",
        type=str,
        default=None,
        help="Skip ScriptAgent and use a pre-generated script file",
    )
    parser.add_argument(
        "--video_model",
        type=str,
        default="veo3.1",
        help="Video generation model name (default: veo3.1). "
        "veo3.1/veo3.1-fast need --gemini_api_key; "
        "sora2-pro/sora2 need --openai_api_key.",
    )
    parser.add_argument(
        "--style",
        type=str,
        default="anime",
        help="Visual style (default: anime)",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip evaluation steps",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="[%(asctime)s][%(levelname)s] %(message)s",
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # --- Validate API keys ---
    veo_models = {"veo3.1", "veo3.1-fast"}
    sora_models = {"sora2-pro", "sora2"}

    if args.video_model in veo_models and not args.gemini_api_key:
        parser.error(
            f"--gemini_api_key (or GEMINI_API_KEY env var) is required for {args.video_model}. "
            "Set it in .env or pass it on the command line."
        )
    if args.video_model in sora_models and not args.openai_api_key:
        parser.error(
            f"--openai_api_key (or OPENAI_API_KEY env var) is required for {args.video_model}. "
            "Set it in .env or pass it on the command line."
        )
    if not args.skip_eval:
        if not args.openai_api_key:
            parser.error(
                "--openai_api_key (or OPENAI_API_KEY env var) is required for script evaluation. "
                "Set it in .env, pass it on the command line, or use --skip_eval."
            )
        if not args.gemini_api_key:
            parser.error(
                "--gemini_api_key (or GEMINI_API_KEY env var) is required for video evaluation. "
                "Set it in .env, pass it on the command line, or use --skip_eval."
            )

    # --- Resolve dialogue input ---
    dialogue_text = args.dialogue
    if os.path.isfile(dialogue_text):
        LOGGER.info("Reading dialogue from file: %s", dialogue_text)
        with open(dialogue_text, "r", encoding="utf-8") as f:
            dialogue_text = f.read()

    # --- Step 1: ScriptAgent ---
    if args.script_path:
        LOGGER.info("Using pre-generated script: %s", args.script_path)
        with open(args.script_path, "r", encoding="utf-8") as f:
            script_text = f.read()
    else:
        LOGGER.info("Running ScriptAgent to generate script...")
        from script_agent import generate_script

        script_text = generate_script(dialogue_text, model_path=args.model_path)

    script_output = os.path.join(args.output_dir, "generated_script.txt")
    with open(script_output, "w", encoding="utf-8") as f:
        f.write(script_text)
    LOGGER.info("Script saved to %s", script_output)

    # --- Step 2: DirectorAgent ---
    LOGGER.info("Running DirectorAgent...")
    from director_agent import (
        Api,
        StoryVideoGenerator,
        extract_story_components,
        prepare_output_structure,
        process_story,
        resolve_style_key,
        slugify_model_name,
    )

    # Gemini key is shared: Veo (video gen) and Gemini (video eval) use the same key.
    api = Api(
        openai_api_key=args.openai_api_key,
        gemini_api_key=args.gemini_api_key,
    )
    style_key = resolve_style_key(args.style)

    model_dir, node_root, final_root, responses_path, model_slug = (
        prepare_output_structure(args.output_dir, args.video_model)
    )

    from director_agent import MODEL_DEFAULT_CONFIG

    model_config = MODEL_DEFAULT_CONFIG.get(args.video_model, {})
    size = model_config.get("size", "1280x720")
    seconds = model_config.get("seconds", 8)

    generator = StoryVideoGenerator(
        api=api,
        output_dir=args.output_dir,
        model=args.video_model,
        size=size,
        seconds=seconds,
        max_retry=3,
        reference_mode="first",
        style_key=style_key,
    )

    components = extract_story_components(script_text)
    style_slug = slugify_model_name(style_key)

    video_path = process_story(
        generator,
        script_text=script_text,
        model_slug=model_slug,
        story_index=1,
        node_root=node_root,
        final_root=final_root,
        responses_path=responses_path,
        components=components,
        style_slug=style_slug,
    )
    LOGGER.info("Video generated: %s", video_path)

    # --- Step 3 & 4: Evaluation ---
    results = {
        "script_path": script_output,
        "video_path": video_path,
    }

    if not args.skip_eval:
        # Script evaluation
        LOGGER.info("Running script evaluation (OpenAI GPT-4o)...")
        from critic_agent_script import ScriptEvaluator

        script_evaluator = ScriptEvaluator(api_key=args.openai_api_key)
        script_eval = script_evaluator.evaluate(
            source_dialogue=dialogue_text,
            generated_script=script_text,
        )
        results["script_evaluation"] = script_eval

        script_eval_path = os.path.join(args.output_dir, "script_evaluation.json")
        with open(script_eval_path, "w", encoding="utf-8") as f:
            json.dump(script_eval, f, indent=2, ensure_ascii=False)
        LOGGER.info("Script evaluation saved to %s", script_eval_path)

        # Video evaluation
        LOGGER.info("Running video evaluation (Gemini)...")
        from critic_agent_video import GeminiVideoEvaluator

        video_evaluator = GeminiVideoEvaluator(api_key=args.gemini_api_key)
        video_eval = video_evaluator.evaluate(
            script_text=script_text,
            video_path=video_path,
        )
        results["video_evaluation"] = video_eval

        video_eval_path = os.path.join(args.output_dir, "video_evaluation.json")
        with open(video_eval_path, "w", encoding="utf-8") as f:
            json.dump(video_eval, f, indent=2, ensure_ascii=False)
        LOGGER.info("Video evaluation saved to %s", video_eval_path)

    # Save combined results
    combined_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    LOGGER.info("Combined results saved to %s", combined_path)


if __name__ == "__main__":
    main()
