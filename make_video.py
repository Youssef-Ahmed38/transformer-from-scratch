"""Render the narrated explainer video from the project's real outputs.

Needs ffmpeg, found on PATH or through --ffmpeg. Narration uses the offline Windows voice.
"""
import argparse
import json
import os
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path

import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from tfs.model import Config, Transformer
from tfs.tokenizer import BPETokenizer

W, H, FPS = 1280, 720, 24
FADE = 8
BG, INK, MUTED, LINE = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
BLUE, ORANGE, CHIP = "#2a78d6", "#eb6834", "#f1efe8"
TERMINAL, TERMINAL_INK = "#1a1a19", "#f0efec"
RESULTS = Path("results")


def font(size, kind="regular"):
    names = {
        "regular": ("segoeui.ttf", "DejaVuSans.ttf"),
        "bold": ("segoeuib.ttf", "DejaVuSans-Bold.ttf"),
        "mono": ("consola.ttf", "DejaVuSansMono.ttf"),
    }[kind]
    try:
        return ImageFont.truetype(names[0], size)
    except OSError:
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf" / names[1]), size)


F = {
    "hero": font(64, "bold"),
    "title": font(38, "bold"),
    "big": font(76, "bold"),
    "body": font(26),
    "bold": font(26, "bold"),
    "small": font(21),
    "caption": font(27),
    "mono": font(29, "mono"),
    "mono_small": font(23, "mono"),
}


def wrap(draw, text, face, width):
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=face) > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return [*lines, line]


def mix(a, b, amount):
    a, b = (np.array(ImageColorRGB(c)) for c in (a, b))
    return tuple(int(v) for v in a + (b - a) * amount)


def ImageColorRGB(color):
    return tuple(int(color[i : i + 2], 16) for i in (1, 3, 5))


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


def tokens_scene(ids, tokens):
    def draw(d, t):
        x, y = 90, 215
        shown = max(1, int(len(tokens) * ease(t / 0.7)))
        for i, (token, token_id) in enumerate(zip(tokens[:shown], ids)):
            text = token.replace(" ", "·")
            width = max(d.textlength(text, font=F["mono"]), d.textlength(str(token_id), font=F["mono_small"])) + 26
            if x + width > W - 80:
                x, y = 90, y + 150
            d.rounded_rectangle((x, y, x + width, y + 58), 8, fill=CHIP, outline=MUTED)
            d.text((x + width / 2, y + 29), text, font=F["mono"], fill=INK, anchor="mm")
            d.text((x + width / 2, y + 86), str(token_id), font=F["mono_small"], fill=BLUE, anchor="mm")
            x += width + 12

    return draw


def architecture_scene():
    stages = ["Token IDs", "Embeddings", "6 blocks", "Output", "Scores"]
    block = ["Layer norm", "Masked attention", "Add", "Layer norm", "Feed-forward", "Add"]

    def draw(d, t):
        active = min(int(t * 5.5), 4)
        for i, name in enumerate(stages):
            x = 60 + i * 238
            on = i == active
            d.rounded_rectangle((x, 170, x + 205, 245), 10, fill=BLUE if on else CHIP, outline=MUTED)
            d.text((x + 102, 207), name, font=F["bold"], fill="#ffffff" if on else INK, anchor="mm")
            if i < 4:
                d.line((x + 205, 207, x + 233, 207), fill=MUTED, width=2)
        reveal = ease((t - 0.3) / 0.35)
        if reveal > 0:
            d.line((640, 245, 640, 345), fill=mix(BG, MUTED, reveal), width=2)
            d.text((640, 325), "Inside each block", font=F["small"], fill=mix(BG, MUTED, reveal), anchor="mm")
            highlight = int((t - 0.45) / 0.5 * 6) if t > 0.45 else -1
            for i, name in enumerate(block):
                x = 42 + i * 203
                on = i == highlight
                color = mix(BG, BLUE if on else CHIP, reveal)
                d.rounded_rectangle((x, 370, x + 184, 455), 10, fill=color, outline=mix(BG, MUTED, reveal))
                ink = "#ffffff" if on else mix(BG, INK, reveal)
                d.text((x + 92, 412), name, font=F["small"], fill=ink, anchor="mm")
            d.text((640, 515), "384 numbers per token, 6 heads, 12.31M parameters", font=F["body"], fill=mix(BG, MUTED, reveal), anchor="mm")

    return draw


def attention_scene(demo, queries):
    tokens, weights = demo["tokens"], demo["layers"][-1]
    cell = F["mono"].getlength(" ")
    spots = []
    x, y = 0, 0
    for token in tokens:
        boxes = []
        for n, part in enumerate(token.split("\n")):
            if n:
                x, y = 0, y + 1
            indent = len(part) - len(part.lstrip(" ")) if n else 0
            if part.strip():
                boxes.append((x + indent, y, part[indent:]))
            x += len(part)
        spots.append(boxes)

    def draw(d, t):
        query = queries[min(int(t * len(queries)), len(queries) - 1)]
        row = weights[query]
        peak = max(row[: query + 1])
        for i, boxes in enumerate(spots):
            for cx, cy, text in boxes:
                left, top = 70 + cx * cell, 165 + cy * 62
                right = left + len(text) * cell
                if i == query:
                    d.rounded_rectangle((left - 3, top - 4, right + 3, top + 46), 6, fill=ORANGE)
                    ink = "#ffffff"
                elif i < query:
                    d.rounded_rectangle((left - 3, top - 4, right + 3, top + 46), 6, fill=mix(BG, BLUE, 0.85 * row[i] / peak))
                    ink = INK if row[i] / peak < 0.6 else "#ffffff"
                else:
                    ink = "#c9c8c2"
                d.text((left, top), text, font=F["mono"], fill=ink)
        d.text((850, 165), "Strongest attention", font=F["bold"], fill=INK)
        top = sorted(range(query + 1), key=lambda i: -row[i])[:4]
        for n, i in enumerate(top):
            y = 218 + n * 62
            label = tokens[i].strip() or "newline"
            d.text((850, y), label[:12], font=F["mono_small"], fill=INK)
            d.rounded_rectangle((1010, y + 3, 1010 + 120, y + 23), 5, fill=CHIP)
            d.rounded_rectangle((1010, y + 3, 1010 + max(8, 120 * row[i] / peak), y + 23), 5, fill=BLUE)
            d.text((1145, y), f"{row[i] * 100:.0f}%", font=F["small"], fill=MUTED)

    return draw


def training_scene(rows, bigram):
    rows = [r for r in rows if r["step"] >= 500]
    left, right, top, bottom = 120, 1150, 150, 540
    px = lambda step: left + (step - 500) / 7500 * (right - left)
    py = lambda loss: bottom - loss / 4.4 * (bottom - top)

    def draw(d, t):
        for loss in range(5):
            d.line((left, py(loss), right, py(loss)), fill=LINE, width=1)
            d.text((left - 18, py(loss)), str(loss), font=F["small"], fill=MUTED, anchor="rm")
        d.line((left, py(bigram), right, py(bigram)), fill=MUTED, width=2)
        d.text((right, py(bigram) - 16), "Bigram baseline 4.06", font=F["small"], fill=MUTED, anchor="rm")
        shown = max(2, int(len(rows) * ease(t / 0.75)))
        for key, color in (("train_loss", BLUE), ("val_loss", ORANGE)):
            d.line([(px(r["step"]), py(r[key])) for r in rows[:shown]], fill=color, width=4, joint="curve")
        end = rows[shown - 1]
        d.text((px(end["step"]) + 8, py(end["train_loss"]) + 18), "Train", font=F["bold"], fill=BLUE)
        d.text((px(end["step"]) + 8, py(end["val_loss"]) - 42), "Unseen files", font=F["bold"], fill=ORANGE)
        d.text(((left + right) / 2, 575), "Training step, from 500 to 8,000", font=F["small"], fill=MUTED, anchor="mm")

    return draw


def speed_scene(rows):
    def draw(d, t):
        peak = max(v for _, v in rows)
        for i, (name, value) in enumerate(rows):
            y = 165 + i * 66
            share = ease(t / 0.6 - i * 0.08)
            d.text((390, y + 19), name, font=F["body"], fill=INK, anchor="rm")
            d.rounded_rectangle((410, y, 410 + max(10, 690 * value / peak * share), y + 40), 5, fill=BLUE)
            if share > 0.9:
                d.text((424 + 690 * value / peak, y + 19), f"{value:.2f}", font=F["bold"], fill=INK, anchor="lm")
        d.text((640, 575), "Training steps per second, median of 3 runs", font=F["small"], fill=MUTED, anchor="mm")

    return draw


def sample_scene(text):
    lines = text.splitlines()[:13]

    def draw(d, t):
        d.rounded_rectangle((70, 135, W - 70, 590), 12, fill=TERMINAL)
        budget = int(sum(len(line) + 1 for line in lines) * ease(t / 0.85))
        y = 158
        for line in lines:
            part = line[: budget]
            budget -= len(line) + 1
            if budget < -len(line):
                break
            d.text((100, y), part, font=F["mono_small"], fill=TERMINAL_INK)
            y += 32

    return draw


def early_exit_scene(saved, swapped):
    cards = [(f"{saved:.1f}%", "Compute saved"), (f"{swapped:.1f}%", "Saved with data halves swapped"), ("Stop", "Decision set before the test")]

    def draw(d, t):
        for i, (value, label) in enumerate(cards):
            amount = ease(t / 0.45 - i * 0.5)
            x = 70 + i * 390
            d.rounded_rectangle((x, 190, x + 360, 450), 14, fill=mix(BG, CHIP, amount), outline=mix(BG, MUTED, amount))
            d.text((x + 180, 285), value, font=F["big"], fill=mix(BG, ORANGE if value == "Stop" else INK, amount), anchor="mm")
            for n, line in enumerate(wrap(d, label, F["body"], 310)):
                d.text((x + 180, 372 + n * 34), line, font=F["body"], fill=mix(BG, MUTED, amount), anchor="mm")

    return draw


def terminal_scene(lines):
    def draw(d, t):
        d.rounded_rectangle((70, 135, W - 70, 590), 12, fill=TERMINAL)
        budget = int(sum(len(c) + 8 for c, _ in lines) * ease(t / 0.9))
        y = 165
        for command, comment in lines:
            d.text((100, y), comment, font=F["mono_small"], fill="#9c9a92")
            y += 34
            d.text((100, y), "$ " + command[: max(0, budget)], font=F["mono_small"], fill=TERMINAL_INK)
            budget -= len(command) + 8
            y += 60

    return draw


def title_scene(title, subtitle):
    def draw(d, t):
        d.text((W / 2, 300), title, font=F["hero"], fill=INK, anchor="mm")
        d.text((W / 2, 390), subtitle, font=F["body"], fill=MUTED, anchor="mm")

    return draw


def real_sample():
    path = RESULTS / "sample.txt"
    if not path.exists():
        torch.manual_seed(0)
        state = torch.load("checkpoints/run/best.pt", map_location="cpu", weights_only=False)
        model = Transformer(Config(**state["config"])).eval()
        model.load_state_dict(state["model"])
        tokenizer = BPETokenizer.load("data/tokenizer.json")
        start = torch.tensor([tokenizer.encode("def read_config(path):")])
        path.write_text(tokenizer.decode(model.generate(start, 120, 0.8, 50)[0].tolist()), encoding="utf-8")
    return path.read_text(encoding="utf-8")


def build_scenes():
    demo = json.loads((RESULTS / "attention_demo.json").read_text(encoding="utf-8"))
    bench = json.loads((RESULTS / "benchmark.json").read_text(encoding="utf-8"))
    meta = json.loads(Path("data/meta.json").read_text(encoding="utf-8"))
    log = [json.loads(line) for line in Path("checkpoints/run/log.jsonl").read_text(encoding="utf-8").splitlines()]
    early = json.loads((RESULTS / "early_exit.json").read_text(encoding="utf-8"))
    swapped = json.loads((RESULTS / "early_exit_swapped.json").read_text(encoding="utf-8"))
    tokenizer = BPETokenizer.load("data/tokenizer.json")

    ids = tokenizer.encode("def add_item(items, value):")
    tokens = [tokenizer.decode([i]) for i in ids]
    last = len(demo["tokens"]) - 1
    speed = [("Baseline", bench["training"][0]["steps_per_sec"]), ("All changes on", bench["training"][-1]["steps_per_sec"])]
    speed += [(r["stage"].replace("without", "Without"), r["steps_per_sec"]) for r in bench["ablation"]]
    gated = early["runs"]["learned/gated"]["saved_pct_held_out"]
    gated_swapped = swapped["runs"]["learned/gated"]["saved_pct_held_out"]
    commands = [
        ("pip install torch numpy matplotlib pillow pytest", "# 1. Install"),
        ("python -m pytest -q", "# 2. Run the tests"),
        ("python reproduce.py", "# 3. Rebuild the data, model and results"),
        ('python sample.py --prompt "def read_config(path):"', "# 4. Generate code"),
    ]
    return [
        (4, "", title_scene("A Transformer from scratch", "How it works, what it learned, and how to run it"), ["Built by hand, tested, and measured honestly."]),
        (8, "Step 1: Turn code into tokens", tokens_scene(ids, tokens), ["A custom tokenizer splits code into tokens.", "Each token becomes a number the model can use."]),
        (11, "Step 2: The architecture", architecture_scene(), ["Each token flows through embeddings, six blocks, and an output layer.", "Every block has masked attention and a feed-forward network."]),
        (12, "Step 3: Attention, from the real model", attention_scene(demo, [8, 14, 17, last - 2, last - 1]), ["The orange token looks back at earlier tokens. Darker blue means more attention.", "Grey tokens are in the future. The mask hides them, so the model cannot cheat."]),
        (9, "Step 4: Training on Python source code", training_scene(log, meta["bigram_val_loss"]), ["Loss fell to 2.205 on unseen files, far below the bigram baseline.", "Training loss kept falling after that, which is overfitting."]),
        (9, "Step 5: Measuring speed", speed_scene(speed), ["Training became 2.3x faster overall.", "bfloat16 and SDPA attention did the work. The other two changes made no measurable difference."]),
        (9, "Step 6: What it generates", sample_scene(real_sample()), ["This is real output. It looks like Python.", "But only 1.0% of samples were valid complete functions."]),
        (9, "Step 7: Testing an idea, honestly", early_exit_scene(gated, gated_swapped), ["Can easy tokens leave the network early? The rule was written before the test.", "It saved almost nothing on this model, so the study stopped."]),
        (10, "How to run it", terminal_scene(commands), ["Four commands take you from a fresh install to generating code.", "A CUDA GPU is needed to reproduce everything."]),
        (4, "", title_scene("Read the README for the full results", "Code, tests, charts and the study write-up are all in this repo"), ["Thanks for watching."]),
    ]


SPEAK = """
Add-Type -AssemblyName System.Speech
$voice = New-Object System.Speech.Synthesis.SpeechSynthesizer
$voice.SelectVoice('Microsoft Zira Desktop')
$voice.Rate = 0
$format = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(22050, 'Sixteen', 'Mono')
$voice.SetOutputToWaveFile($env:VOICE_PATH, $format)
$voice.Speak($env:VOICE_TEXT)
$voice.Dispose()
"""


def speak(text, path):
    env = {**os.environ, "VOICE_TEXT": text.replace("bfloat16", "b float sixteen").replace("SDPA", "S D P A"), "VOICE_PATH": str(path)}
    subprocess.run(["powershell", "-NoProfile", "-Command", SPEAK], env=env, check=True, capture_output=True)
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes(), audio.readframes(audio.getnframes())


def narrate(scenes, folder, track):
    """Fit each scene to its narration, and write one continuous audio track."""
    fitted = []
    with wave.open(str(track), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(22050)
        for index, (seconds, heading, draw, captions) in enumerate(scenes):
            samples, audio = speak(" ".join(captions), Path(folder) / f"{index}.wav")
            frames = max(int(seconds * FPS), int((samples / 22050 + 1.2) * FPS))
            lead = int(0.4 * 22050)
            total = int(frames / FPS * 22050)
            out.writeframes(b"\0\0" * lead + audio + b"\0\0" * max(0, total - lead - samples))
            fitted.append((frames, heading, draw, captions))
    return fitted


def caption_at(captions, t):
    used, total = 0, sum(len(c) for c in captions)
    for caption in captions:
        used += len(caption)
        if t <= used / total:
            return caption
    return captions[-1]


def render(scenes):
    for index, (frames, heading, draw, captions) in enumerate(scenes):
        for n in range(frames):
            t = n / max(1, frames - 1)
            image = Image.new("RGB", (W, H), BG)
            d = ImageDraw.Draw(image)
            if heading:
                d.text((60, 52), heading, font=F["title"], fill=INK)
            draw(d, t)
            d.rectangle((0, 615, W, 700), fill=CHIP)
            lines = wrap(d, caption_at(captions, t), F["caption"], W - 140)[:2]
            for i, line in enumerate(lines):
                d.text((W / 2, 657 + (i - (len(lines) - 1) / 2) * 34), line, font=F["caption"], fill=INK, anchor="mm")
            d.rectangle((0, 714, W * (index + t) / len(scenes), 720), fill=BLUE)
            fade = min(n / FADE, (frames - 1 - n) / FADE, 1.0)
            if fade < 1.0:
                image = Image.blend(Image.new("RGB", (W, H), BG), image, fade)
            yield index, image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ffmpeg", default=shutil.which("ffmpeg"))
    parser.add_argument("--encoder", default="libx264")
    args = parser.parse_args()
    if not args.ffmpeg:
        raise SystemExit("ffmpeg not found. Install it, or pass --ffmpeg with its path.")

    gif_frames = []
    with tempfile.TemporaryDirectory() as folder:
        track = Path(folder) / "narration.wav"
        scenes = narrate(build_scenes(), folder, track)
        command = [
            args.ffmpeg, "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
            "-i", str(track),
            "-c:v", args.encoder, "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", "-shortest", str(RESULTS / "how_it_works.mp4"),
        ]
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE)
        for n, (index, image) in enumerate(render(scenes)):
            encoder.stdin.write(image.tobytes())
            if index == 3 and n % 3 == 0:
                gif_frames.append(image.resize((640, 360), Image.LANCZOS).quantize(64))
        encoder.stdin.close()
        if encoder.wait() != 0:
            raise SystemExit("ffmpeg failed")
    gif_frames[0].save(RESULTS / "attention.gif", save_all=True, append_images=gif_frames[1:], duration=125, loop=0, optimize=True)


if __name__ == "__main__":
    main()
