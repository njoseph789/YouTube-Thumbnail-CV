"""Grounding DINO text prompts for the labeled thumbnail subset.

Keywords were compiled by reviewing all images in images/labeled_subset.
Grounding DINO truncates prompts at ~256 tokens, so keywords are split into
category prompts that should be run separately and the detections merged.

Objects only: faces and overlaid text are handled elsewhere by teammates, so
face/text keywords are intentionally excluded.

Each prompt is a lowercase string of noun phrases separated by " . ".
"""

# Suggested starting thresholds (overlay graphics may need lower values).
BOX_THRESHOLD = 0.30
TEXT_THRESHOLD = 0.25

KEYWORDS = {
    "people": [
        "person", "man", "woman", "boy", "child", "head", "hand",
        "finger", "hair", "beard",
        "crowd of people", "basketball player", "astronaut",
    ],
    "accessories": [
        "headphones", "glasses", "baseball cap", "helmet", "face mask",
        "microphone", "hard hat", "welding mask", "ear protection", "headband",
    ],
    "cartoon_characters": [
        "cartoon character", "anime character", "emoji",
    ],
    "sports": [
        "basketball", "basketball jersey", "basketball court", "scoreboard",
        "team logo", "jersey",
    ],
    "food": [
        "pizza", "chicken nuggets", "oreo cookies", "sandwich", "burger",
        "plate of food", "bowl", "cotton candy", "candy", "powder",
        "petri dish",
    ],
    "tools_electronics": [
        "knife", "pencil", "ruler", "chain", "padlock", "metal box",
        "circuit board", "wires", "magnet", "speaker", "camera", "lens",
        "phone", "smartphone", "video game console", "game case", "clamp",
        "bead chain", "CNC machine", "robot", "exoskeleton",
        "3D printed part", "power tool", "machine arm", "heater",
        "light bulb", "alarm clock", "kettle", "milk bottle", "jars", "shelf",
        "glass display cabinet", "workbench", "pegboard", "toolbox",
    ],
    "vehicles_structures": [
        "car", "truck", "pickup truck", "motorcycle", "jet engine",
        "race track", "bridge", "reservoir", "river", "mountain", "ruins",
        "tree", "building", "tower", "lake", "fence",
    ],
    "weapons_effects": [
        "gun", "rifle", "sword", "boxing gloves", "armor suit",
        "iron man suit", "mandalorian helmet", "lightning", "fire", "flames",
        "smoke", "explosion", "sparks", "steam", "ice cube",
        "magnifying glass", "green slime", "paint splatter", "confetti",
        "money", "dollar bills", "briefcase of cash",
    ],
    "games_characters": [
        "pokemon", "squirtle", "gyarados", "dragon ball character", "goku",
        "action figure", "toy figures", "video game screenshot", "pixel art",
        "drawn monster", "demon", "crown", "trash can", "garbage bag",
        "umbrella", "stool",
    ],
    "logos": [
        "google logo", "epic games logo", "youtube logo", "postmates logo",
        "doordash logo", "ubereats logo", "channel logo", "nintendo switch",
        "playstation logo",
    ],
    "overlays": [
        "arrow", "red arrow", "circle", "red circle", "question mark",
        "exclamation mark", "thumbs up icon", "thumbs down icon",
        "speech bubble", "dollar sign", "sticker", "border outline",
    ],
}


def make_prompt(keywords):
    """Join keywords into a Grounding DINO prompt: 'a . b . c .'"""
    return " . ".join(k.lower() for k in keywords) + " ."


PROMPTS = {name: make_prompt(kws) for name, kws in KEYWORDS.items()}

if __name__ == "__main__":
    for name, prompt in PROMPTS.items():
        print(f"[{name}] ({len(KEYWORDS[name])} keywords)\n{prompt}\n")
