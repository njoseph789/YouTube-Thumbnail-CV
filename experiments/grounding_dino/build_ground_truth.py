"""Build an image-level ground truth for the 100-image labeled subset.

Each image is annotated with the keywords from prompts.py that are visibly
present in it (presence only, no boxes). Annotations were made by manually
reviewing every thumbnail. Output: ground_truth.csv (image, category, label).

Run: python experiments/grounding_dino/build_ground_truth.py
"""
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prompts import KEYWORDS  # noqa: E402

IMAGE_DIR = HERE.parents[1] / "images" / "labeled_subset"
OUT_CSV = HERE / "ground_truth.csv"

ANN = {}


def A(idx, channel, labels):
    ANN[idx] = (channel, [s.strip() for s in labels.split(",") if s.strip()])


A(0, "AM HOOPS", "person, man, basketball player, head, hand, hair, basketball, basketball jersey, jersey, team logo")
A(1, "AM HOOPS", "person, man, basketball player, head, beard, basketball, basketball jersey, jersey, team logo, arrow, hand, hair")
A(2, "AM HOOPS", "person, man, basketball player, head, hair, beard, basketball jersey, jersey")
A(3, "AM HOOPS", "person, man, basketball player, head, hair, beard, headband, basketball jersey, jersey, team logo")
A(4, "Almost Friday TV", "person, man, woman, head, hair, glasses")
A(5, "Almost Friday TV", "person, man, head, hair, baseball cap, glasses")
A(6, "Almost Friday TV", "person, man, woman, head, hair, glasses")
A(7, "AlphaPhoenixChannel", "ice cube, magnifying glass, lens, steam, channel logo")
A(8, "Casually Explained", "cartoon character, camera, lens, umbrella, stool")
A(9, "DIYPerks", "person, man, head, hair, hand, speaker, metal box, wires")
A(10, "Degenerocity", "person, man, woman, head, hand, glasses")
A(11, "Degenerocity", "trash can, garbage bag")
A(12, "Degenerocity", "person, boy, man, head, hand, glasses, smartphone, phone, hair")
A(13, "Domics", "cartoon character, hair")
A(14, "Drew Gooden", "person, man, woman, head, hair, hand, emoji, microphone")
A(15, "Drew Gooden", "person, man, head, hair, postmates logo, doordash logo, ubereats logo")
A(16, "Drew Gooden", "person, man, head, beard, baseball cap, hair")
A(17, "Drew Gooden", "person, man, woman, head, hair, hand, question mark")
A(18, "ElectroBOOM", "person, man, head, heater, channel logo")
A(19, "ElectroBOOM", "person, man, head, hand, headphones, car, red arrow, arrow, lightning, channel logo")
A(20, "ElectroBOOM", "person, man, head, hand, finger, channel logo")
A(21, "ElectroBOOM", "person, man, head, hand, knife, baseball cap, lightning, channel logo")
A(22, "Fireship", "person, man, head, hair, hand, epic games logo, google logo, channel logo")
A(23, "Good Work", "person, man, head, hair, hand, glasses, microphone, building")
A(24, "Ididathing", "knife, pencil, ruler")
A(25, "Ididathing", "person, man, head, hair, paint splatter, arrow")
A(26, "Incognito Mode", "astronaut, cartoon character")
A(27, "Ironcane", "person, man, head, hair, headphones, anime character, dragon ball character, thumbs up icon, thumbs down icon, circle, red circle")
A(28, "Ironcane", "person, man, head, hair, headphones, anime character, dragon ball character, sticker")
A(29, "JxmyHighroller", "person, man, head, hand, crowd of people, basketball court")
A(30, "JxmyHighroller", "person, man, basketball player, basketball court, head, hand, hair, basketball jersey, jersey")
A(31, "JxmyHighroller", "person, man, woman, head, hair, hand, beard, crowd of people, basketball player, basketball jersey, jersey, scoreboard, team logo")
A(32, "JxmyHighroller", "person, man, head, hand, basketball player, basketball jersey, jersey, headband, basketball court, hair")
A(33, "Kyle Tsuchiya", "person, man, head, hand, hair, beard")
A(34, "Make With Miles", "pegboard")
A(35, "MarkRober", "person, man, head, hair, hand, baseball cap, glasses, sandwich, burger")
A(36, "MarkRober", "person, pickup truck, truck, tower, explosion, red circle, circle, red arrow, arrow")
A(37, "MarkRober", "person, man, boy, child, head, hair, hand")
A(38, "Matt Stonie", "chicken nuggets")
A(39, "Matt Stonie", "person, man, head, hair, headband, baseball cap, plate of food, hand, tree")
A(40, "Matt Stonie", "pizza, smartphone, phone, red arrow, arrow")
A(41, "MichaelReeves", "person, boy, head, hair, hand, camera, lens, wires, arrow")
A(42, "MichaelReeves", "person, boy, head, hair, glasses, boxing gloves, wires, lightning")
A(43, "Nanogenix", "goku, dragon ball character, anime character")
A(44, "Nanogenix", "person, man, head, hair, hand, headphones, smartphone, phone, goku, dragon ball character, anime character, playstation logo")
A(45, "Nanogenix", "person, man, head, hair, hand, headphones, goku, dragon ball character, anime character")
A(46, "Nanogenix", "action figure, toy figures, glass display cabinet, shelf")
A(47, "NileRed", "powder, petri dish")
A(48, "PracticalEngineeringChannel", "bridge, reservoir, lake, tree, river")
A(49, "PracticalEngineeringChannel", "channel logo")
A(50, "Scamboli Reviews", "anime character, hair")
A(51, "Scamboli Reviews", "drawn monster")
A(52, "Scamboli Reviews", "drawn monster")
A(53, "Scamboli Reviews", "anime character, cartoon character, crown, demon, drawn monster, hair")
A(54, "Sidetalk", "person, man, woman, head, hair, hand, crowd of people, microphone, face mask, baseball cap, glasses, smartphone, phone, building")
A(55, "Sidetalk", "person, man, head, beard, baseball cap, face mask, building, hair, microphone")
A(56, "Sidetalk", "person, man, head, microphone, hair, beard")
A(57, "Sidetalk", "person, man, head, baseball cap, face mask, building, tree")
A(58, "SmarterEveryDay", "mountain, ruins, tree")
A(59, "SteveMould", "hand, finger")
A(60, "SteveMould", "bead chain, chain, clamp")
A(61, "Summoning Salt", "video game screenshot, race track")
A(62, "TheActionLab", "hand, finger, oreo cookies")
A(63, "TheActionLab", "hand, finger, red arrow, arrow")
A(64, "TheActionLab", "hand, finger, red arrow, arrow")
A(65, "TheActionLab", "person, man, head, hair, hand, cotton candy")
A(66, "TheBackyardScientist", "person, man, head, camera, fence, helmet, tree")
A(67, "TheBackyardScientist", "person, man, head, hand, glasses, rifle, gun, ear protection, pickup truck, tree")
A(68, "TheBackyardScientist", "person, man, head, hair, hand, metal box, chain, padlock")
A(69, "Thinking Basketball", "person, man, basketball player, head, basketball jersey, jersey, hair")
A(70, "Trash Taste", "person, man, head, hair, hand, microphone, anime character, channel logo, beard")
A(71, "Veritasium", "person, man, head, hand, tree, hair, beard")
A(72, "Veritasium", "green slime, youtube logo, channel logo")
A(73, "Veritasium", "person, man, head, hand, glasses, basketball, tree, hair, beard")
A(74, "WolfeyVGC", "person, man, head, hair, beard, glasses, pokemon, red arrow, arrow, flames")
A(75, "WolfeyVGC", "person, man, head, hair, beard, hand, finger, game case, nintendo switch, pokemon")
A(76, "WolfeyVGC", "video game screenshot, pixel art, pokemon")
A(77, "WolfeyVGC", "person, man, head, hair, beard, glasses, hand, pokemon, gyarados")
A(78, "WolfeyVGC", "person, man, head, hair, beard, glasses, pokemon, squirtle")
A(79, "allenpan", "person, man, hand, emoji")
A(80, "colinfurze", "person, man, head, hair, hand, flames, fire")
A(81, "colinfurze", "person, man, head, hair, hand, light bulb")
A(82, "colinfurze", "person, man, head, hair, hand, kettle, milk bottle, jars, steam, smoke, flames, fire")
A(83, "colinfurze", "person, building, fence")
A(84, "colinfurze", "person, car")
A(85, "hacksmith", "person, man, head, hair, beard, hand, briefcase of cash, money, dollar bills, dollar sign, iron man suit, red arrow, arrow")
A(86, "hacksmith", "person, man, head, hair, beard, armor suit, mandalorian helmet, helmet, smoke, hand")
A(87, "hacksmith", "")
A(88, "hacksmith", "person, man, head, hair, beard, hand, channel logo")
A(89, "hacksmith", "person, motorcycle, jet engine, flames, smoke, channel logo, race track, helmet, car")
A(90, "jamesbruton", "person, man, head, hair, helmet, armor suit, iron man suit")
A(91, "jamesbruton", "person, man, head, hair, hand, finger, glasses, CNC machine, wires")
A(92, "jamesbruton", "exoskeleton, robot, wires, 3D printed part")
A(93, "simonegiertz", "person, woman, head, hair, hand, alarm clock")
A(94, "simonegiertz", "person, woman, head, hair, hand, face mask, machine arm, power tool, glasses")
A(95, "simonegiertz", "person, woman, head, hair, hand")
A(96, "simonegiertz", "person, woman, head, hair, hand, glasses, stool, workbench, toolbox")
A(97, "whatslaps", "person, man, basketball player, head, hair, beard, basketball jersey, jersey")
A(98, "whatslaps", "person, man, basketball player, head, beard, hand, basketball jersey, jersey, hair, headband")
A(99, "whatslaps", "person, man, head, beard, money, dollar bills, dollar sign, jersey, hair, basketball jersey, hand")


def main():
    files = sorted(f.name for f in IMAGE_DIR.glob("*.jpg"))
    assert len(files) == len(ANN) == 100, (len(files), len(ANN))

    label_to_cat = {}
    for cat, kws in KEYWORDS.items():
        for kw in kws:
            assert kw not in label_to_cat, f"duplicate keyword {kw!r}"
            label_to_cat[kw] = cat

    rows = []
    for idx, fname in enumerate(files):
        channel, labels = ANN[idx]
        assert fname.startswith(channel + "__"), (idx, fname, channel)
        assert len(set(labels)) == len(labels), (idx, "duplicate label")
        for lab in labels:
            assert lab in label_to_cat, (idx, fname, f"unknown keyword {lab!r}")
            rows.append((fname, label_to_cat[lab], lab))

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "category", "label"])
        w.writerows(rows)

    used = {r[2] for r in rows}
    print(f"wrote {len(rows)} rows for {len(files)} images -> {OUT_CSV}")
    print(f"keywords used: {len(used)}/{len(label_to_cat)}")
    print("unused keywords:", sorted(set(label_to_cat) - used))


if __name__ == "__main__":
    main()
