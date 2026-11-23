# Face detection: notes on questionable labels

Scored against `images/face_labels.csv` (100 images in `images/labeled_subset`).
Pipeline: 6 detectors, boxes kept at 2+ votes, CLIP real-vs-cartoon filter (`p_real >= 0.3`).
See `final_mismatches.csv` (real-face disagreements) and `facecheck.csv` (any-face disagreements) for the full lists.

## Label questions (please confirm with the team)

| Image | Label | Pipeline | Note |
|---|---|---|---|
| Thinking Basketball__OSmKEaZD_vQ.jpg | `has_real_face=1` | not marked real | Drawn face labeled as a real face. Looks like label noise. |
| allenpan__LGD6ustR8Zs.jpg | `has_face=0` | face found | Mannequin. Face-like but not a person; labeled as not a face. Decide whether mannequins count. |

## Expected misses (not label errors)

| Image | Note |
|---|---|
| TheBackyardScientist__-_AshChTEY0.jpg | Real face almost fully hidden by a welding mask with glare. Only one detector (YuNet) boxes it, so it stays below the 2-vote threshold. |

## Flagged, not yet checked by eye

These appear as disagreements but nobody has confirmed whether the label or the detector is wrong.

- jamesbruton__kUsIIin0j6c.jpg: ensemble finds a face (3 votes); label says no face.
- Veritasium__sWu9CyDwuFs.jpg: ensemble finds a face (gdino + YOLO); label says no face.
- Scamboli Reviews__NS0ryWdNJwA.jpg, Scamboli Reviews__Sk8VtJ9ELQQ.jpg, Domics__LAS4CSLAiyY.jpg: labeled `has_face=1`; only Grounding DINO sees a face, so the ensemble misses them (stylized faces).

## Open question for the team

Do cartoon, anime, mascot or emoji faces count as faces? `faces_final.csv` keeps them all and sets `is_real` (CLIP score at 0.3), so either definition can be selected by filtering.

## Caveats

- Thresholds were tuned on the same 100 images; a held-out check on new images is still to do.
- Image-level labels do not validate box position; 36 of the boxes were reviewed by eye and looked tight and consistent.
