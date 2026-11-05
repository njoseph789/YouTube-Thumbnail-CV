# YouTube Thumbnail Feature Extraction: Proposed Project Plan

## 1. Collect and store
Pull thumbnails and metadata from the YouTube API, with images and metadata linked by video ID.

## 2. Extract features (Neural Networks)

- **Objects:** Customized detector, with Grounding DINO (Nathan), Vision Transformer Model (Nathan) and YOLO-World (Eric) as candidates. The transfer learning owner will compare and experiment with configurations. Get a 100 image sample set (Eric). Test DINO and YOLO-World off this to determine which one is a better baseline.
- Assess text extraction between text in the background versus overlayed (To separate when passing to Florence-2 OCR model, nice to have but not critical)
- **Faces:** MediaPipe Face Detection and RetinaFace as candidates, plus an emotion classifier. - Flagged in the pipeline if a face is detected (Ask Alex to pick one --> Colin looks into it because Alex is focused on the generative model)
- **Text:** Florence-2 (Nick). - Flagged in the pipeline if text is detected
- **Color & Visual Noise:** Computed directly from the pixels (saturation hue, warm vs cool, rgb range). - Secondary predictor behind the extracted features (Eric)
- Object, face, and text detection already record size and locations within the thumbnail framing.
- Using Grad-CAM to highlight on which areas are actually being looked at by the model (Supplemental feature if we have time)

## 3. Build the feature table
One row per thumbnail, joined with the API metadata. Full per-detection data is kept (counts, emotions, locations).

## 4. Add performance metrics
Our own metrics, normalized for channel size and audience, are the targets.

## 5. Regression or Clustering Even?
Relate the features to the metrics to see which are associated with the best-performing videos.

## Open for later
- How to condense the per-detection data for the regression.
- How to define genres (May be out of scope)

## Tentative Timeline
- 100 images by 11/7
- Models tested by 11/12

