# YouTube Thumbnail Feature Extraction: Proposed Project Plan

## 1. Collect and store
Pull thumbnails and metadata from the YouTube API, with images and metadata linked by video ID.

## 2. Extract features (Neural Networks)

- **Objects:** Customized detector, with Grounding DINO and YOLO-World as candidates. The transfer learning owner will compare and experiment with configurations.
- **Faces:** MediaPipe Face Detection and RetinaFace as candidates, plus an emotion classifier.
- **Text:** PaddleOCR and Florence-2 as candidates.
- **Color and lighting:** Computed directly from the pixels (brightness, contrast, saturation, warm vs cool, dominant colors).
- Object, face, and text detection already record size and locations within the thumbnail framing.

## 3. Build the feature table
One row per thumbnail, joined with the API metadata. Full per-detection data is kept (counts, emotions, locations).

## 4. Add performance metrics
Our own metrics, normalized for channel size and audience, are the targets.

## 5. Regression
Relate the features to the metrics to see which are associated with the best-performing videos.

## Open for later
- How to condense the per-detection data for the regression.
