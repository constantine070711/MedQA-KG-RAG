# MedQA v2 Image Recognition

MedQA v2 supports images of drug boxes, package inserts, and laboratory reports
as an input aid. AI vision is the primary mode, with browser-local OCR as a
fallback. The feature is intentionally limited to text recognition and
knowledge-term lookup; it does not diagnose from an image or interpret personal
test values.

## Workflow

1. The user selects a JPG, PNG, or WebP image up to 8 MB.
2. The user chooses AI recognition or local recognition.
3. AI recognition uploads the image to `/api/vision/extract`; the FastAPI
   backend sends it to the configured multimodal model.
4. If the AI request fails, the browser automatically retries with Tesseract.js
   `chi_sim+eng` in a local Web Worker.
5. Recognized text is displayed in an editable textarea.
6. The browser compares recognized text with the public curated entity list.
7. The user can:
   - ask a general knowledge question about a detected disease or drug;
   - place the edited text into the question composer and review it before
     sending.

In AI mode, the image is sent only to the MedQA API and the configured
multimodal provider. The backend does not persist the original image or
recognized text. In local mode, no image leaves the browser.

## Recognition Modes

- **AI recognition:** better for complex reports, glare, small text, and tables;
  requires network access and a vision-capable model.
- **Local OCR:** keeps the image inside the browser and works offline after the
  Tesseract runtime is cached; accuracy is lower for photographed documents.

## Runtime Data

The Tesseract.js package itself is bundled by npm. On the first OCR run, the
browser downloads the Chinese/English trained-data and WebAssembly runtime, and
the browser may cache them for later scans.

AI recognition uses `VISION_MODEL` and falls back to the text model settings
when separate vision variables are not configured. The selected model must
accept OpenAI-compatible `image_url` content.

## Safety Boundary

- OCR text is editable and is not treated as trusted medical evidence.
- AI recognition results may still be wrong and always require review.
- Personal laboratory values are not automatically interpreted.
- Detected entities lead only to general knowledge questions such as
  `请介绍二甲双胍相关的医学知识。`
- Diagnosis, dosage, and safety checks remain enforced by the existing Agent
  workflow and reliability gate.

## Limitations

- Handwriting, low-resolution photos, glare, and rotated images reduce OCR
  accuracy.
- The backend vision model must support image input; otherwise the UI
  automatically falls back to local OCR.
- The curated graph currently covers a limited disease/drug vocabulary.
- PDF documents are not supported directly; convert a page to an image first.
- First recognition can be slow while the OCR model downloads.
