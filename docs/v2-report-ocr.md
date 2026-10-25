# MedQA v2 Image OCR

MedQA v2 supports images of drug boxes, package inserts, and laboratory reports
as an input aid. The feature is intentionally limited to text recognition and
knowledge-term lookup; it does not diagnose from an image or interpret personal
test values.

## Workflow

1. The user selects a JPG, PNG, or WebP image up to 8 MB.
2. Tesseract.js runs OCR locally in a browser Web Worker using
   `chi_sim+eng`.
3. Recognized text is displayed in an editable textarea.
4. The browser compares recognized text with the public curated entity list.
5. The user can:
   - ask a general knowledge question about a detected disease or drug;
   - place the edited text into the question composer and review it before
     sending.

No image or OCR text is sent to the MedQA API unless the user explicitly uses
the recognized text in a chat request.

## Why Local OCR

- Keeps the source image inside the browser.
- Avoids a server-side image file upload and retention path.
- Works without adding a vision-model API key.
- Keeps the interaction focused on terminology lookup rather than automated
  diagnosis.

## Runtime Data

The Tesseract.js package itself is bundled by npm. On the first OCR run, the
browser downloads the Chinese/English trained-data and WebAssembly runtime, and
the browser may cache them for later scans.

## Safety Boundary

- OCR text is editable and is not treated as trusted medical evidence.
- Personal laboratory values are not automatically interpreted.
- Detected entities lead only to general knowledge questions such as
  `请介绍二甲双胍相关的医学知识。`
- Diagnosis, dosage, and safety checks remain enforced by the existing Agent
  workflow and reliability gate.

## Limitations

- Handwriting, low-resolution photos, glare, and rotated images reduce OCR
  accuracy.
- The curated graph currently covers a limited disease/drug vocabulary.
- PDF documents are not supported directly; convert a page to an image first.
- First recognition can be slow while the OCR model downloads.
