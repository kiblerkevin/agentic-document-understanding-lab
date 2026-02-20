# Migration Guide: PaddleOCR to Unstructured.io

## Overview
This guide explains the changes made to replace PaddleOCR with Unstructured.io to avoid LangChain dependency conflicts.

## Key Changes

### 1. Import Changes
**Before (PaddleOCR):**
```python
from paddleocr import PaddleOCR, LayoutDetection
ocr = PaddleOCR(lang="en")
layout_engine = LayoutDetection()
```

**After (Unstructured):**
```python
from unstructured.partition.image import partition_image
elements = partition_image(filename=image_path, strategy="hi_res", infer_table_structure=True)
```

### 2. OCR Extraction
- **PaddleOCR**: Returns `rec_texts`, `rec_scores`, `rec_polys`
- **Unstructured**: Returns unified `elements` with `.text`, `.category`, `.metadata.coordinates`

### 3. Layout Detection
- **PaddleOCR**: Separate `LayoutDetection()` engine
- **Unstructured**: Built-in layout detection via `partition_image()` with categories like `Title`, `Table`, `Image`, `NarrativeText`

### 4. Confidence Scores
- **PaddleOCR**: Provides confidence scores for each detection
- **Unstructured**: Does not provide confidence scores (defaults to 1.0)

## Installation

### Step 1: Create a new virtual environment (recommended)
```bash
python3.11 -m venv .venv_unstructured
source .venv_unstructured/bin/activate  # On macOS/Linux
```

### Step 2: Install system dependencies (macOS)
```bash
brew install tesseract
brew install poppler
```

### Step 3: Install Python packages
```bash
pip install unstructured[all-docs]
pip install python-dotenv pillow opencv-python matplotlib numpy
pip install langchain langchain-openai langchain-core
pip install transformers torch
```

### Step 4: Run the updated script
```bash
python src/main_unstructured.py
```

## Advantages of Unstructured.io

1. ✅ **No LangChain conflicts** - Works with modern LangChain versions
2. ✅ **Unified API** - Single function for OCR + layout detection
3. ✅ **Better document support** - Handles PDFs, DOCX, HTML, etc.
4. ✅ **Active development** - Regular updates and improvements
5. ✅ **Table structure extraction** - Built-in table parsing

## Disadvantages

1. ❌ **No confidence scores** - Cannot filter low-confidence detections
2. ❌ **Slower** - More comprehensive but takes longer to process
3. ❌ **Larger dependencies** - Requires more system packages

## Troubleshooting

### Issue: "tesseract not found"
```bash
brew install tesseract  # macOS
sudo apt-get install tesseract-ocr  # Ubuntu
```

### Issue: "poppler not found"
```bash
brew install poppler  # macOS
sudo apt-get install poppler-utils  # Ubuntu
```

### Issue: Detectron2 installation fails
Detectron2 may not be needed for basic image processing. Try running without it first.

## Alternative: Simpler Installation

If you encounter issues with the full installation, use a minimal setup:

```bash
pip install unstructured
pip install pillow opencv-python matplotlib numpy
pip install langchain langchain-openai langchain-core
pip install transformers torch
```

This will use basic OCR without advanced layout detection.
