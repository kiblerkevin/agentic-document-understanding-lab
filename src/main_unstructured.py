import os
from dotenv import load_dotenv

_ = load_dotenv(override=True)

from PIL import Image
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any

# Replace PaddleOCR with Unstructured
from unstructured.partition.auto import partition

# Document path - supports both images and PDFs
document_path = "report_original.png"  # Change to .pdf for PDF files

###########################
# 1.1 Run OCR on Document #
###########################
# Use Unstructured to extract elements (works for both images and PDFs)
elements = partition(
    filename=document_path,
    strategy="hi_res",
    infer_table_structure=True,
)

print(f"Extracted {len(elements)} elements")
print("\nFirst 10 elements:")
for i, elem in enumerate(elements[:10]):
    text_preview = (elem.text[:40] if elem.text else "<empty>").ljust(40)
    print(f"{text_preview} | {elem.category:15} | {elem.metadata.coordinates if hasattr(elem.metadata, 'coordinates') else 'N/A'}")

######################################
# 1.2 Visualizing OCR Bounding Boxes #
######################################
# Handle both PDF and image files
if document_path.endswith('.pdf'):
    from pdf2image import convert_from_path
    pages = convert_from_path(document_path, first_page=1, last_page=1)
    img = cv2.cvtColor(np.array(pages[0]), cv2.COLOR_RGB2BGR)
    temp_image_path = "temp_page.png"
    cv2.imwrite(temp_image_path, img)
    image_path = temp_image_path
else:
    img = cv2.imread(document_path)
    image_path = document_path

img_plot = img.copy()
show_text = False

for elem in elements:
    if hasattr(elem.metadata, 'coordinates') and elem.metadata.coordinates:
        points = elem.metadata.coordinates.points
        pts = np.array([[int(p[0]), int(p[1])] for p in points], dtype=int)
        cv2.polylines(img_plot, [pts], True, (0, 255, 0), 2)
        if show_text and elem.text:
            x, y = pts[0]
            cv2.putText(img_plot, elem.text[:20], (x, y - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("Unstructured Bounding Boxes")
plt.show()

###############################################
# 1.3 Structuring OCR Results using Dataclass #
###############################################
@dataclass
class OCRRegion:
    text: str
    bbox: list  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    confidence: float
    
    @property
    def bbox_xyxy(self):
        """Return bbox as [x1, y1, x2, y2] format."""
        x_coords = [p[0] for p in self.bbox]
        y_coords = [p[1] for p in self.bbox]
        return [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]

ocr_regions: List[OCRRegion] = []
for elem in elements:
    if hasattr(elem.metadata, 'coordinates') and elem.metadata.coordinates:
        points = elem.metadata.coordinates.points
        bbox = [[int(p[0]), int(p[1])] for p in points]
        ocr_regions.append(OCRRegion(
            text=elem.text or "", 
            bbox=bbox, 
            confidence=1.0
        ))

print(f"Stored {len(ocr_regions)} OCR regions")

###############################
# 1.4 Layout LM Reading Order #
###############################
from transformers import LayoutLMv3ForTokenClassification
from helpers import prepare_inputs, boxes2inputs, parse_logits

print("Loading LayoutReader model...")
model_slug = "hantian/layoutreader"
layout_model = LayoutLMv3ForTokenClassification.from_pretrained(model_slug)
print("Model loaded successfully!")

def get_reading_order(ocr_regions):
    """Use LayoutReader to determine reading order of OCR regions."""
    max_x = max_y = 0
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    image_width = max_x * 1.1
    image_height = max_y * 1.1

    boxes = []
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        left = int((x1 / image_width) * 1000)
        top = int((y1 / image_height) * 1000)
        right = int((x2 / image_width) * 1000)
        bottom = int((y2 / image_height) * 1000)
        boxes.append([left, top, right, bottom])

    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, layout_model)
    logits = layout_model(**inputs).logits.cpu().squeeze(0)
    reading_order = parse_logits(logits, len(boxes))

    return reading_order

reading_order = get_reading_order(ocr_regions)

print(f"Reading order determined for {len(reading_order)} regions")
print(f"First 20 positions: {reading_order[:20]}")

#####################################
# 1.5 Visualizing the Reading Order #
#####################################
import matplotlib.patches as patches

def visualize_reading_order(ocr_regions, image_path, reading_order, title="Reading Order"):
    """Visualize OCR regions with their reading order numbers."""
    img = cv2.imread(image_path)
    
    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    order_map = {i: order for i, order in enumerate(reading_order)}
    
    for i, region in enumerate(ocr_regions):
        bbox = region.bbox
        if bbox and len(bbox) >= 4:
            ax.add_patch(patches.Polygon(bbox, linewidth=2, 
                                         edgecolor='blue',
                                         facecolor='none', alpha=0.7))
            xs = [p[0] for p in bbox]
            ys = [p[1] for p in bbox]
            ax.text(sum(xs)/len(xs), sum(ys)/len(ys), 
                    str(order_map.get(i, i)),
                    fontsize=13, color='red', 
                    ha='center', va='center', fontweight='bold')
    
    ax.set_title(title, fontsize=14)
    ax.axis('off')
    plt.tight_layout()
    plt.show()

visualize_reading_order(ocr_regions, image_path, 
                        reading_order, "LayoutLM Reading Order")

########################################
# 1.6 Creating the Ordered Text Output #
########################################
def get_ordered_text(ocr_regions, reading_order):
    """Return OCR regions sorted by reading order."""
    indexed_regions = [(reading_order[i], i, ocr_regions[i]) for i in range(len(ocr_regions))]
    indexed_regions.sort(key=lambda x: x[0])  
    
    ordered_text = []
    for position, original_idx, region in indexed_regions:
        ordered_text.append({
            "position": position,
            "text": region.text,
            "confidence": region.confidence,
            "bbox": region.bbox_xyxy
        })
    
    return ordered_text

ordered_text = get_ordered_text(ocr_regions, reading_order)

print("Text in reading order:")
print("=" * 70)
print(ordered_text[:5])

##################################
# 2.1 Processing Document Layout #
##################################
def process_document(elements):
    """Get layout regions from Unstructured elements."""
    regions = []
    
    category_map = {
        'Title': 'title',
        'NarrativeText': 'text',
        'Text': 'text',
        'Table': 'table',
        'Image': 'figure',
        'FigureCaption': 'caption',
        'ListItem': 'list',
    }
    
    for elem in elements:
        if hasattr(elem.metadata, 'coordinates') and elem.metadata.coordinates:
            points = elem.metadata.coordinates.points
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            bbox = [min(x_coords), min(y_coords), max(x_coords), max(y_coords)]
            
            regions.append({
                'label': category_map.get(elem.category, 'text'),
                'score': 1.0,
                'bbox': bbox,
            })
    
    return regions

layout_results = process_document(elements)

print(f"Detected {len(layout_results)} layout regions:")
for r in layout_results[:10]:
    print(f"  {r['label']:20} score: {r['score']:.3f}  bbox: {[int(x) for x in r['bbox']]}")

##################################
# 2.2 Structuring Layout Results #
##################################
@dataclass
class LayoutRegion:
    region_id: int
    region_type: str
    bbox: list
    confidence: float
    
layout_regions: List[LayoutRegion] = []
for i, r in enumerate(layout_results):
    layout_regions.append(LayoutRegion(
        region_id=i,
        region_type=r['label'],
        bbox=[int(x) for x in r['bbox']],
        confidence=r['score']
    ))

print(f"Stored {len(layout_regions)} layout regions")

####################################
# 2.3 Visualizing Layout Detection #
####################################
from matplotlib import colormaps

def visualize_layout(image_path, layout_regions, min_confidence=0.5, title="Layout Detection"):
    """Visualize layout detection results."""
    img = cv2.imread(image_path)
    img_plot = img.copy()
    
    labels = list(set(r.region_type for r in layout_regions))
    cmap = colormaps.get_cmap('tab20')
    color_map = {}
    for i, label in enumerate(labels):
        rgba = cmap(i % 20)
        color_map[label] = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
    
    for region in layout_regions:
        if region.confidence < min_confidence:
            continue
            
        color = color_map[region.region_type]
        x1, y1, x2, y2 = region.bbox
        
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int)
        cv2.polylines(img_plot, [pts], True, color, 2)
        
        text = f"{region.region_id}: {region.region_type} ({region.confidence:.2f})"
        cv2.putText(img_plot, text, (x1, y1-8), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=(12, 16))
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.title(title)
    plt.show()
    
    return img_plot

visualize_layout(image_path, layout_regions, 
                 min_confidence=0.5, title="Unstructured Layout Detection")

print(f"\nProcessed document: {document_path}")
print(f"Document type: {'PDF' if document_path.endswith('.pdf') else 'Image'}")

########################################
# 2.4 Cropping Regions for Agent Tools #
########################################
import base64
from io import BytesIO

def crop_region(image, bbox, padding=10):
    """Crop a region from image with optional padding."""
    x1, y1, x2, y2 = bbox
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(image.width, x2 + padding)
    y2 = min(image.height, y2 + padding)
    return image.crop((x1, y1, x2, y2))

def image_to_base64(img):
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

pil_image = Image.open(image_path)

region_images = {}
for region in layout_regions:
    cropped = crop_region(pil_image, region.bbox)
    region_images[region.region_id] = {
        'image': cropped,
        'base64': image_to_base64(cropped),
        'type': region.region_type,
        'bbox': region.bbox
    }

print(f"Cropped {len(region_images)} regions")

full_image_base64 = image_to_base64(pil_image)

fig, axes = plt.subplots(5, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (region_id, data) in enumerate(list(region_images.items())[:15]):
    if i < len(axes):
        axes[i].imshow(data['image'])
        axes[i].set_title(f"Region {region_id}: {data['type']}")
        axes[i].axis('off')

for j in range(i+1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

##############################
# 3.1 VLM Helper and Prompts #
##############################
from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

vlm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

CHART_ANALYSIS_PROMPT = """You are a Chart Analysis specialist. 
Analyze this chart/figure image and extract:

1. **Chart Type**: (line, bar, scatter, pie, etc.)
2. **Title**: (if visible)
3. **Axes**: X-axis label, Y-axis label, and tick values
4. **Data Points**: Key values (peaks, troughs, endpoints)
5. **Trends**: Overall pattern description
6. **Legend**: (if present)

Return a JSON object with this structure:
```json
{{
  "chart_type": "...",
  "title": "...",
  "x_axis": {{"label": "...", "ticks": [...]}},
  "y_axis": {{"label": "...", "ticks": [...]}},
  "key_data_points": [...],
  "trends": "...",
  "legend": [...]
}}
```
"""

TABLE_ANALYSIS_PROMPT = """You are a Table Extraction specialist. 
Extract structured data from this table image.

1. **Identify Structure**: 
    - Column headers, row labels, data cells
2. **Extract All Data**: 
    - Preserve exact values and alignment
3. **Handle Special Cases**: 
    - Merged cells, empty cells (mark as null), multi-line headers

Return a JSON object with this structure:
```json
{{
  "table_title": "...",
  "column_headers": ["header1", "header2", ...],
  "rows": [
    {{"row_label": "...", "values": [val1, val2, ...]}},
    ...
  ],
  "notes": "any footnotes or source info"
}}
```
"""

def call_vlm_with_image(image_base64: str, prompt: str) -> str:
    """Call VLM with an image and prompt."""
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_base64}"}
            }
        ]
    )
    response = vlm.invoke([message])
    return response.content

######################################
# 3.2 Creating the AnalyzeChart Tool #
######################################
@tool
def AnalyzeChart(region_id: int) -> str:
    """Analyze a chart or figure region using VLM. 
    Use this tool when you need to extract data from charts, graphs, or figures.
    
    Args:
        region_id: The ID of the layout region to analyze (must be a chart/figure type)
    
    Returns:
        JSON string with chart type, axes, data points, and trends
    """
    if region_id not in region_images:
        return f"Error: Region {region_id} not found. Available regions: {list(region_images.keys())}"
    
    region_data = region_images[region_id]
    
    if region_data['type'] not in ['chart', 'figure']:
        return f"Warning: Region {region_id} is type '{region_data['type']}', not a chart/figure. Proceeding anyway."
    
    result = call_vlm_with_image(region_data['base64'], CHART_ANALYSIS_PROMPT)
    
    return result

print("AnalyzeChart tool defined")

######################################
# 3.3 Creating the AnalyzeTable Tool #
######################################
@tool
def AnalyzeTable(region_id: int) -> str:
    """
    Extract structured data from a table region using VLM.
    Use this tool when you need to extract tabular data 
    with headers and rows.
    
    Args:
        region_id: The ID of the layout region to analyze (must be a table type)
    
    Returns:
        JSON string with table headers, rows, and any notes
    """
    if region_id not in region_images:
        return f"Error: Region {region_id} not found. Available regions: {list(region_images.keys())}"
    
    region_data = region_images[region_id]
    
    if region_data['type'] != 'table':
        return f"Warning: Region {region_id} is type '{region_data['type']}', not a table. Proceeding anyway."
    
    result = call_vlm_with_image(region_data['base64'], TABLE_ANALYSIS_PROMPT)
    return result

print("AnalyzeTable tool defined")

#########################
# 3.4 Testing the Tools #
#########################
print("Testing AnalyzeChart...")
chart_regions = [r for r in layout_regions if r.region_type in ['chart', 'figure']]
if chart_regions:
    test_result = AnalyzeChart.invoke({"region_id": chart_regions[0].region_id})
    print(f"Chart analysis result:\n{test_result[:500]}...")
else:
    print("No chart regions found")
    
print("Testing AnalyzeTable...")
table_regions = [r for r in layout_regions if r.region_type == 'table']
if table_regions:
    test_result = AnalyzeTable.invoke({"region_id": table_regions[0].region_id})
    print(f"Table analysis result:\n{test_result[:500]}...")
else:
    print("No table regions found")
    
########################################
# 4.1 Formatting Context for the Agent #
########################################
def format_ordered_text(ordered_text, max_items=50):
    """Format ordered text for the system prompt."""
    lines = []
    for item in ordered_text[:max_items]:
        lines.append(f"[{item['position']}] {item['text']}")
    
    if len(ordered_text) > max_items:
        lines.append(f"... and {len(ordered_text) - max_items} more text regions")
    
    return "\n".join(lines)

def format_layout_regions(layout_regions):
    """Format layout regions for the system prompt."""
    lines = []
    for region in layout_regions:
        lines.append(f"  - Region {region.region_id}: {region.region_type} (confidence: {region.confidence:.3f})")
    return "\n".join(lines)

ordered_text_str = format_ordered_text(ordered_text)
layout_regions_str = format_layout_regions(layout_regions)

print("Formatted context for agent:")
print(f"- Ordered text: {len(ordered_text_str)} chars")
print(f"- Layout regions: {len(layout_regions_str)} chars")

##################################
# 4.2 Creating the System Prompt #
##################################
SYSTEM_PROMPT = f"""You are a Document Intelligence Agent. 
You analyze documents by combining OCR text with visual analysis tools.

## Document Text (in reading order)
The following text was extracted using OCR and ordered using LayoutLM.

{ordered_text_str}

## Document Layout Regions
The following regions were detected in the document:

{layout_regions_str}

## Your Tools
- **AnalyzeChart(region_id)**: 
    - Use for chart/figure regions to extract data points, axes, and trends
- **AnalyzeTable(region_id)**: 
    - Use for table regions to extract structured tabular data

## Instructions
1. For TEXT regions: 
    - Use the OCR text provided above (it's already extracted)
2. For TABLE regions: 
    - Use the AnalyzeTable tool to get structured data
3. For CHART/FIGURE regions: 
    - Use the AnalyzeChart tool to extract visual data

When answering questions about the document, 
use the appropriate tools to get accurate information.
"""

print("System prompt created")
print(f"Total length: {len(SYSTEM_PROMPT)} characters")

############################
# 4.3 Assembling the Agent #
############################
tools = [AnalyzeChart, AnalyzeTable]

agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents import create_tool_calling_agent
from langchain.agents import AgentExecutor

prompt = ChatPromptTemplate.from_messages(
    [
        ("system",SYSTEM_PROMPT),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(agent_llm, tools, prompt)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#########################
# 4.4 Testing the Agent #
#########################
response = agent_executor.invoke({
    "input": "What types of content are in this document?"
              "List the main sections.",
})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])

response = agent_executor.invoke({
    "input": "Extract the data from the table in this document." 
             "Return it in a structured format."})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])

response = agent_executor.invoke({
    "input": "Analyze the chart/figure in this document." 
    "What trends does it show?"})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])
