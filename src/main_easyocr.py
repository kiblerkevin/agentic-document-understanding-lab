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

# Use EasyOCR - simple, no LangChain conflicts
import easyocr

# Initialize EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

# Load Image
image_path = "report_original.png"

###########################
# 1.1 Run OCR on Document #
###########################
# EasyOCR returns: [bbox, text, confidence]
result = reader.readtext(image_path)

texts = [item[1] for item in result]
scores = [item[2] for item in result]
boxes = [item[0] for item in result]

print(f"Extracted {len(texts)} text regions")
print("\nFirst 10 regions:")
for text, score, box in list(zip(texts, scores, boxes))[:10]:
    print(f"{text:40} | {score:.3f} | {box}")

######################################
# 1.2 Visualizing OCR Bounding Boxes #
######################################
img = cv2.imread(image_path)
img_plot = img.copy()
show_text = False

for text, box in zip(texts, boxes):
    pts = np.array(box, dtype=int)
    cv2.polylines(img_plot, [pts], True, (0, 255, 0), 2)
    x, y = pts[0]
    if show_text:
        cv2.putText(img_plot, text, (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

plt.figure(figsize=(8, 10))
plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.title("EasyOCR Bounding Boxes")
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
for text, score, box in zip(texts, scores, boxes):
    ocr_regions.append(OCRRegion(
        text=text, 
        bbox=[[int(p[0]), int(p[1])] for p in box], 
        confidence=score
    ))

print(f"Stored {len(ocr_regions)} OCR regions")

###############################
# 1.4 Layout LM Reading Order #
###############################
from transformers import LayoutLMv3ForTokenClassification
from helpers import prepare_inputs, boxes2inputs, parse_logits

# Load LayoutReader model
print("Loading LayoutReader model...")
model_slug = "hantian/layoutreader"
layout_model = LayoutLMv3ForTokenClassification.from_pretrained(model_slug)
print("Model loaded successfully!")

def get_reading_order(ocr_regions):
    """
    Use LayoutReader to determine reading order of OCR regions.
    Returns list of reading order positions for each region index.
    """
    # 1. Calculate image dimensions from bounding boxes (with padding)
    max_x = max_y = 0
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    image_width = max_x * 1.1   # Add 10% padding
    image_height = max_y * 1.1

    # 2. Convert bboxes to LayoutReader format (normalized to 0-1000)
    boxes = []
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox_xyxy
        # Normalize to 0-1000 range
        left = int((x1 / image_width) * 1000)
        top = int((y1 / image_height) * 1000)
        right = int((x2 / image_width) * 1000)
        bottom = int((y2 / image_height) * 1000)
        boxes.append([left, top, right, bottom])

    # 3. Prepare inputs
    inputs = boxes2inputs(boxes)
    inputs = prepare_inputs(inputs, layout_model)
    
    # 4. Run inference
    logits = layout_model(**inputs).logits.cpu().squeeze(0)
    
    # 5. Parse the model's outputs to get reading order
    reading_order = parse_logits(logits, len(boxes))

    return reading_order

# Get reading order
reading_order = get_reading_order(ocr_regions)

print(f"Reading order determined for {len(reading_order)} regions")
print(f"First 20 positions: {reading_order[:20]}")

#####################################
# 1.5 Visualizing the Reading Order #
#####################################
import matplotlib.patches as patches

def visualize_reading_order(ocr_regions, image_path, reading_order, title="Reading Order"):
    """
    Visualize OCR regions with their reading order numbers using matplotlib.
    """
    img = cv2.imread(image_path)
    
    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Create order mapping: index -> reading order position
    order_map = {i: order for i, order in enumerate(reading_order)}
    
    for i, region in enumerate(ocr_regions):
        bbox = region.bbox
        if bbox and len(bbox) >= 4:
            # Draw polygon
            ax.add_patch(patches.Polygon(bbox, linewidth=2, 
                                         edgecolor='blue',
                                         facecolor='none', alpha=0.7))
            # Add reading order number at center
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
    """
    Return OCR regions sorted by reading order
    with their text and confidence.
    """
    # 1. Create (reading_position, index, region) tuples and sort
    indexed_regions = [(reading_order[i], 
                        i, 
                        ocr_regions[i]) for i in range(len(ocr_regions))]
    
    # 2. Sort by reading position
    indexed_regions.sort(key=lambda x: x[0])  
    
    # 3. Extract ordered text info
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
# Simple layout detection using bounding box analysis
def process_document(ocr_regions, image_path):
    """Get layout regions from OCR bounding boxes."""
    img = cv2.imread(image_path)
    img_height, img_width = img.shape[:2]
    
    regions = []
    
    for i, region in enumerate(ocr_regions):
        x1, y1, x2, y2 = region.bbox_xyxy
        width = x2 - x1
        height = y2 - y1
        area = width * height
        
        # Simple heuristic classification
        aspect_ratio = width / height if height > 0 else 0
        
        # Classify based on size and position
        if aspect_ratio > 3 and y1 < img_height * 0.2:
            label = 'title'
        elif area > img_width * img_height * 0.1:
            label = 'figure'
        elif aspect_ratio > 2:
            label = 'table'
        else:
            label = 'text'
        
        regions.append({
            'label': label,
            'score': region.confidence,
            'bbox': [x1, y1, x2, y2],
        })
    
    return regions

layout_results = process_document(ocr_regions, image_path)

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
    bbox: list  # [x1, y1, x2, y2]
    confidence: float
    
# Store layout regions in structured format
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

def visualize_layout(image_path, layout_regions, min_confidence=0.5, 
                     title="Layout Detection"):
    """
    Visualize layout detection results using cv2.
    """
    img = cv2.imread(image_path)
    img_plot = img.copy()
    
    # Get unique labels and generate colors
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
        
        # Draw rectangle
        pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=int)
        cv2.polylines(img_plot, [pts], True, color, 2)
        
        # Add label
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
                 min_confidence=0.5, title="EasyOCR Layout Detection")

########################################
# 2.4 Cropping Regions for Agent Tools #
########################################
import base64
from io import BytesIO

# Crop and save layout regions for agent tools
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

# Load image for cropping
pil_image = Image.open(image_path)

# Store cropped regions in dictionary
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

# Also store full image
full_image_base64 = image_to_base64(pil_image)

# Show cropped regions
fig, axes = plt.subplots(5, 3, figsize=(15, 10))
axes = axes.flatten()

for i, (region_id, data) in enumerate(list(region_images.items())[:15]):
    if i < len(axes):
        axes[i].imshow(data['image'])
        axes[i].set_title(f"Region {region_id}: {data['type']}")
        axes[i].axis('off')

# Hide unused subplots
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

# Initialize VLM for tools
vlm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Tool prompts
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
# Test the tools
print("Testing AnalyzeChart...")
chart_regions = [r for r in layout_regions if r.region_type in ['chart', 'figure']]
if chart_regions:
    test_result = AnalyzeChart.invoke({"region_id": chart_regions[0].region_id})
    print(f"Chart analysis result:\n{test_result[:500]}...")
else:
    print("No chart regions found")
    
# Test table tool
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
#Prepare context for the agent
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

# Create the formatted strings
ordered_text_str = format_ordered_text(ordered_text)
layout_regions_str = format_layout_regions(layout_regions)

print("Formatted context for agent:")
print(f"- Ordered text: {len(ordered_text_str)} chars")
print(f"- Layout regions: {len(layout_regions_str)} chars")

##################################
# 4.2 Creating the System Prompt #
##################################

# System prompt for the agent
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
# Initialize the agent (using LangChain 0.1.x API)
tools = [AnalyzeChart, AnalyzeTable]

# LLM for the agent 
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


# 4. Create the tool-calling agent
agent = create_tool_calling_agent(agent_llm, tools, prompt)

# 5. Set up the AgentExecutor to run the tool-enabled loop
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#########################
# 4.4 Testing the Agent #
#########################
# Test the agent with a simple question
response = agent_executor.invoke({
    "input": "What types of content are in this document?"
              "List the main sections.",
})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])

# Test with table extraction
response = agent_executor.invoke({
    "input": "Extract the data from the table in this document." 
             "Return it in a structured format."})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])

# Test with chart analysis
response = agent_executor.invoke({
    "input": "Analyze the chart/figure in this document." 
    "What trends does it show?"})
print("\n" + "="*60)
print("Agent Response:")
print("="*60)
print(response["output"])
