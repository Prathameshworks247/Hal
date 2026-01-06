# Multimodal RAG Architecture

## Overview
Extension of the current RAG system to support multimodal queries and documents (text + images) with automatic OCR for scanned PDFs.

---

## Architecture Components

### 1. Document Ingestion Pipeline

#### 1.1 PDF Processing Flow
```
PDF Input
    ↓
[PDF Type Detector]
    ├──→ Native Text PDF → Direct Text Extraction
    └──→ Scanned PDF (Images) → OCR Pipeline
    └──→ Mixed PDF (Text + Images) → Hybrid Processing
```

#### 1.2 OCR Pipeline (for Scanned PDFs)
- **OCR Engine**: Tesseract OCR / EasyOCR / PaddleOCR
- **Preprocessing**: 
  - Image enhancement (denoising, contrast adjustment)
  - Page orientation detection and correction
  - Skew correction
- **Text Extraction**: Extract text with bounding box coordinates
- **Post-processing**: 
  - Text cleaning and validation
  - Layout analysis (columns, tables, headers)

#### 1.3 Image Extraction
- Extract images from PDF pages (diagrams, charts, photos)
- Store images with metadata:
  - Source page number
  - Bounding box coordinates
  - Image type (diagram, photo, chart, etc.)
  - Associated text context (surrounding text)

---

### 2. Multimodal Document Storage

#### 2.1 Document Chunk Structure
```
Document Chunk {
    chunk_id: str
    text_content: str
    images: List[ImageReference] {
        image_id: str
        image_path: str
        page_number: int
        coordinates: BoundingBox
        caption: str (if available)
    }
    metadata: {
        source_file: str
        page_number: int
        chunk_index: int
        document_type: str
        ocr_applied: bool
    }
}
```

#### 2.2 Storage Components
- **Vector Store (FAISS)**: 
  - Text embeddings (existing)
  - Image embeddings (new - using vision-language model)
- **File System**: 
  - Store extracted images in organized directories
  - Path structure: `uploads/images/{pb_number}/{document_id}/{page}_{image_index}.jpg`
- **Metadata Database**: 
  - Link text chunks to images
  - Store OCR results with confidence scores
  - Track document processing status

---

### 3. Multimodal Embedding System

#### 3.1 Text Embeddings (Existing)
- **Model**: all-MiniLM-L6-v2 (existing)
- **Usage**: Text chunks, OCR-extracted text

#### 3.2 Image Embeddings (New)
- **Model Options**:
  - CLIP (OpenAI) - Vision-Language model
  - BLIP (Salesforce) - Image understanding
  - MiniCPM-V / Qwen-VL - Multimodal LLM embeddings
- **Embedding Strategy**:
  - Extract features from images
  - Generate embeddings compatible with text embeddings
  - Optional: Use vision-language model for image descriptions

#### 3.3 Unified Embedding Space
- **Approach 1**: Use vision-language model (CLIP) that generates embeddings in same space for both text and images
- **Approach 2**: Project image embeddings to text embedding space using learned mapping
- **Approach 3**: Maintain separate indices, combine results during retrieval

---

### 4. Multimodal Retrieval System

#### 4.1 Query Processing
```
User Query Input
    ↓
[Query Type Detector]
    ├──→ Text Query → Text Embedding → Text Retrieval
    ├──→ Image Query → Image Embedding → Image Retrieval
    └──→ Mixed Query → Separate Embeddings → Hybrid Retrieval
```

#### 4.2 Retrieval Strategies

**Text-Only Query**:
1. Generate text embedding
2. Retrieve from text vector store (FAISS)
3. Include associated images from retrieved chunks

**Image Query**:
1. Generate image embedding
2. Retrieve similar images from image vector store
3. Retrieve associated text chunks from same pages

**Mixed Query (Text + Image)**:
1. Generate separate embeddings for text and image
2. Retrieve from both indices
3. Combine and re-rank results
4. Weight: 70% text relevance, 30% image relevance (configurable)

#### 4.3 Result Fusion
- **Score Fusion**: Combine text and image similarity scores
- **Deduplication**: Remove duplicate chunks from same document
- **Reranking**: Use cross-modal model to rerank results
- **Top-K Selection**: Return top-K most relevant chunks with images

---

### 5. OCR Service Architecture

#### 5.1 OCR Detection
- **Method**: Analyze PDF structure
  - Check for text layers
  - Detect image-only pages
  - Use heuristics (page count, file size, metadata)

#### 5.2 OCR Processing Service
```
OCR Service {
    - Extract images from PDF pages
    - Preprocess images (enhance quality)
    - Run OCR on each page
    - Post-process text (correct common errors)
    - Extract layout structure
    - Return: {
        text: str
        confidence_scores: List[float]
        bounding_boxes: List[BoundingBox]
        page_number: int
    }
}
```

#### 5.3 OCR Configuration
- **Language**: Support multiple languages (English, Hindi, etc.)
- **OCR Engine Selection**: 
  - Tesseract (good for standard text)
  - EasyOCR (better for complex layouts)
  - PaddleOCR (better for Asian languages)
- **Confidence Threshold**: Filter low-confidence OCR results

---

### 6. API Extensions

#### 6.1 New Endpoints

**POST /user/ingest-multimodal**
- Accept PDF files
- Automatically detect and apply OCR if needed
- Extract and store images
- Generate embeddings for both text and images
- Return processing status and statistics

**POST /user/query-multimodal**
- Accept text query, image query, or both
- Support query types:
  - Text only
  - Image upload
  - Mixed (text + image)
- Return: Response with relevant text chunks AND associated images

**POST /admin/ocr/batch-process**
- Batch process multiple PDFs with OCR
- Queue-based processing for large volumes
- Progress tracking and status updates

#### 6.2 Enhanced Existing Endpoints

**POST /user/rectify** (Enhanced)
- Accept optional image alongside text query
- Retrieve relevant images in response
- Include image references in citations

---

### 7. Data Flow

#### 7.1 Document Ingestion Flow
```
1. User uploads PDF
   ↓
2. PDF Type Detection
   ↓
3a. If Scanned → OCR Service → Extract Text
3b. If Native → Direct Text Extraction
3c. Extract Images (both cases)
   ↓
4. Chunk Documents (text + images)
   ↓
5. Generate Embeddings
   - Text embeddings (all-MiniLM-L6-v2)
   - Image embeddings (CLIP or similar)
   ↓
6. Store in Vector DB
   - Text chunks → FAISS text index
   - Images → FAISS image index
   - Metadata → Link text and images
   ↓
7. Return ingestion status
```

#### 7.2 Query Processing Flow
```
1. User submits query (text/image/both)
   ↓
2. Query Type Detection
   ↓
3. Generate Embeddings
   - Text query → text embedding
   - Image query → image embedding
   ↓
4. Multimodal Retrieval
   - Query text index (if text query)
   - Query image index (if image query)
   - Combine results
   ↓
5. Result Fusion & Reranking
   ↓
6. Retrieve Full Context
   - Text chunks
   - Associated images
   - Metadata
   ↓
7. LLM Generation
   - Include text context
   - Include image references
   - Generate response with citations
   ↓
8. Return Response
   - Text answer
   - Relevant images (URLs or base64)
   - Citations (text + images)
```

---

### 8. Technology Stack

#### 8.1 OCR Libraries
- **Primary**: Tesseract OCR (via pytesseract)
- **Alternatives**: EasyOCR, PaddleOCR
- **Preprocessing**: OpenCV, PIL/Pillow

#### 8.2 Vision Models
- **Image Embeddings**: CLIP (openai/clip-vit-base-patch32)
- **Image Understanding**: BLIP-2 or Qwen-VL
- **Image Generation**: Not needed for RAG

#### 8.3 Image Processing
- **PDF Processing**: PyMuPDF (fitz), pdf2image
- **Image Handling**: PIL/Pillow, OpenCV
- **Image Storage**: Local filesystem (or S3 for production)

#### 8.4 Embedding Infrastructure
- **Text Embeddings**: HuggingFace Transformers (existing)
- **Vision Embeddings**: CLIP from HuggingFace or OpenAI
- **Vector Store**: FAISS (extend existing)
  - Separate indices for text and images
  - Or unified index with type metadata

---

### 9. File Structure

```
services/
├── ocr_service.py          # OCR processing logic
├── image_extractor.py      # Extract images from PDFs
├── multimodal_embedder.py  # Generate image embeddings
├── multimodal_retriever.py # Multimodal retrieval logic
└── multimodal_ingest.py    # Multimodal ingestion pipeline

utils/
├── ocr_detector.py         # Detect if PDF needs OCR
├── image_processor.py      # Image preprocessing
└── pdf_analyzer.py         # Analyze PDF structure

uploads/
├── images/                 # Extracted images
│   └── {pb_number}/
│       └── {doc_id}/
│           ├── page_1_img_0.jpg
│           └── page_2_img_1.jpg
└── pdfs/                   # Original PDFs (existing)
```

---

### 10. Implementation Phases

#### Phase 1: OCR Integration
- [ ] Add OCR detection for scanned PDFs
- [ ] Implement OCR service with Tesseract
- [ ] Test OCR accuracy on sample documents
- [ ] Store OCR results in existing chunk structure

#### Phase 2: Image Extraction
- [ ] Extract images from PDFs
- [ ] Store images with metadata
- [ ] Link images to text chunks
- [ ] Create image management utilities

#### Phase 3: Image Embeddings
- [ ] Integrate CLIP or similar vision model
- [ ] Generate image embeddings
- [ ] Create image vector index (FAISS)
- [ ] Test image similarity search

#### Phase 4: Multimodal Retrieval
- [ ] Implement hybrid retrieval (text + image)
- [ ] Result fusion and reranking
- [ ] Test retrieval accuracy
- [ ] Optimize retrieval performance

#### Phase 5: API Integration
- [ ] Extend ingestion endpoints for multimodal
- [ ] Create multimodal query endpoint
- [ ] Return images in response
- [ ] Update frontend to display images

---

### 11. Performance Considerations

#### 11.1 OCR Performance
- **Caching**: Cache OCR results to avoid reprocessing
- **Parallelization**: Process multiple pages in parallel
- **Queue System**: Use background tasks for large documents
- **Progress Tracking**: Report OCR progress for long documents

#### 11.2 Image Storage
- **Compression**: Compress images before storage
- **Thumbnails**: Generate thumbnails for quick preview
- **CDN**: Use CDN for image delivery in production
- **Cleanup**: Implement image cleanup for deleted documents

#### 11.3 Embedding Generation
- **Batch Processing**: Generate embeddings in batches
- **GPU Acceleration**: Use GPU for image embeddings if available
- **Caching**: Cache embeddings to avoid recomputation

#### 11.4 Retrieval Performance
- **Index Optimization**: Optimize FAISS indices for faster search
- **Hybrid Search**: Balance text and image retrieval times
- **Result Limiting**: Limit image results to prevent slow responses

---

### 12. Error Handling

#### 12.1 OCR Failures
- **Low Confidence**: Flag low-confidence OCR results
- **Failed Pages**: Skip problematic pages, continue processing
- **Fallback**: Provide manual OCR option for failed documents

#### 12.2 Image Extraction Failures
- **Corrupted Images**: Skip corrupted images, log errors
- **Large Images**: Resize or compress oversized images
- **Format Issues**: Handle various image formats gracefully

#### 12.3 Embedding Failures
- **Model Errors**: Fallback to text-only if image embedding fails
- **Timeout Handling**: Set timeouts for embedding generation
- **Resource Management**: Handle GPU memory issues

---

### 13. Testing Strategy

#### 13.1 OCR Testing
- Test on various scanned PDF types
- Test multiple languages
- Test on poor quality scans
- Measure OCR accuracy

#### 13.2 Multimodal Retrieval Testing
- Test text-only queries
- Test image queries
- Test mixed queries
- Measure retrieval precision/recall
- Test with various document types

#### 13.3 Integration Testing
- End-to-end ingestion flow
- End-to-end query flow
- Test error scenarios
- Performance benchmarking

---

### 14. Future Enhancements

- **Advanced OCR**: Handwriting recognition, table extraction
- **Image Analysis**: Object detection in images, diagram understanding
- **Video Support**: Extract frames from video documents
- **3D Models**: Support 3D CAD models from technical documents
- **Real-time OCR**: Stream OCR results as document is processed
- **Custom OCR Models**: Train custom OCR for domain-specific documents

