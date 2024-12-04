from flask import Flask, request, jsonify
import fitz
import cv2
import numpy as np
import networkx as nx
import os
import tempfile

app = Flask(__name__)

class SchematicConverter:
    def __init__(self):
        self.components = []
        self.connections = []
    
    def process_pdf(self, pdf_path):
        doc = fitz.open(pdf_path)
        # Process each page
        for page in doc:
            pix = page.get_pixmap()
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n)
            
            # Convert to grayscale for processing
            if pix.n == 4:  # RGBA
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
            elif pix.n == 3:  # RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
                
            # Detect components and connections
            self.detect_components(img)
            self.trace_connections(img)
            
        return self.generate_brd()
    
    def detect_components(self, img):
        # Apply threshold
        _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, 
                                     cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            # Analyze contour shape and size
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                self.components.append({
                    'type': self.classify_component(contour),
                    'position': (x, y),
                    'size': (w, h)
                })
    
    def trace_connections(self, img):
        # Edge detection
        edges = cv2.Canny(img, 50, 150)
        
        # HoughLines to detect straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, 
                               minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                self.connections.append({
                    'start': (x1, y1),
                    'end': (x2, y2)
                })
    
    def classify_component(self, contour):
        # Basic component classification based on shape
        return 'generic_component'
    
    def generate_brd(self):
        # Create a simple BRD format
        brd_content = "Version 1.0\n"
        
        # Add components
        for i, comp in enumerate(self.components):
            brd_content += f"Component{i} {comp['type']} "
            brd_content += f"{comp['position'][0]} {comp['position'][1]}\n"
        
        # Add connections
        for i, conn in enumerate(self.connections):
            brd_content += f"Connection{i} "
            brd_content += f"{conn['start'][0]} {conn['start'][1]} "
            brd_content += f"{conn['end'][0]} {conn['end'][1]}\n"
        
        return brd_content

@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Only PDF files are supported'}), 400
    
    # Save uploaded file temporarily
    temp_dir = tempfile.mkdtemp()
    temp_path = os.path.join(temp_dir, 'input.pdf')
    file.save(temp_path)
    
    try:
        converter = SchematicConverter()
        brd_content = converter.process_pdf(temp_path)
        
        # Clean up
        os.remove(temp_path)
        os.rmdir(temp_dir)
        
        return jsonify({
            'success': True,
            'brd_content': brd_content
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)