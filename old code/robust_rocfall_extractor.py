# robust_rocfall_extractor.py
# Extracts Line Seeder and Point Seeder coordinates from Rocfall3 PDF reports
# Handles both text extraction and OCR fallback
# Requires: pdfplumber, pandas, pdf2image, pytesseract, Pillow

import re
import sys
from pathlib import Path
import pandas as pd
import pdfplumber
from typing import List, Tuple, Optional
import logging

# OCR imports (will be imported only if needed)
try:
    from pdf2image import convert_from_path
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("OCR libraries not available. Install with: pip install pdf2image pytesseract pillow")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class RocfallCoordinateExtractor:
    def __init__(self, pdf_path: str, output_dir: str = "extracted_data"):
        self.pdf_path = Path(pdf_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    def extract_text_from_pdf(self) -> str:
        """Extract all text from PDF using pdfplumber."""
        try:
            with pdfplumber.open(str(self.pdf_path)) as pdf:
                pages_text = []
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        pages_text.append(f"--- PAGE {page_num + 1} ---\n{text}")
                    else:
                        logger.warning(f"No text found on page {page_num + 1}")
                
                full_text = "\n".join(pages_text)
                logger.info(f"Extracted {len(full_text)} characters from PDF")
                return full_text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def parse_line_seeder_coordinates(self, text: str) -> List[Tuple[int, float, float, float]]:
        """
        Extract Line Seeder 1 coordinates from text.
        Handles multiple formatting patterns.
        """
        logger.info("Parsing Line Seeder coordinates...")
        
        # Find the Line Seeder 1 section
        patterns = [
            r"Line Seeder 1.*?Point X Y Z(.*?)(?=Point Seeder|Run Properties|Summary Results|$)",
            r"Line Seeder 1.*?Point\s+X\s+Y\s+Z(.*?)(?=Point Seeder|Run Properties|Summary Results|$)",
            r"Line Seeder 1(.*?)(?=Point Seeder 1|Run Properties|Summary Results|$)"
        ]
        
        seeder_section = None
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                seeder_section = match.group(1)
                logger.info(f"Found Line Seeder section using pattern: {pattern[:30]}...")
                break
        
        if not seeder_section:
            logger.warning("Could not find Line Seeder 1 section")
            return []
        
        # Extract coordinate rows with multiple patterns
        coordinate_patterns = [
            # Pattern: "0: 160067.008 3058765.402 2124.319"
            r'(\d+):\s*([-+]?\d+(?:\.\d+)?)\s*([-+]?\d+(?:\.\d+)?)\s*([-+]?\d+(?:\.\d+)?)',
            # Pattern: "0 160067.008 3058765.402 2124.319"
            r'(\d+)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)\s+([-+]?\d+(?:\.\d+)?)',
            # Pattern with more flexible spacing
            r'(\d+)[\s:]+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'
        ]
        
        coordinates = []
        for pattern in coordinate_patterns:
            matches = re.findall(pattern, seeder_section, re.MULTILINE)
            if matches:
                logger.info(f"Found {len(matches)} coordinate matches with pattern")
                for match in matches:
                    try:
                        idx = int(match[0])
                        x = float(match[1])
                        y = float(match[2])
                        z = float(match[3])
                        
                        # Sanity check for reasonable coordinate values
                        if self._is_valid_coordinate(x, y, z):
                            coordinates.append((idx, x, y, z))
                    except (ValueError, IndexError) as e:
                        logger.debug(f"Skipping invalid coordinate: {match}, error: {e}")
                
                if coordinates:
                    break  # Use first pattern that gives results
        
        # Remove duplicates and sort
        coordinates = list(set(coordinates))
        coordinates.sort(key=lambda x: x[0])
        
        logger.info(f"Extracted {len(coordinates)} Line Seeder coordinates")
        return coordinates
    
    def parse_point_seeder_coordinates(self, text: str) -> List[Tuple[int, float, float, float]]:
        """Extract Point Seeder coordinates from text."""
        logger.info("Parsing Point Seeder coordinates...")
        
        # Find Point Seeder section
        patterns = [
            r"Point Seeder 1.*?Point X Y Z(.*?)(?=Run Properties|Summary Results|$)",
            r"Point Seeder 1.*?Point\s+X\s+Y\s+Z(.*?)(?=Run Properties|Summary Results|$)",
            r"Point Seeder 1(.*?)(?=Run Properties|Summary Results|$)"
        ]
        
        seeder_section = None
        for pattern in patterns:
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                seeder_section = match.group(1)
                break
        
        if not seeder_section:
            logger.warning("Could not find Point Seeder 1 section")
            return []
        
        # Extract coordinates
        coordinates = []
        coordinate_pattern = r'(\d+):\s*([-+]?\d+(?:\.\d+)?)\s*([-+]?\d+(?:\.\d+)?)\s*([-+]?\d+(?:\.\d+)?)'
        matches = re.findall(coordinate_pattern, seeder_section, re.MULTILINE)
        
        for match in matches:
            try:
                idx = int(match[0])
                x = float(match[1])
                y = float(match[2])
                z = float(match[3])
                
                if self._is_valid_coordinate(x, y, z):
                    coordinates.append((idx, x, y, z))
            except (ValueError, IndexError):
                continue
        
        coordinates.sort(key=lambda x: x[0])
        logger.info(f"Extracted {len(coordinates)} Point Seeder coordinates")
        return coordinates
    
    def _is_valid_coordinate(self, x: float, y: float, z: float) -> bool:
        """Basic validation for coordinate values."""
        # Assuming UTM-like coordinates for x,y and reasonable elevation for z
        return (1e4 < abs(x) < 1e8 and 1e4 < abs(y) < 1e8 and 100 < abs(z) < 10000)
    
    def extract_coordinates_ocr(self) -> Tuple[List[Tuple[int, float, float, float]], List[Tuple[int, float, float, float]]]:
        """
        Extract coordinates using OCR as fallback.
        Returns (line_coordinates, point_coordinates)
        """
        if not OCR_AVAILABLE:
            logger.error("OCR libraries not available")
            return [], []
        
        logger.info("Using OCR to extract coordinates...")
        
        try:
            # Convert PDF to images
            images = convert_from_path(str(self.pdf_path), dpi=300)
            logger.info(f"Converted PDF to {len(images)} images")
            
            all_text = ""
            for i, image in enumerate(images):
                # Use pytesseract to extract text
                text = pytesseract.image_to_string(image, config='--psm 6')
                all_text += f"\n--- PAGE {i+1} OCR ---\n{text}"
                logger.info(f"OCR extracted {len(text)} characters from page {i+1}")
            
            # Parse coordinates from OCR text
            line_coords = self.parse_line_seeder_coordinates(all_text)
            point_coords = self.parse_point_seeder_coordinates(all_text)
            
            return line_coords, point_coords
            
        except Exception as e:
            logger.error(f"OCR extraction failed: {e}")
            return [], []
    
    def extract_path_results(self, text: str) -> pd.DataFrame:
        """Extract path results from text."""
        logger.info("Extracting path results...")
        
        # Pattern for path results: "ID: travel_time events stopping_reason runout"
        pattern = r'^\s*(\d+)\s*:\s*([0-9]+\.[0-9]+)\s+(\d+)\s+([A-Za-z]+)\s+([0-9]+\.[0-9]+)'
        matches = re.findall(pattern, text, re.MULTILINE)
        
        if not matches:
            logger.warning("No path results found")
            return pd.DataFrame()
        
        df = pd.DataFrame(matches, columns=["id", "travel_time_s", "events", "stopping_reason", "runout_m"])
        df = df.astype({"id": int, "travel_time_s": float, "events": int, "runout_m": float})
        df = df.drop_duplicates(subset="id").sort_values("id").reset_index(drop=True)
        
        logger.info(f"Extracted {len(df)} path results")
        return df
    
    def run_extraction(self) -> bool:
        """
        Main extraction method.
        Returns True if successful, False otherwise.
        """
        logger.info(f"Starting extraction from {self.pdf_path}")
        
        # Step 1: Try text extraction
        text = self.extract_text_from_pdf()
        
        if not text:
            logger.error("No text could be extracted from PDF")
            return False
        
        # Save extracted text for debugging
        with open(self.output_dir / "extracted_text_debug.txt", 'w', encoding='utf-8') as f:
            f.write(text)
        logger.info("Saved extracted text to extracted_text_debug.txt")
        
        # Extract coordinates from text
        line_coords = self.parse_line_seeder_coordinates(text)
        point_coords = self.parse_point_seeder_coordinates(text)
        
        # Step 2: If no coordinates found, try OCR
        if not line_coords and not point_coords:
            logger.warning("No coordinates found in text extraction, trying OCR...")
            line_coords, point_coords = self.extract_coordinates_ocr()
        
        # Step 3: Extract path results
        path_df = self.extract_path_results(text)
        
        # Step 4: Save results
        success = False
        
        if line_coords:
            df_line = pd.DataFrame(line_coords, columns=["index", "x", "y", "z"])
            output_file = self.output_dir / "rocfall_line_seeder_coords.csv"
            df_line.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df_line)} Line Seeder coordinates to {output_file}")
            success = True
        else:
            logger.warning("No Line Seeder coordinates found")
        
        if point_coords:
            df_point = pd.DataFrame(point_coords, columns=["index", "x", "y", "z"])
            output_file = self.output_dir / "rocfall_point_seeder_coords.csv"
            df_point.to_csv(output_file, index=False)
            logger.info(f"Saved {len(df_point)} Point Seeder coordinates to {output_file}")
            success = True
        else:
            logger.warning("No Point Seeder coordinates found")
        
        if not path_df.empty:
            output_file = self.output_dir / "rocfall_path_results.csv"
            path_df.to_csv(output_file, index=False)
            logger.info(f"Saved {len(path_df)} path results to {output_file}")
            success = True
        
        return success


def main():
    # Configuration
    PDF_PATH = "DefaultMergeableReport.pdf"  # Update this path as needed
    OUTPUT_DIR = "extracted_data"
    
    if len(sys.argv) > 1:
        PDF_PATH = sys.argv[1]
    if len(sys.argv) > 2:
        OUTPUT_DIR = sys.argv[2]
    
    try:
        extractor = RocfallCoordinateExtractor(PDF_PATH, OUTPUT_DIR)
        success = extractor.run_extraction()
        
        if success:
            logger.info("Extraction completed successfully!")
            logger.info(f"Check the '{OUTPUT_DIR}' directory for output files.")
        else:
            logger.error("Extraction failed - no data could be extracted")
            return 1
            
    except Exception as e:
        logger.error(f"Extraction failed with error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())