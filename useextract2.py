# requirements.txt
"""
pdfplumber>=0.7.0
pandas>=1.3.0
pdf2image>=3.1.0
pytesseract>=0.3.10
Pillow>=9.0.0
"""

# simple_usage_example.py
"""
Simple example of how to use the Rocfall coordinate extractor.
"""

from robust_rocfall_extractor import RocfallCoordinateExtractor
import pandas as pd

def main():
    # Initialize the extractor
    pdf_path = "C:\\Users\\eisas\\Downloads\\DefaultMergeableReport.pdf" # Your PDF file path
    output_dir = "extracted_data"           # Output directory
    
    extractor = RocfallCoordinateExtractor(pdf_path, output_dir)
    
    # Run the extraction
    if extractor.run_extraction():
        print("✅ Extraction successful!")
        
        # Read and display the results
        try:
            # Line Seeder coordinates
            line_df = pd.read_csv(f"{output_dir}/rocfall_line_seeder_coords.csv")
            print(f"\n📍 Line Seeder Coordinates ({len(line_df)} points):")
            print(line_df.head())
            
            # Point Seeder coordinates  
            try:
                point_df = pd.read_csv(f"{output_dir}/rocfall_point_seeder_coords.csv")
                print(f"\n📍 Point Seeder Coordinates ({len(point_df)} points):")
                print(point_df.head())
            except FileNotFoundError:
                print("\n⚠️  No Point Seeder coordinates file found")
            
            # Path results
            try:
                path_df = pd.read_csv(f"{output_dir}/rocfall_path_results.csv")
                print(f"\n📊 Path Results ({len(path_df)} results):")
                print(path_df.head())
            except FileNotFoundError:
                print("\n⚠️  No path results file found")
                
        except FileNotFoundError as e:
            print(f"❌ Could not read results: {e}")
    else:
        print("❌ Extraction failed")

if __name__ == "__main__":
    main()