import os
import shutil
import sys
# Make sure src is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.pdf_generator import convert_report_to_pdf

def test_pdf_fallback_plots():
    print("Testing PDF Generator Fallback Plot Discovery...")
    
    # 1. Setup Environment
    plots_dir = "static/plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # Create a dummy plot file
    dummy_plot = os.path.join(plots_dir, "test_fallback_plot.png")
    # Write some random bytes to simulate an image (valid enough for file existence, 
    # though xhtml2pdf might complain if it's not a real image, we just need the path logic to work.
    # Actually, to be safe, let's copy a real image if we have one, or use a tiny valid 1x1 png base64 to avoid "Image Not Found" red text if xhtml2pdf verifies header).
    # For now, let's try a real generic 1x1 png file creation.
    
    # Minimal 1x1 transparent PNG
    with open(dummy_plot, "wb") as f:
        f.write(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82')
        
    output_pdf = "test_output.pdf"
    if os.path.exists(output_pdf):
        os.remove(output_pdf)
        
    try:
        # 2. Convert Report with NO images in markdown
        markdown_content = "# Test Report\n\nNo manual images here."
        success = convert_report_to_pdf(markdown_content, output_pdf)
        
        if not success:
            print("❌ Convert function returned False.")
            return False
            
        # 3. Verify PDF content
        with open(output_pdf, "rb") as f:
            pdf_bytes = f.read()
            
        # Check for Image Subtype or the filename in the stream (binary check is hard for filename, but text content might be there)
        # xhtml2pdf usually embeds images.
        # We look for /Subtype /Image
        
        if b"/Subtype /Image" in pdf_bytes:
            print("✅ '/Subtype /Image' found in PDF. Image was embedded.")
        else:
            print("❌ '/Subtype /Image' NOT found in PDF. Fallback might have failed.")
            return False
            
        # Optional: Check if the alt text "test_fallback_plot.png" appears (it's inserted as <p>)
        # PDF compression might hide it, but usually text is readable unless stream compressed. 
        # But we rely on the byte check as requested.
        
        return True
        
    except Exception as e:
        print(f"❌ Exception during test: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        if os.path.exists(dummy_plot):
            os.remove(dummy_plot)
        if os.path.exists(output_pdf):
            os.remove(output_pdf)

if __name__ == "__main__":
    if test_pdf_fallback_plots():
        print("\nPASS: PDF Logic Verified.")
        sys.exit(0)
    else:
        print("\nFAIL: PDF Logic Failed.")
        sys.exit(1)
