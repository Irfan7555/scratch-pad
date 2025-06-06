import streamlit as st
import fitz  # PyMuPDF
import easyocr
import os
from tempfile import TemporaryDirectory
import time

st.set_page_config(page_title="PDF OCR with EasyOCR", layout="centered")
st.title("📄 PDF OCR (English + Arabic) with EasyOCR")

uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_pdf:
    st.info("⏳ OCR is in progress... please wait.")

    start_time = time.time()
    reader = easyocr.Reader(['en', 'ar'], gpu=False)

    with TemporaryDirectory() as temp_dir:
        pdf_path = os.path.join(temp_dir, uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

        doc = fitz.open(pdf_path)
        zoom = 2
        mat = fitz.Matrix(zoom, zoom)

        all_text = ""
        max_pages = min(10, len(doc))

        progress_bar = st.progress(0, text="Processing pages...")
        for i, page in enumerate(doc):
            if i >= max_pages:
                break

            pix = page.get_pixmap(matrix=mat)
            image_path = os.path.join(temp_dir, f"page_{i+1}.png")
            pix.save(image_path)

            result = reader.readtext(image_path, detail=0)
            page_text = "\n".join(result)
            all_text += f"--- Page {i+1} ---\n{page_text}\n\n"

            progress_bar.progress((i + 1) / max_pages, text=f"Processing page {i + 1}/{max_pages}...")

        doc.close()
        progress_bar.empty()

        time_taken = round(time.time() - start_time, 2)
        st.success(f"OCR Completed in {time_taken} seconds ✅")

        st.subheader("📃 Extracted Text:")
        st.text_area("Text Output", value=all_text.strip(), height=300)

        # ✅ Create .txt file INSIDE the temp directory block
        txt_path = os.path.join(temp_dir, "ocr_result.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(all_text.strip())

        with open(txt_path, "rb") as f:
            st.download_button(
                label="📥 Download OCR Text as .txt",
                data=f.read(),
                file_name="ocr_result.txt",
                mime="text/plain"
            )
