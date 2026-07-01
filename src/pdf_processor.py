from pathlib import Path
import shutil

import fitz  # PyMuPDF
from PIL import Image
import pytesseract


# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# 输入 PDF 文件夹
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "documents"

# 输出文件夹
PAGE_IMAGE_DIR = PROJECT_ROOT / "data" / "processed" / "pdf_pages"
EXTRACTED_TEXT_DIR = PROJECT_ROOT / "data" / "processed" / "extracted_text"


def setup_tesseract():
    """
    设置 Tesseract OCR 路径。
    如果 tesseract 已经加入 PATH，就直接使用。
    如果没有加入 PATH，就尝试使用 Windows 默认安装路径。
    """
    if shutil.which("tesseract"):
        return

    default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    if Path(default_path).exists():
        pytesseract.pytesseract.tesseract_cmd = default_path
    else:
        raise FileNotFoundError(
            "Cannot find tesseract.exe. Please install Tesseract OCR "
            "or add it to system PATH."
        )


def ocr_image(image_path: Path) -> str:
    """
    对图片进行 OCR。
    优先使用英文+简体中文。
    如果没有安装中文语言包，就自动退回英文。
    """
    image = Image.open(image_path)

    try:
        text = pytesseract.image_to_string(image, lang="eng+chi_sim")
    except pytesseract.TesseractError:
        text = pytesseract.image_to_string(image, lang="eng")

    return text.strip()


def process_pdf(pdf_path: Path):
    """
    处理单个 PDF：
    1. 抽取 PDF 原本文字
    2. 每页转成图片
    3. 对每页图片 OCR
    4. 保存每页的合并文本
    """
    print(f"Processing PDF: {pdf_path.name}")

    doc = fitz.open(pdf_path)

    for page_index, page in enumerate(doc):
        page_number = page_index + 1

        # 1. 提取 PDF 中可复制的文字
        pdf_text = page.get_text().strip()

        # 2. 把 PDF 当前页渲染成图片
        pix = page.get_pixmap(dpi=200)

        image_name = f"{pdf_path.stem}_page_{page_number}.png"
        image_path = PAGE_IMAGE_DIR / image_name
        pix.save(image_path)

        # 3. 对当前页图片做 OCR
        ocr_text = ocr_image(image_path)

        # 4. 合并 PDF text 和 OCR text
        combined_text = f"""
SOURCE_FILE: {pdf_path.name}
PAGE_NUMBER: {page_number}
PAGE_IMAGE: {image_path}

PDF_TEXT:
{pdf_text}

OCR_TEXT:
{ocr_text}
""".strip()

        # 5. 保存成 txt，后面 build_index.py 会读取这些 txt
        output_name = f"{pdf_path.stem}_page_{page_number}.txt"
        output_path = EXTRACTED_TEXT_DIR / output_name

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(combined_text)

        print(f"Saved page {page_number}: {output_path.name}")

    doc.close()


def process_all_pdfs():
    """
    处理 data/raw/documents/ 里的所有 PDF 文件。
    """
    setup_tesseract()

    PAGE_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    EXTRACTED_TEXT_DIR.mkdir(parents=True, exist_ok=True)

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {PDF_DIR}")
        return

    for pdf_path in pdf_files:
        process_pdf(pdf_path)

    print("\nAll PDF files processed successfully.")


if __name__ == "__main__":
    process_all_pdfs()