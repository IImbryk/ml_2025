import aiofiles

async def extract_txt(file_path: str) -> str:
    """Извлечение текста из TXT файла"""
    encodings = ['utf-8', 'cp1251', 'latin1', 'cp866']
    
    for encoding in encodings:
        try:
            async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                content = await f.read()
            return content
        except UnicodeDecodeError:
            continue
    
    raise Exception("Не удалось определить кодировку файла")

async def extract_docx(file_path: str) -> str:
    """Извлечение текста из DOCX файла"""
    try:
        import docx2txt
        content = docx2txt.process(file_path)
        return content
    except ImportError:
        raise Exception("Для DOCX файлов установи: pip install python-docx docx2txt")

async def extract_pdf(file_path: str) -> str:
    """Извлечение текста из PDF файла"""
    try:
        import PyPDF2
        
        content = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    content += f"--- Страница {page_num + 1} ---\n{page_text}\n\n"
        
        return content
        
    except ImportError:
        raise Exception("Для PDF файлов установи: pip install PyPDF2")
