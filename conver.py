import os
import PyPDF2

def pdf_to_text(pdf_folder, text_folder):
    """
    Converte arquivos PDF em uma pasta para arquivos de texto e os salva em outra pasta.

    Args:
    pdf_folder (str): Caminho da pasta contendo os arquivos PDF.
    text_folder (str): Caminho da pasta onde os arquivos de texto serão salvos.
    """
    if not os.path.exists(text_folder):
        os.makedirs(text_folder)

    for pdf_file in os.listdir(pdf_folder):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, pdf_file)
            text_path = os.path.join(text_folder, pdf_file.replace(".pdf", ".txt"))

            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)

                with open(text_path, 'w', encoding='utf-8') as text_file:
                    for page_num in range(len(reader.pages)):
                        page = reader.pages[page_num]
                        text_file.write(page.extract_text())
                        text_file.write('\n')

# Exemplo de uso
pdf_folder = 'pdfs'  # Substitua com o caminho real
text_folder = 'txts' # Substitua com o caminho real
pdf_to_text(pdf_folder, text_folder)
