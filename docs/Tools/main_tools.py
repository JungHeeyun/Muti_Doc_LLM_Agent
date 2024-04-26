from langchain.pydantic_v1 import BaseModel, Field
from langchain.tools import StructuredTool
from Tools.__init__ import process_pdf_image_query, rag, csv

def init_tools(file_index, faiss_index):

    class SearchDoc(BaseModel):
        index: int = Field(description="Index number of the doc that you want to search for")
        query: str = Field(description="keyword or concept in words only")

    def searchdocument(index: int, query: str) -> str:
        index = faiss_index.get(index)
        return rag(index, query)

    doc_search = StructuredTool.from_function(
        func=searchdocument,
        name="SearchPDF",
        description="search about some specific key concepts or words in the specific pdf that the user ask about",
        args_schema=SearchDoc,
        return_direct=False,
    )

    class PDFPage(BaseModel):
        pdf_index: int = Field(description="index of the pdf that the user asked for")
        page_number: int = Field(description="pdf's page number that user asked about")
        query: str = Field(description="user query about this specific page of this pdf file.")

    def pdf_page_view(pdf_index: int, page_number: int, query: str) -> str:        
        pdf_path = file_index.get(pdf_index)
        return process_pdf_image_query(pdf_path, page_number, query)

    pdf_analyzer = StructuredTool.from_function(
        func=pdf_page_view,
        name="PDFPage_Analyzer",
        description="provide detailed information about specific page of the specific pdf doc.",
        args_schema=PDFPage,
        return_direct=False,
    )

    class AskCsv(BaseModel):
        index: int = Field(description="Index number of the csv file that you want to ask about")
        query: str = Field(description="user query about this csv file")

    def ask_csv(index: int, query: str) -> str:
        csv_path = file_index.get(index)
        return csv(csv_path, query)

    csv_search = StructuredTool.from_function(
        func=ask_csv,
        name="AskCsv",
        description="search about some specific key concepts or words in the specific pdf that the user ask about",
        args_schema=AskCsv,
        return_direct=False,
    )

    tools = [doc_search, pdf_analyzer, csv_search]
    return tools

