import base64
from io import BytesIO  
from pdf2image import convert_from_path
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

def convert_pdf_to_image(pdf_path, page_number):
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    return images[0]

def encode_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def ask_gpt_about_image(encoded_image, query):
    chat = ChatOpenAI(model="gpt-4o-2024-05-13")
    response = chat.invoke(
        [
            SystemMessage(
                content=[{"type": "text", "text": "Provide correct answers to user queries for a given image."}]
            ),
            HumanMessage(
                content=[
                    {"type": "text", "text": query},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}},
                ]
            )
        ]
    )
    if isinstance(response, AIMessage):
        response = response.content
    return response


def process_pdf_image_query(pdf_path, page_number, query):
    image = convert_pdf_to_image(pdf_path, page_number)
    encoded_image = encode_image_to_base64(image)
    response = ask_gpt_about_image(encoded_image, query)
    return response


