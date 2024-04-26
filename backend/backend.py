from fastapi import FastAPI, File, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
import fitz
from qdrant_client.models import PointStruct
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import google.generativeai as gemini_client
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import pandas as pd
from PIL import Image

data = r"D:\7_Finsight_pro\pdf_data"

from helper_fx import create_chunks, embed_chunks, generate_table_summary_text_llm, make_prompt,read_image_from_path
from table_processing import detect_tables
from prompts import TABLE_SUMMARY_PROMPT

app = FastAPI()

# CORS middleware to allow cross-origin requests (if needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


GEMINI_API_KEY = "AIzaSyBkblsnqDJt99VWJ9OF0czRcGP0ngzkYNA"

gemini_client.configure(api_key=GEMINI_API_KEY)
vision_model = gemini_client.GenerativeModel("gemini-pro-vision")
text_model = gemini_client.GenerativeModel("gemini-1.5-pro-latest")
model = gemini_client.GenerativeModel("gemini-1.0-pro-latest")
search_client = QdrantClient(path=r"D:\7_Finsight_pro\backend\database")


def calculate_text_embeddings(final_dicts):
    for chunk_no, chunk_dict in final_dicts.items():
        final_dicts[chunk_no]["chunk_content_embedding"] = embed_chunks(chunk_dict)
    return final_dicts


def calculate_table_embeddings(final_table_dicts):
    for chunk_no, chunk_dict in final_table_dicts.items():
        final_table_dicts[chunk_no]["table_summary_embedding"] = embed_chunks(
            chunk_dict, table=True
        )
    return final_table_dicts

text_collection = "text_collection"
table_collection = "table_collection"


# Route to process PDF file
@app.post("/process_pdf/")
async def process_pdf(pdf_file: UploadFile = File(...)):
    final_dicts = {}
    final_table_dicts = {}
    chunk_count = 0
    table_count = 0
    contents = await pdf_file.read()
    doc = fitz.open(stream=contents)
    num_pages = doc.page_count
    file_name = pdf_file.filename
    print(f"PROCESSING {file_name} with {num_pages} pages")
    for page_no, page in enumerate(doc):
        print(f"PAGE# {page_no}")
        page_text = page.get_text()
        # print(page_no, page.get_text())
        page_dicts = create_chunks(file_name, page_no, page_text)
        for no, page_dict in page_dicts.items():
            # print(":> ",page_dict)
            final_dicts[chunk_count] = {}
            final_dicts[chunk_count] = page_dict
            chunk_count = chunk_count + 1

        page_table_dicts = detect_tables(file_name, page_no, page, confidence=0.95)
        for no, page_dict in page_table_dicts.items():
            # print("table:> ",page_dict)
            # import sys
            # sys.exit()
            final_table_dicts[table_count] = {}
            final_table_dicts[table_count] = page_dict
            # final_table_dicts[table_count]['table_summary']= generate_table_summary(page_dict,vision_model,table_summary_prompt1)
            final_table_dicts[table_count]["table_summary"] = (
                generate_table_summary_text_llm(
                    page_dict, text_model, TABLE_SUMMARY_PROMPT
                )
            )
            table_count = table_count + 1
            print("***********************" * 5)

    for key, value in final_table_dicts.items():
        # print(value)
        if "table_content" in value and "table_summary" in value:
            final_table_dicts[key]["full_table_data"] = (
                value["table_content"] + "\n\n" + value["table_summary"]
            )

    print(
        "df shape before embeddings:>",
        pd.DataFrame(final_table_dicts).transpose().shape,
    )
    final_dicts = calculate_text_embeddings(final_dicts)

    final_table_dicts = calculate_table_embeddings(final_table_dicts)
    print(
        "df shape after embeddings:>", pd.DataFrame(final_table_dicts).transpose().shape
    )

    
    search_client.recreate_collection(
        text_collection,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
        ),
    )

    
    search_client.recreate_collection(
        table_collection,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE,
        ),
    )

    points = []
    payload = {}
    for chunk_no, chunk_dict in final_dicts.items():
        idx = chunk_no
        vectors = chunk_dict["chunk_content_embedding"]["embedding"]
        payload = {
            key: chunk_dict[key]
            for key in ["file_name", "page_no", "chunk_content"]
            if key in chunk_dict
        }
        print(f"payload:> {payload} \n")
        points.append(PointStruct(id=idx, vector=vectors, payload=payload))
    search_client.upsert(collection_name=text_collection, wait=True, points=points)

    points = []
    payload = {}

    for chunk_no, chunk_dict in final_table_dicts.items():
        idx = chunk_no
        vectors = chunk_dict["table_summary_embedding"]["embedding"]
        payload = {
            key: chunk_dict[key]
            for key in [
                "file_name",
                "page_no",
                "table_no",
                "table_summary",
                "full_table_data",
                "table_image_path",
            ]
            if key in chunk_dict
        }
        points.append(PointStruct(id=idx, vector=vectors, payload=payload))
    search_client.upsert(collection_name=table_collection, wait=True, points=points)

    return 'Files uploaded successfully !!'

@app.post("/search_analyze/")
async def search_analyze(request: Request):
    data = await request.json()
    search_string = data["search_term"]
    print("search string:> ",search_string)
    
    search_results = search_client.search(
    collection_name=text_collection,
    limit=10,
    query_vector=gemini_client.embed_content(
        model="models/embedding-001",
        content=search_string,
        task_type="retrieval_query",
    )["embedding"],)

    # print("text results:> ",search_results)
    text_chunks = ""
    text_extracts_list= []
    for no, result in enumerate(search_results):
        chunk_no = "\n <#CHUNK " + str(no + 1) + ">:\n"
        print('payload:> ',result.payload)
        text_extracts_list.append({'file_name':result.payload['file_name'], 
                                    'page_no':result.payload['page_no'], 
                                    'chunk_content': result.payload['chunk_content'].replace("'", "").replace('"', "").replace("\n", " ")})
        
        text_chunks = text_chunks + chunk_no + result.payload["chunk_content"]
    # print("final relevant chunks:>", text_chunks)
#----------------------------------------------------------------------------------------------------------------------------
    search_results = search_client.search(
    collection_name=table_collection,
    limit=3,
    query_vector=gemini_client.embed_content(
        model="models/embedding-001",
        content=search_string,
        task_type="retrieval_query",
    )["embedding"],)

    print("table results:>",search_results)

    table_extracts = ""
    table_extracts_list= []
    for no, result in enumerate(search_results):
        chunk_no = "\n <#TABLE " + str(no + 1) + ">:\n"
        print(result.payload['file_name'])
        print(result.payload['page_no'])
        print(result.payload['table_image_path'])
        image = read_image_from_path(result.payload['table_image_path'])
        table_extracts_list.append({'file_name':result.payload['file_name'], 
                                    'page_no':result.payload['page_no'], 
                                    'image':image})
        # image.show()
        # print(result.payload['table_summary'])
        table_extracts = table_extracts + chunk_no + result.payload["full_table_data"]

    # print("final relevant tables:>", table_extracts)
    
    prompt = make_prompt(search_string, text_chunks, table_extracts)
    
    
    answer = model.generate_content(prompt)
    
    print('LLM answer:> ',answer.text)
    
    return {"answer": answer.text,"text_citations":text_extracts_list, "image_citations":table_extracts_list}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
