from fastapi import FastAPI, Body, status,Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google.oauth2.service_account import Credentials
import google.generativeai as gemini_client
from prompts import QUESTIONS_GENERATION_PROMPT, ANSWER_GENERATION_PROMPT
from helper_fx import group_parent_child, generate_questions, generate_answers,read_image_from_path,make_prompt
import json,os

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

report_template = "8.1_financial_stmt.json"
cashflow_report_template= "9.1_cashflow_analysis_report.json"
# report_template= r""
cred_path = r"finsightpro-08cbe597df83.json"
with open(cred_path) as f:
    creds = json.load(f)
Credentials_ = Credentials.from_service_account_info(creds)

GEMINI_API_KEY = "AIzaSyBkblsnqDJt99VWJ9OF0czRcGP0ngzkYNA"
gemini_client.configure(api_key=GEMINI_API_KEY)

PROJECT_ID = "finsightpro"
LOCATION = "us"
LOCATION_VERTEXAI = "us-central1"
PROCESSOR_ID = "d98c1f090a1c3280"
MIME_TYPE = "application/pdf"
BUCKET_NAME = ""

import vertexai

vertexai.init(project=PROJECT_ID, location=LOCATION_VERTEXAI, credentials=Credentials_)
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part
from langchain_google_vertexai import VertexAI
from qdrant_client import QdrantClient, models

# search_client = QdrantClient(path=r"D:\7_Finsight_pro\database")
search_client = QdrantClient(
    url="https://d0c5403e-53d0-40b2-8e02-f2e3e12d8666.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="s60thH3vPbOBlPOWkmxjXcU2RIV5VGTh8WPyVd7E1HNu6-T7MPG0rA",
)
table_collection = "table_collection"
text_collection = "text_collection"

model = VertexAI(model_name="gemini-pro")

from typing import List
import time
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


class Questions(BaseModel):
    question_list: List[str] = Field(description="list of questions")


ques_parser = PydanticOutputParser(pydantic_object=Questions)

ques_prompt = PromptTemplate(
    template=QUESTIONS_GENERATION_PROMPT,
    input_variables=["json_template"],
    partial_variables={"format_instructions": ques_parser.get_format_instructions()},
)


# template = ANSWER_GENERATION_PROMPT
response_schema = [
    ResponseSchema(
        name="answer",
        description="Answer to the user's question in minimum words possible. Do not give descriptive answer",
    )
]
ans_output_parser = StructuredOutputParser.from_response_schemas(response_schema)
format_instructions = ans_output_parser.get_format_instructions()

ans_prompt = PromptTemplate(
    template=ANSWER_GENERATION_PROMPT,
    input_variables=["query", "text_top_k", "table_top_k"],
    partial_variables={"format_instructions": format_instructions},
)


@app.get("/")
def home_page():
    return {"Cheers!! The backend is running fine -Ashish Johnson"}

@app.post("/check_citations/")
async def check_citations(request: Request):
    data = await request.json()
    print("-->",data)
    return data


@app.get("/company_names")
async def get_company_names():
    while True:
        try:
            # print("got request")
            results = search_client.scroll(collection_name=text_collection, limit=10000)
            break
        except Exception as e:
            print("Sleeping for 10 sec, Exception occured:> ",e)
            time.sleep(10)
    records = results[0]
    company_names_list = []
    for record in records:
        company_names_list.append(record.payload["company_name"])
    print('unique company list:> ',list(set(company_names_list)))

    return list(set(company_names_list))


@app.get("/unique_years")
async def get_unique_years():
    while True:
        try:
            results = search_client.scroll(collection_name=text_collection, limit=10000)
            break
        except Exception as e:
            print("Sleeping for 10 sec, Exception occured:> ",e)
            time.sleep(10)
    records = results[0]
    year_list = []
    for record in records:
        year_list.append(record.payload["year"])
        
    print('unique year list:> ',list(set(year_list)))
    return list(set(year_list))


@app.post("/generate_report")
async def process_data(
    company_name: str = Body(...), year: int = Body(...), llm_choice: str = Body(...), report_name: str= Body(...)
):
    if llm_choice == "Gemini-1.0-pro":
        llm = VertexAI(
            model_name="gemini-1.0-pro-002",
            max_output_tokens=1000,
            request_timeout=120,
            temperature=0,
        )
    elif llm_choice == "Gemini-1.5-pro":
        llm = VertexAI(
            model_name="gemini-1.5-pro-preview-0409",
            max_output_tokens=1000,
            request_timeout=120,
            temperature=0,
        )

    if report_name == 'financial_statement_report':
        print("Financial report selected")
        report_template_path= os.path.join("report_templates", report_template)
        with open(report_template_path, "r") as f:
            data = json.load(f)
    elif report_name == 'cashflow_report':
        print("Cashflow report selected")
        cashflow_report_template_path= os.path.join("report_templates", cashflow_report_template)
        with open(cashflow_report_template_path, "r") as f:
            data = json.load(f)
            
    grouped_data = group_parent_child(data)

    financial_ques = generate_questions(grouped_data, ques_prompt, llm, ques_parser)
    financial_report_list = generate_answers(
        financial_ques,
        text_collection,
        table_collection,
        search_client,
        models,
        company_name,
        year,
        ans_prompt,
        llm,
        ans_output_parser,
    )
    print("final op:> ", len(financial_report_list), financial_report_list)

    return financial_report_list

def check_citations(answer,table_extracts,text_extracts):
    import textwrap
    print("LLM answer:> ",answer)
    print("-"*50)
    print("table extracts list:> ",table_extracts)
    print("-"*50)
    print("text extracts list:> ",text_extracts)
    print("-"*50)

    PROMPT= """
    You are a friendly and experienced financial analyst with a decade of expertise in verifying financial information and working with citations.
    Your task is to fact-check a financial newsletter for accuracy. You will be provided with:
    A "final newsletter" document: This contains financial information and claims that need verification.
    Corresponding citations: These are references to the sources of the information, presented in both tabular and textual formats.
    Please carefully review the "final newsletter" and verify if the information presented is accurately supported by the provided citations.
    #newsletter:> ```{query}```
    #text citations:> {chunks}
    #table citations:> {tables}
    Here's how to proceed:
    THINK STEP-BY-STEP:
    1. Locate information requiring verification: Identify claims, statistics, or financial data within the newsletter that needs to be checked.
    2. Find the corresponding citation: Search for the specific citation (either in the tables or text) that supports the information you are verifying.
    3. Verify accuracy: Carefully examine the citation to ensure it accurately supports the information presented in the newsletter.
    4. After you have verified the correctness of the information in the newsletter, you have to ADD citations to the newsletter. How to do? 
    You simply add chunk no as citation no, table no at the end. In this way people reading newletter can go to the respective Citations.
    
    for example:> 
    Example1:
    Dominance of Advertising: Advertising, through "Google Search & other," "YouTube ads," and "Google Network," forms the bedrock of the company's revenue,
    accounting for over 80% of the total. **[Citation 1, Citation 10]**
    
    Example2:
    Google Cloud revenue more than doubled between 2020 and 2022, reaching $26.280 billion in 2022. 
    This signals the success of Google's endeavors in the cloud computing space. **[Citation 5, Table 3]**
    
    Important things to remember! 
    - Do not give any new headings. For example: if the heading is ```Google's Revenue Breakdown in 2022: A Closer Look``` , then do not make it ```Google's Revenue Breakdown in 2022: A Closer Look with Citations```,
    similarly for the rest of the contents in Newsletter, Do not change the content, just add citations in the format given above.
    - You simply have to read the newsletter, add the citations wherever needed at the end of sentence in the above format, and then return it as it is. That's all !!
    - Do not edit the newsletter other than adding citations. 
    - Do not add citations from your side as well. 
    - Only Cite information from provided Tables or Chunks.
    
    Please provide your final output in Markdown format, with the verified citations clearly marked in bold.
    
    Remember, your expertise and attention to detail are crucial in ensuring the accuracy and credibility of the financial information presented in the newsletter.
    """
    
    PROMPT= """
    ## Your Role: Financial Analyst & Fact-Checker

You are a friendly and experienced financial analyst with a decade of expertise in verifying financial information and working with citations. Your task is to ensure the accuracy of a financial newsletter by meticulously checking its claims against provided sources. 

**Here's what you'll receive:**

* **"Final Newsletter" document:** This contains financial information and statements that require verification.  **#newsletter:>** ```{query}``` 
* **Citations:** References to the sources of information, provided in both textual ("chunks") and tabular formats.  **#text citations:>** {chunks}   **#table citations:>** {tables}

**Your Mission:**

1. **Identify claims:**  Scan the newsletter for specific claims, statistics, or financial data that need verification.
2. **Match with Citations:** Locate the corresponding citation (either within the text chunks or tables) that directly supports the information you're verifying.
3. **Verify Accuracy:** Carefully compare the information in the newsletter with the source material to ensure it is accurate and accurately reflects the cited source.
4. **Add Citations:** Once verified, add the appropriate citation(s) to the end of the sentence containing the information. Use the following format: 

> **Example 1 (Text Citation):** Dominance of Advertising: Advertising, through "Google Search & other," "YouTube ads," and "Google Network," forms the bedrock of the company's revenue, accounting for over 80% of the total. **[Chunk 1, Chunk 10]**
>
> **Example 2 (Table Citation):** Google Cloud revenue more than doubled between 2020 and 2022, reaching $26.280 billion in 2022. This signals the success of Google's endeavors in the cloud computing space. **[Chunk 5, Table 3]**

**Important Guidelines:** 

* **Maintain Formatting:** Do not alter the existing headings or content of the newsletter. Only add citations in the specified format.
* **No External Sources:**  Only use the provided "chunks" and "tables" for citations. Do not introduce any external sources or personal knowledge.
* **Focus on Accuracy:** Your primary goal is to ensure the information presented in the newsletter is accurate and properly supported by the provided references. 
* **Correct Citation:** The citation(s) should not be mixed up. Be extra cautious when citating information from chunks and tables. Remember A citation is identified by the chunk no, table no. 
* Don't add similar sentences in the output: ```Financial Newsletter with Verified Claims```

**Output Format:**

* Please provide your final output in Markdown format, with the added citations clearly marked in bold. 

**Remember, your diligence and expertise are crucial in maintaining the credibility and accuracy of the financial information presented in the newsletter.**    
    """
    
    PROMPT="""
## Your Role: Financial Analyst & Fact-Checker

You are a friendly and experienced financial analyst with a decade of expertise in verifying financial information and working with citations. Your task is to ensure the accuracy of a financial newsletter by meticulously checking its claims against provided sources. 

**Here's what you'll receive:**

* **"Final Newsletter" document:** This contains financial information and statements that require verification.  **#newsletter:>** ```{query}``` 
* **Citations:** References to the sources of information, provided in both textual ("chunks") and tabular formats.  **#text citations:>** {chunks}   **#table citations:>** {tables}

**Your Mission:**

1. **Identify claims:**  Scan the newsletter for specific claims, statistics, or financial data that need verification.
2. **Match with Citations:** Locate the corresponding citation (either within the text chunks or tables) that directly supports the information you're verifying.
3. **Verify Accuracy:** Carefully compare the information in the newsletter with the source material to ensure it is accurate and accurately reflects the cited source.
4. **Add Citations:** Once verified, add the appropriate citation(s) to the end of the sentence containing the information. Use the following format: 

> **Example 1 (Text Citation):** Dominance of Advertising: Advertising, through "Google Search & other," "YouTube ads," and "Google Network," forms the bedrock of the company's revenue, accounting for over 80% of the total. **[Chunk 1, Chunk 10]**
>
> **Example 2 (Table Citation):** Google Cloud revenue more than doubled between 2020 and 2022, reaching $26.280 billion in 2022. This signals the success of Google's endeavors in the cloud computing space. **[Chunk 5, Table 3]**

**Important Guidelines:** 

* **Maintain Formatting:** Do not alter the existing headings or content of the newsletter. Only add citations in the specified format.
* **No External Sources:**  Only use the provided "chunks" and "tables" for citations. Do not introduce any external sources or personal knowledge.
* **Focus on Accuracy:** Your primary goal is to ensure the information presented in the newsletter is accurate and properly supported by the provided references. 
* **Citation Precision:** Pay close attention to ensure each citation accurately corresponds to the information it supports. Avoid mixing up citations or assigning incorrect references. Remember, each citation is uniquely identified by its chunk or table number.

**Output Format:**

* Please provide your final output in Markdown format, with the added citations clearly marked in bold. 

**Remember, your diligence and expertise are crucial in maintaining the credibility and accuracy of the financial information presented in the newsletter.  Accurate citations are essential for transparency and building trust with the readers.**
"""
    
    escaped_text = text_extracts.replace("'", "").replace('"', "").replace("\n", " ")
    escaped_table = table_extracts.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent(PROMPT).format(
        query=answer, chunks=escaped_text, tables=escaped_table
    )
    answer = model.invoke(prompt)
    return answer

@app.post("/search_analyze/")
async def search_analyze(request: Request):
    data = await request.json()
    print("-->",data)
    search_string = data["search_term"]
    company= data['name_dropdown']
    year= data['year_dropdown']
    print("search string:> ",search_string)
    while True:
        try:
            search_results = search_client.search(
            collection_name=text_collection,
            limit=10,
            query_vector=gemini_client.embed_content(
                model="models/embedding-001",
                content=search_string,
                task_type="retrieval_query",
            )["embedding"],
            query_filter= models.Filter(must=[models.FieldCondition(key="year",match=models.MatchValue(value=year)), models.FieldCondition(key="company_name",match=models.MatchValue(value=company))]))
            break
        except Exception as e:
            print("Exception caught:> ",e)
            print("Sleeping for 10 sec")
            time.sleep(10)

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
    while True:
        try:
            search_results = search_client.search(
            collection_name=table_collection,
            limit=3,
            query_vector=gemini_client.embed_content(
                model="models/embedding-001",
                content=search_string,
                task_type="retrieval_query",
            )["embedding"],query_filter= models.Filter(must=[models.FieldCondition(key="year",match=models.MatchValue(value=year)), models.FieldCondition(key="company_name",match=models.MatchValue(value=company))]))
            break
        except Exception as e:
            print("Exception caught:> ",e)
            print("Sleeping for 10 seconds")
            time.sleep(10)

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
    
    prompt = make_prompt(search_string, text_chunks, table_extracts, year)
    answer = model.invoke(prompt)
    
    print('LLM answer:> ',answer)
    
    citation_check= check_citations(answer,table_extracts,text_chunks)
    
    return {"answer": answer,"text_citations":text_extracts_list, "image_citations":table_extracts_list, 'checked_citations': citation_check}



if __name__ == "__main__":
    import uvicorn

    uvicorn.run("report_generation_backend:app", host="0.0.0.0", port=8001, reload=True)
