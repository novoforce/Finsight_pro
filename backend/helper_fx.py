import time
from PIL import Image as PILImage
# from IPython.display import display
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
import google.generativeai as gemini_client
import textwrap
from prompts import FINANCIAL_ANALYST_PROMPT,ANSWER_GENERATION_PROMPT
from PIL import Image
import PIL
import numpy as np


def create_chunks(
    file_name,
    page_no,
    page_text,
    splitter_type="recursive",
    chunk_size=500,
    chunk_overlap=100,
):
    """input the text of the page with the page no, return dict of page_no, chunks"""
    page_dict = {}
    if splitter_type == "recursive":
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        chunk_lists = text_splitter.split_text(
            page_text
        )  # u will get chunk list per page

        # print('page_no:> ',page_no, '\n No of chunks per page:> ', len(chunk_lists),'\n Chunk list:> ', chunk_lists)
        for chunk_no, chunk in enumerate(chunk_lists):
            page_dict[chunk_no] = {}  # for the initialization purpose
            page_dict[chunk_no]["file_name"] = file_name
            # page_dict[chunk_no]["year"] = file_name.split("_")[0]
            # page_dict[chunk_no]["company_name"] = file_name.split("_")[1]
            page_dict[chunk_no]["page_no"] = page_no
            page_dict[chunk_no]["chunk_content"] = chunk
        # print(f"TOTAL CHUNKS PER PAGE:> {len(page_dict)}, CHUNKS:> {page_dict}")
        return page_dict


def embed_chunks(chunk_dict, table=False):
    """Input with the chunk dict which has filename, page no, chunk no and  chunk content and finally return the embedding of the chunk content"""
    try:
        if table:
            print("table embedding generation")
            if len(chunk_dict["full_table_data"]):
                chunk_text_embedding = gemini_client.embed_content(
                    model="models/embedding-001",
                    content=chunk_dict["full_table_data"],
                    task_type="retrieval_document",
                    title=chunk_dict["file_name"],
                )
            else:
                print("no table embedding generation")
                chunk_text_embedding = {"embedding": [0] * 768}
        else:
            print("text embedding generation")
            chunk_text_embedding = gemini_client.embed_content(
                model="models/embedding-001",
                content=chunk_dict["chunk_content"],
                task_type="retrieval_document",
                title=chunk_dict["file_name"],
            )
    except Exception as e:
        print(f"Exception caught:> ", e)
        time.sleep(120)
        if table:
            print("table embedding generation")
            if len(chunk_dict["table_summary"]):
                chunk_text_embedding = gemini_client.embed_content(
                    model="models/embedding-001",
                    content=chunk_dict["table_summary"][0],
                    task_type="retrieval_document",
                    title=chunk_dict["file_name"],
                )
            else:
                print("no table embedding generation")
                chunk_text_embedding = {"embedding": [0] * 768}
        else:
            print("text embedding generation")
            chunk_text_embedding = gemini_client.embed_content(
                model="models/embedding-001",
                content=chunk_dict["chunk_content"],
                task_type="retrieval_document",
                title=chunk_dict["file_name"],
            )
    return chunk_text_embedding


def generate_table_summary(
    page_dict,
    vision_model,
    prompt="Write a summary of the content present in the image. Extract the content and include it with the summary",
):
    """input the table image and return summary of the table"""
    while True:
        try:
            img_path = page_dict["table_image_path"]
            print("img_path:> ", img_path)
            img = PILImage.open(img_path)
            # display(img)
            table_summary = vision_model.generate_content([prompt, img], stream=True)
            table_summary.resolve()
            print("table summary:> ", table_summary)
            for candidate in table_summary.candidates:
                return [part.text for part in candidate.content.parts]
            break
        except Exception as e:
            print(f"Exception caught:> ", e)
            time.sleep(120)
            # img_path= page_dict['table_image_path']
            # print("img_path:> ",img_path)
            # img= PILImage.open(img_path)
            # display(img)
            # table_summary= vision_model.generate_content([prompt,img], stream=True)
            # table_summary.resolve()
            # print("table summary:> ",table_summary)
            # for candidate in table_summary.candidates:
            #     return [part.text for part in candidate.content.parts]

    # return table_summary.text


def generate_table_summary_text_llm(page_dict, text_model, TABLE_SUMMARY_PROMPT):
    while True:
        try:
            table_data = page_dict["table_content"]
            print(":>", TABLE_SUMMARY_PROMPT.format(table_data=table_data))
            table_summary = text_model.generate_content(
                TABLE_SUMMARY_PROMPT.format(table_data=table_data)
            )
            table_summary.resolve()
            # print("table summary:> ",table_summary)
            for candidate in table_summary.candidates:
                return [part.text for part in candidate.content.parts][0]
            break
        except Exception as e:
            print(f"Exception caught:> ", e)
            time.sleep(120)
            # table_data= page_dict['table_content']
            # table_summary= text_model.generate_content(TABLE_SUMMARY_PROMPT.format(table_data= table_data))
            # table_summary.resolve()
            # # print("table summary:> ",table_summary)
            # for candidate in table_summary.candidates:
            #     return [part.text for part in candidate.content.parts][0]


def plot_df(df):
    # Render DataFrame as an image
    fig, ax = plt.subplots()
    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc="center",
        cellLoc="center",
        colColours=["black"] * len(df.columns),
        cellColours=[["white"] * len(df.columns)] * len(df),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)  # Adjust font size
    table.scale(3, 3)  # Adjust cell padding

    # Set column heading font color to white
    for key, cell in table.get_celld().items():
        if key[0] == 0:  # First row contains column headings
            cell.set_text_props(
                fontweight="bold", color="white"
            )  # Set font color to white

    # Dynamically adjust column widths
    cell_text = table.get_celld()
    max_col_widths = [
        max([len(str(cell_text[(i, j)].get_text())) for i in range(len(df) + 1)])
        for j in range(len(df.columns))
    ]
    for j, width in enumerate(max_col_widths):
        table.auto_set_column_width([j])

    # ax.set_title('DataFrame', fontweight='bold', fontsize=12, color='black')

    # Adjust font properties
    for key, cell in table.get_celld().items():
        if key[0] != 0:  # Skip first row (column headings)
            cell.set_text_props(
                fontweight="bold", color="black"
            )  # Set font color of cell text to black

    plt.savefig("dataframe_image.png", bbox_inches="tight")
    plt.show()


def make_prompt(query, text_chunks, table_chunks,year):
    escaped = text_chunks.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = textwrap.dedent(FINANCIAL_ANALYST_PROMPT).format(
        query=query, chunks=escaped, tables=table_chunks, year=year
    )
    return prompt


def read_image_from_path(image_path):
    try:
        image = np.array(Image.open(image_path)).tolist()
        return image
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at: {image_path}")
    except PIL.UnidentifiedImageError:
        raise PIL.UnidentifiedImageError(f"Unsupported image format: {image_path}")

#------------------FOR REPORT GENERATION---------------------------------

def group_parent_child(data):
    groups = []

    def traverse(current_data, parent_key=None):
        if not isinstance(current_data, dict):
            return  # Base case: Reached a leaf node, no grouping needed

        if all(not isinstance(value, dict) for value in current_data.values()):
            # All children are leaf nodes, so this is the last parent level
            groups.append((parent_key, current_data))
        else:
            # Explore children to find the last parent level
            for key, value in current_data.items():
                traverse(value, key)

    traverse(data)
    return groups

def generate_questions(grouped_data,prompt,llm, parser):
    financial_ques=[]
    for parent, template in grouped_data:
        print("parent:> ",parent)
        print("Template:> ",template)
        while True:
            try:
                chain = prompt | llm | parser
                result= chain.invoke({"json_template": template})
                if len(result.question_list) != len(template): raise ValueError('The Length of the generated list of questions and template is not same')
                break
            except Exception as e:
                print("Error occurred:", e)
                print("Retrying in 60 seconds...")
                time.sleep(5)
        print("question list:>",result.question_list)    
        print("--------------------------------"*3)
        
        # for (template, result.question_list):
            
        for item, (key, value) in zip(result.question_list, template.items()):
            print(f"List item: {item}, Dictionary key: {key}, Dictionary value: {value}")
            financial_ques.append({'topic':key, 'question':item, 'answer':value})
            
        print("*"*40)
    return financial_ques


import google.generativeai as genai

genai.configure(api_key="AIzaSyBkblsnqDJt99VWJ9OF0czRcGP0ngzkYNA")

# from qdrant_client import QdrantClient,models
# search_client = QdrantClient(path=r"D:\7_Finsight_pro\database")
# table_collection= "table_collection"
# text_collection= "text_collection"

def text_semantic_search(text_collection,search_client, models,query:str, year:str="2022"):
    """Useful for searching company's internal annual report database and get top 10 relevant TEXT output. The return type is a string"""
    search_results= search_client.search(
    collection_name=text_collection, limit=10,
    query_vector=genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query",
    )["embedding"],
    query_filter= models.Filter(must=[models.FieldCondition(
                key="year",
                match=models.MatchValue(value=str(year)),
            )])
    )
    text_chunks= ''
    page_no=[]
    print('text search results:> ',len(search_results))
    for no, result in enumerate(search_results):
        chunk_no= '\n <#CHUNK ' + str(no+1) + '>:\n'
        print(result.payload)
        # print(result.payload['page_no'])
        page_no.append(result.payload['page_no'])
        text_chunks = text_chunks + chunk_no + result.payload['chunk_content'] + '\n#Confidence score:>'+ str(result.score) + '\n'+'-'*50
    
    # print('final relevant chunks:>',text_chunks)
    return text_chunks, page_no

def table_semantic_search(table_collection,search_client,models, query:str, year:str= "2022"):
    """Useful for searching company's internal annual report database and get top 3 relevant TABLE output. The return type is a string"""
    search_results= search_client.search(
    collection_name=table_collection, limit=3,
    query_vector=genai.embed_content(
        model="models/embedding-001",
        content=query,
        task_type="retrieval_query",
    )["embedding"],
    query_filter= models.Filter(must=[models.FieldCondition(
                key="year",
                match=models.MatchValue(value=str(year)),
            )])
    )

    # print(search_results)

    table_extracts= ''
    page_no=[]
    print('table search results:> ',len(search_results))
    for no, result in enumerate(search_results):
        chunk_no= '\n <#TABLE ' + str(no+1) + '>:\n'
        print(result.payload)
        # print(result.payload['table_image_path'])
        page_no.append(result.payload['page_no'])
        table_extracts = table_extracts + chunk_no + result.payload['full_table_data'] + '\n#Confidence score:>'+ str(result.score) + '\n'+'-'*50
        
    # print('final relevant tables:>',table_extracts)
    return table_extracts, page_no

def generate_answers(financial_ques,text_collection,table_collection,search_client,models,company_name, year, ans_prompt,llm,ans_output_parser):
    financial_report_list= []
    for no, q in enumerate(financial_ques):
        print(f"Q{no+1}:> {q['question']}")
        while True:
            try:
                text_top_k, page_no_text =text_semantic_search(text_collection,search_client,models,query=q['question'],year=year)
                table_top_k, page_no_table =table_semantic_search(table_collection,search_client,models, query=q['question'],year=year)
                break
            except Exception as e:
                print("Qdrant ran into problem.. so sleeping for 10 seconds",e)
                time.sleep(10)
        while True:
            try:
                chain = ans_prompt | llm | ans_output_parser
                response= chain.invoke({"query": q,"text_top_k": text_top_k,"table_top_k": table_top_k})
                break
            except Exception as e:
                print("Error occurred:", e)
                print("Retrying in 120 seconds...")
                time.sleep(120)
                
        print(f"A{no+1}:> {response['answer']} \n")
        q['answer']= response['answer']
        q['year']= year
        financial_report_list.append({'topic':q['topic'], 'question':q['question'], 'answer':q['answer'], 'year':q['year']})
    
    return financial_report_list