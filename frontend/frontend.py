import numpy as np
import gradio as gr
import requests
from PIL import Image
import json
from helper_fx import get_company_names,get_unique_years, generate_cashflow_reports, generate_reports


text_citations = []
image_citations = []
    
def search_and_analyze(name_dropdown,year_dropdown,search_term):
    """MAKE AN API CALL TO THE BACKEND AND GET THE TEXT RESPONSE, CITATIONS,IMAGE CITATIONS"""
    data = {"search_term": search_term, "name_dropdown":name_dropdown,"year_dropdown": year_dropdown}
    response = requests.post("http://localhost:8001/search_analyze/", data=json.dumps(data),headers={"Content-Type": "application/json"})
    if response.status_code == 200:
        result = response.json()
        # text_output = result["answer"]
        image_citations= result["image_citations"]
        text_citations= result["text_citations"]
        checked_citations= result['checked_citations']
    else:
        print(f"Error: {response.status_code}")
        return 'ERROR dummy text', 'ERROR dummy text citations', 'ERROR dummy image citations'
    return  text_citations, image_citations, checked_citations #text_output,

# LOGIC PART starts from here
def search_and_analyze_wrapper(name_dropdown,year_dropdown,search_term):
    text_citations, image_citations, checked_citations = search_and_analyze(name_dropdown,year_dropdown,search_term) #text_output,
    # print("PDF File:>", pdf_file)
    print("Search Term:>", search_term)
    # print("Text Output:>", text_output)
    print("Text citations:>", text_citations)
    print("Image citations:>", image_citations)
    return text_citations, image_citations, checked_citations #text_output,

def on_search_button_click(name_dropdown,year_dropdown,search_term):
    print("search parameters:> ",name_dropdown,year_dropdown)
    """Updated function to display UI values as output"""
    text_citations, image_citations, checked_citations = search_and_analyze_wrapper(name_dropdown,year_dropdown,search_term) #text_output, 
    # text_output_display.value = text_output
    citations_display.value = ""
    images = []
    labels= []
    
    for no, citation in enumerate(text_citations):
        citations_display.value += f"Citation {no+1}: {citation['chunk_content']}\n"
        citations_display.value += f"Source: {citation['file_name']}, Page: {citation['page_no']}\n"
        citations_display.value += "---\n"
    
    if image_citations:
        for citation in image_citations:
            img_array = np.array(citation["image"],dtype=np.uint8)  # Assuming "image" key is correct
            # print("Image array shape:", img_array.shape)
            # print("Image array type:", img_array.dtype)
            images.append(Image.fromarray(img_array))
            labels.append(str(citation['file_name']) + ': page# ' +str(citation['page_no']))
        gallery.value = images
        gallery.label= labels
    return citations_display.value, gallery.value, checked_citations   #text_output_display.value,

def on_upload_button_click(pdf_file):
    """Uploads the file to the backend and enables search functionality"""
    gr.Info("Uploading started")
    files = {"pdf_file": open(pdf_file, 'rb')}
    response = requests.post("http://localhost:8000/process_pdf/", files=files)
    if response.status_code == 200:
        print("File uploaded successfully!")
        search_term_input.interactive = True
        search_button.interactive = True
        returned_string = response.text  
        print('returned string:> ',returned_string)
    else:
        print(f"Error uploading file: {response.status_code}")
    return gr.update(interactive=True), gr.update(interactive=True)   



with gr.Blocks( title='Finsight Pro',theme="monochrome", css=r'.\static\style.css') as demo: #theme='monochrome',
    gr.HTML("""
    <div style="background: linear-gradient(to right, #ffafbd, #ffc3a0); padding: 10px; border-radius: 0px; margin-bottom: 20px;">
        <h1 style="color: black; text-align: center;">Finsight Pro</h1>
        <p style="color: black; text-align: center;">Revolutionizing Financial Document Analysis & Report Generation</p>
    </div>""")
    
    company_names= get_company_names()
    report_years= get_unique_years()
    
    with gr.Tab("Enterprise Search"):
        with gr.Row():
            with gr.Column():
                name_dropdown = gr.Dropdown(choices=company_names, label="Select Company Name", interactive=True)
            with gr.Column():
                year_dropdown = gr.Dropdown(choices=report_years , label="Choose the financial year", interactive=True)
        with gr.Row():
            with gr.Column():
                search_term_input = gr.Textbox(label="Search Query", interactive=True)
            with gr.Column():
                search_button = gr.Button("Search and Analyze", elem_classes="search_btn", interactive=True) 
        # text_output_display = gr.Textbox(label="Generative Output")
        # text_output_display = gr.Markdown(label="Generative Output")
        checked_citations_output_display = gr.Markdown(label="Verified citations")
        with gr.Accordion(label="Citations", open=True, elem_classes='accordian'):
            with gr.Column():
                citations_display = gr.Textbox(label="Text Citations")
                # citations_display= gr.Markdown(label="Text Citations")
            with gr.Row():
                gallery = gr.Gallery(label="Image Citations")
        
        search_button.click(on_search_button_click, inputs=[name_dropdown,year_dropdown,search_term_input], outputs=[citations_display, gallery,checked_citations_output_display]) #text_output_display
    
    
    
# ----------------------------------------------------THE FIRST FEATURE---------------------------------------------------- 
    with gr.Tab("Cash flow analysis Report"):
        with gr.Row():
            with gr.Column():
                name_dropdown = gr.Dropdown(choices=company_names, label="Select Company Name", interactive=True)
            with gr.Column():
                year_dropdown = gr.Dropdown(choices=report_years , label="Choose the financial year", interactive=True)
            with gr.Column():
                llm_choice= gr.Radio(choices=['Gemini-1.0-pro','Gemini-1.5-pro'], label="Choose LLM",value='Gemini-1.5-pro')
        with gr.Row():
            report_markdown = gr.Markdown(label="Financial Report")
            # pdf_download_button = gr.Button("Download PDF")
            # doc_state = gr.State() 
        gr.Button("Generate Report").click(
            fn=generate_cashflow_reports, 
            inputs=[name_dropdown, year_dropdown,llm_choice], 
            outputs=[report_markdown]  # report_table
        )
# ----------------------------------------------------THE SECOND FEATURE---------------------------------------------------- 
    with gr.Tab("Financial Statement Report"):
        with gr.Row():
            with gr.Column():
                name_dropdown = gr.Dropdown(choices=company_names, label="Select Company Name", interactive=True)
            with gr.Column():
                year_dropdown = gr.Dropdown(choices=report_years , label="Choose the financial year", interactive=True)
            with gr.Column():
                llm_choice= gr.Radio(choices=['Gemini-1.0-pro','Gemini-1.5-pro'], label="Choose LLM",value='Gemini-1.5-pro')
        with gr.Row():
            report_markdown = gr.Markdown(label="Financial Report")
            # pdf_download_button = gr.Button("Download PDF")
            # doc_state = gr.State()  
        gr.Button("Generate Report").click(
            fn=generate_reports, 
            inputs=[name_dropdown, year_dropdown,llm_choice], 
            outputs=[report_markdown]  # report_table
        )
        
# ----------------------------------------------------THE THIRD FEATURE----------------------------------------------------
    
    # with gr.Tab("Quick search & Analyze"):
    #     with gr.Row():
    #         with gr.Column():
    #             pdf_file_input = gr.File(label="PDF Document", file_count='single', elem_classes="upload-field", file_types=['pdf'])
    #             upload_button = gr.Button("Upload File", elem_classes="search_btn")
    #             # upload_feedback = gr.Textbox(label="Upload Feedback")
    #         with gr.Column():
    #             search_term_input = gr.Textbox(label="Search Query", lines=6.4, interactive=False)
    #             search_button = gr.Button("Search and Analyze", elem_classes="search_btn", interactive=False) 
    #     text_output_display = gr.Textbox(label="Generative Output")
    #     with gr.Accordion(label="Citations", open=True, elem_classes='accordian'):
    #         with gr.Column():
    #             citations_display = gr.Textbox(label="Text Citations")
    #             # citations_display= gr.Markdown(label="Text Citations")
    #         with gr.Row():
    #             gallery = gr.Gallery(label="Image Citations")

    #     upload_button.click(on_upload_button_click, show_progress='full',
    #                 inputs=[pdf_file_input],
    #                 outputs=[search_term_input, search_button],
    #                 )
        
    #     search_button.click(on_search_button_click, inputs=[search_term_input], outputs=[text_output_display, citations_display, gallery])

if __name__ == "__main__":
    demo.launch()