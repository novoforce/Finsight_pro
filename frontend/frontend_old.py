import numpy as np
import gradio as gr
import requests
from PIL import Image


def search_and_analyze(pdf_file, search_term):
    """MAKE AN API CALL TO THE BACKEND AND GET THE TEXT RESPONSE, CITATIONS,IMAGE CITATIONS"""
    """WORK IN PROGRESS"""
    files = {"pdf_file": open(pdf_file, "rb")}  # Prepare file for upload
    data = {"search_term": search_term}
    response = requests.post(
        "http://localhost:8000/process_pdf/", files=files, data=data
    )
    if response.status_code == 200:
        result = response.json()
        final_table_dicts = result["final_table_dicts"]
        print(f"Table dictionary processed: {final_table_dicts}")
        text_output = "Generated output from LLM"

        text_citations = [
            {
                "chunk_text": "This is the first relevant text chunk.",
                "source": "Document A",
                "page": 10,
            },
            {
                "chunk_text": "Another important piece of information.",
                "source": "Document B",
                "page": 5,
            },
        ]

        image_citations = [
            {
                "image_url": "https://example.com/image1.jpg",
                "caption": "Image related to topic 1",
                "image_object": Image.fromarray(
                    np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
                ),
            },
            {
                "image_url": "https://example.com/image2.png",
                "caption": "Visualization of data",
                "image_object": Image.fromarray(
                    np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
                ),
            },
        ]
    else:
        print(f"Error: {response.status_code}")
        return (
            "ERROR dummy text",
            "ERROR dummy text citations",
            "ERROR dummy image citations",
        )

    return text_output, text_citations, image_citations


# LOGIC PART starts from here
def search_and_analyze_wrapper(pdf_file, search_term):
    text_output, text_citations, image_citations = search_and_analyze(
        pdf_file, search_term
    )
    print("PDF File:>", pdf_file)
    print("Search Term:>", search_term)
    print("Text Output:>", text_output)
    print("Text citations:>", text_citations)
    print("Image citations:>", image_citations)
    return text_output, text_citations, image_citations


def on_search_button_click(pdf_file, search_term):
    """Updated function to display UI values as output"""

    text_output, text_citations, image_citations = search_and_analyze_wrapper(
        pdf_file, search_term
    )

    text_output_display.value = text_output
    # citations_display.value = text_citations
    # image_display.value= image_citations
    # return text_output_display.value, citations_display.value, image_display.value
    text_citations = text_citations
    image_citations = image_citations

    citations_display.value = ""
    for citation in text_citations:
        citations_display.value += f"**Chunk Text:** {citation['chunk_text']}\n"
        citations_display.value += (
            f"**Source:** {citation['source']}, Page: {citation['page']}\n"
        )
        citations_display.value += "---\n"
    if image_citations:
        # image_display.value = image_citations[0]["image_object"]
        gallery.value = [citation["image_object"] for citation in image_citations]

    # return text_output_display.value, citations_display.value, image_display.value, gallery.value
    return text_output_display.value, citations_display.value, gallery.value


with gr.Blocks(
    theme="monochrome", title="Finsight Pro", css=r".\static\style.css"
) as demo:
    gr.HTML(
        """
    <div style="background: linear-gradient(to right, #ffafbd, #ffc3a0); padding: 10px; border-radius: 0px; margin-bottom: 20px;">
        <h1 style="color: black; text-align: center;">Finsight Pro</h1>
        <p style="color: black; text-align: center;">Revolutionizing Financial Document Analysis & Report Generation</p>
    </div>
    """
    )
    text_citations = []
    image_citations = []
    with gr.Tab(
        "Search and Analyze",
    ):
        with gr.Row():
            with gr.Column():
                pdf_file_input = gr.File(
                    label="PDF Document",
                    file_count="single",
                    elem_classes="upload-field",
                    file_types=["pdf"],
                )
                search_button = gr.Button(
                    "Search and Analyze", elem_classes="search_btn"
                )  # interactive=False
            with gr.Column():
                search_term_input = gr.Textbox(label="Search Term", lines=12.5)

        text_output_display = gr.Textbox(label="Generative Output")
        # citations_and_metrics_display = gr.Textbox(label="Citations & Metrics")
        with gr.Accordion(label="Citations", open=True, elem_classes="accordian"):
            with gr.Column():
                citations_display = gr.Textbox(label="Text Citations")
            with gr.Row():
                gallery = gr.Gallery(label="Image Citations")
                # image_display = gr.Image(label="Image Citations")
        search_button.click(
            on_search_button_click,
            inputs=[pdf_file_input, search_term_input],
            outputs=[text_output_display, citations_display, gallery],
        )

if __name__ == "__main__":
    demo.launch()
