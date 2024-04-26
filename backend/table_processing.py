import os, fitz
import torch
from torchvision import transforms
from transformers import AutoModelForObjectDetection
from PIL import Image as PILImage
from PIL import ImageDraw
import csv, json, time
import numpy as np
import pandas as pd
from paddleocr import PaddleOCR

from google.oauth2.service_account import Credentials
from google.api_core.client_options import ClientOptions
from google.cloud import documentai
from google.cloud import storage

cred_path = r"D:\7_Finsight_pro\finsightpro-08cbe597df83.json"

with open(cred_path) as f:
    creds = json.load(f)

Credentials_ = Credentials.from_service_account_info(creds)

PROJECT_ID = "finsightpro"
LOCATION = "us"
LOCATION_VERTEXAI = ""
PROCESSOR_ID = "d98c1f090a1c3280"
MIME_TYPE = "application/pdf"
BUCKET_NAME = ""


docai_client = documentai.DocumentProcessorServiceClient(
    credentials=Credentials_,
    client_options=ClientOptions(api_endpoint=f"{LOCATION}-documentai.googleapis.com"),
)

RESOURCE_NAME = docai_client.processor_path(PROJECT_ID, LOCATION, PROCESSOR_ID)

device = "cuda" if torch.cuda.is_available() else "cpu"


class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )

        return resized_image


detection_transform = transforms.Compose(
    [
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
structure_transform = transforms.Compose(
    [
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-detection", revision="no_timm"
).to(device)
structure_model = AutoModelForObjectDetection.from_pretrained(
    "microsoft/table-transformer-structure-recognition-v1.1-all"
).to(device)
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    width, height = size
    boxes = box_cxcywh_to_xyxy(out_bbox)
    boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
    return boxes


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs["pred_boxes"].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == "no object":
            objects.append(
                {
                    "label": class_label,
                    "score": float(score),
                    "bbox": [float(elem) for elem in bbox],
                }
            )
    return objects


def apply_coordinates_padding(coordinate, pad=50):
    coordinate["bbox"][0] = coordinate["bbox"][0] - pad
    coordinate["bbox"][1] = coordinate["bbox"][1] - pad
    coordinate["bbox"][2] = coordinate["bbox"][2] + pad
    coordinate["bbox"][3] = coordinate["bbox"][3] + pad
    return coordinate


def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry["label"] == "table row"]
    columns = [entry for entry in table_data if entry["label"] == "table column"]

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x["bbox"][1])
    columns.sort(key=lambda x: x["bbox"][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [
            column["bbox"][0],
            row["bbox"][1],
            column["bbox"][2],
            row["bbox"][3],
        ]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({"column": column["bbox"], "cell": cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x["column"][0])

        # Append row information to cell_coordinates
        cell_coordinates.append(
            {"row": row["bbox"], "cells": row_cells, "cell_count": len(row_cells)}
        )

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x["row"][1])

    return cell_coordinates


def recognize_table(image):
    # prepare image for the model
    # pixel_values = structure_processor(images=image, return_tensors="pt").pixel_values
    pixel_values = structure_transform(image).unsqueeze(0).to(device)

    # forward pass
    with torch.no_grad():
        outputs = structure_model(pixel_values)

    # postprocess to get individual elements
    id2label = structure_model.config.id2label
    id2label[len(structure_model.config.id2label)] = "no object"
    cells = outputs_to_objects(outputs, image.size, id2label)
    # visualize cells on cropped table
    draw = ImageDraw.Draw(image)
    filtered_cells = []
    for cell in cells:
        cell = apply_coordinates_padding(cell, pad=5)
        if cell["score"] >= 0.95:
            filtered_cells.append(cell)
            draw.rectangle(cell["bbox"], outline="red")

    return image, filtered_cells


def apply_ocr(cell_coordinates, cropped_table, use_paddle=False, use_docai=True):
    # print('first')
    # display(cropped_table)
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
        #   print('second',row)
        row_text = []
        for cell in row["cells"]:
            # print('third',cell)
            # crop cell out of image
            cell_image = np.array(cropped_table.crop(cell["cell"]))
            image = PILImage.fromarray(cell_image)
            # display(image)
            image.save("output_image.jpeg")

            with open("output_image.jpeg", "rb") as file:
                image_binary = file.read()

            if use_docai:
                try:
                    raw_document = documentai.RawDocument(
                        content=image_binary, mime_type="image/jpeg"
                    )
                    request = documentai.ProcessRequest(
                        name=RESOURCE_NAME, raw_document=raw_document
                    )
                    result = docai_client.process_document(request=request)
                    document_object = result.document
                    text = document_object.text
                    # print('text:>',text)
                except:
                    time.sleep(120)
                    raw_document = documentai.RawDocument(
                        content=image_binary, mime_type="image/jpeg"
                    )
                    request = documentai.ProcessRequest(
                        name=RESOURCE_NAME, raw_document=raw_document
                    )
                    result = docai_client.process_document(request=request)
                    document_object = result.document
                    text = document_object.text
                    # print('text:>',text)
                # print("OCR detected text:> ",text)
                if len(text) > 0:
                    row_text.append(text)
                else:
                    text = "nil"
                    row_text.append(text)

            elif use_paddle:
                result = ocr.ocr(cell_image)
                if result[0] is not None:
                    text = result[0][0][1][0]
                    row_text.append(text)
                # result= 'ashish'
                row_text.append(result)
            else:
                # apply OCR
                result = "a"
                if len(result) > 0:
                    text = " ".join([x[1] for x in result])
                    row_text.append(text)
                else:
                    text = "nil"
                    row_text.append(text)

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[str(idx)] = row_text

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for idx, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[str(idx)] = row_data

    # write to csv
    with open("output.csv", "w") as result_file:
        wr = csv.writer(result_file, dialect="excel")

        for row, row_text in data.items():
            wr.writerow(row_text)

    # return as Pandas dataframe
    try:
        df = pd.read_csv("output.csv")
    except Exception as e:
        print("Exception caught:>", e)
        df = pd.DataFrame([])

    return df, data


def detect_and_crop_table(file_name, page_no, image, directory, confidence):
    pixel_values = detection_transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    # postprocess to get detected tables
    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    detected_tables = outputs_to_objects(outputs, image.size, id2label)

    detected_tables = [item for item in detected_tables if item["score"] >= confidence]
    detected_tables = [
        apply_coordinates_padding(table_coordinate)
        for table_coordinate in detected_tables
    ]
    # print(f'detected tables per page with >=95% confidence:> ',len(detected_tables),detected_tables)
    table_dict = {}
    # Cropping function
    # if len(detected_tables)>0: display(image)

    for table_no, table_coordinate in enumerate(detected_tables):
        cropped_table = image.crop(table_coordinate["bbox"])
        cropped_table_copy = cropped_table.copy()
        annotated_table, cells = recognize_table(cropped_table_copy)

        # print("Annotated table", cells)
        # display(annotated_table)
        cell_coordinates = get_cell_coordinates_by_row(cells)
        # print("cell_coordinates:> ", cell_coordinates, len(cell_coordinates))
        if len(cells) and len(cell_coordinates):
            df, data = apply_ocr(cell_coordinates, cropped_table)
        else:
            df = pd.DataFrame([])
        # plot_df(df)
        table_name = str(file_name) + "_" + str(page_no) + "_" + str(table_no) + ".png"
        output_table_path = os.path.join(directory, table_name)
        cropped_table.save(output_table_path)
        table_dict[table_no] = {}
        table_dict[table_no]["file_name"] = file_name
        # table_dict[table_no]["year"] = file_name.split("_")[0]
        # table_dict[table_no]["company_name"] = file_name.split("_")[1]
        table_dict[table_no]["page_no"] = page_no
        table_dict[table_no]["table_no"] = table_no
        table_dict[table_no]["table_image_path"] = output_table_path
        table_dict[table_no]["table_content"] = df.to_json()
        table_dict[table_no]["table_content_df"] = df

    return table_dict


def detect_tables(
    file_name,
    page_no,
    page,
    zoom_factor=2,
    confidence=0.95,
    directory="detected_tables",
):
    os.makedirs(directory, exist_ok=True)
    """Input the file name, page no, and the page object, convert to image, magnify it, detect table, snip the table and save image
    and finally return dict of file_name, page_no, table image paths"""

    pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom_factor, zoom_factor))
    pil_image = PILImage.frombytes("RGB", [pixmap.width, pixmap.height], pixmap.samples)
    tables_per_page_metadata = detect_and_crop_table(
        file_name, page_no, pil_image, directory, confidence
    )
    # display(cropped_table)
    # print('after processing:> ',len(tables_per_page_metadata),tables_per_page_metadata)

    return tables_per_page_metadata
