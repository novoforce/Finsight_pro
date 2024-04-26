import requests
def get_company_names():
    url = "http://localhost:8001/company_names"
    # url= "https://report-generation-backend-ke4dqnrqgq-uc.a.run.app/company_names"
    response = requests.get(url)
    print("response:> ",response)
    if response.status_code == 200:
        company_names = response.json()  # Assuming the response is a JSON list
        print(company_names)
        return company_names
    else:
        print(f"Error: {response.status_code}")
        return []

def get_unique_years():
    url = "http://localhost:8001/unique_years"
    # url= "https://report-generation-backend-ke4dqnrqgq-uc.a.run.app/unique_years"
    response = requests.get(url)

    if response.status_code == 200:
        unique_years = response.json()  # Assuming the response is a JSON list
        print(unique_years)
        return unique_years
    else:
        print(f"Error: {response.status_code}")
        return []
    
def generate_reports(selected_company_name, selected_year,llm_choice):
    # Your code here
    print(f"Selected Company: {selected_company_name}, Year: {selected_year} , Selected llm: {llm_choice}, report_name: cashflow_report")
    
    url = "http://localhost:8001/generate_report"
    # url= "https://report-generation-backend-ke4dqnrqgq-uc.a.run.app/generate_report"
    data = {"company_name": selected_company_name, "year": selected_year,'llm_choice':llm_choice, "report_name":'financial_statement_report'}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        results = response.json()
        markdown_table = "| Topic | Answer | Year |\n|---|---|---|\n"
        for d in results:
            markdown_table += f"| {d['topic']} | {d['answer']} | {d['year']} |\n"
        
        print('Response:> ',results)
    else:
        print(f"Error: {response.status_code}")
    return markdown_table


    
def generate_cashflow_reports(selected_company_name, selected_year,llm_choice):
    # Your code here
    print(f"Selected Company: {selected_company_name}, Year: {selected_year} , Selected llm: {llm_choice},report_name: cashflow_report")
    
    url = "http://localhost:8001/generate_report"
    # url= "https://report-generation-backend-ke4dqnrqgq-uc.a.run.app/generate_report"
    data = {"company_name": selected_company_name, "year": selected_year,'llm_choice':llm_choice, "report_name":'cashflow_report'}
    response = requests.post(url, json=data)
    if response.status_code == 200:
        results = response.json()
        markdown_table = "| Topic | Answer | Year |\n|---|---|---|\n"
        for d in results:
            markdown_table += f"| {d['topic']} | {d['answer']} | {d['year']} |\n"
        
        print('Response:> ',results)
    else:
        print(f"Error: {response.status_code}")
    return markdown_table
