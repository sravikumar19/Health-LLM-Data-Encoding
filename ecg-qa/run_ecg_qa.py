import argparse
import os 
from datetime import datetime
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerationConfig, GenerativeModel, Image, Part
from ecg_qa_helper import *

hz = 500
start_time = 0
time = 10
start_length = int(start_time * hz)
sample_length = int(time * hz)
end_time = start_time + time
t = np.arange(start_time, end_time, 1 / hz)
leads = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def run_ecg_qa():
    parser = argparse.ArgumentParser()
    parser.add_argument('question_type', type=str, choices=['single-verify', 'comparison_consecutive-verify', 'comparison_irrelevant-verify'], help= 'The type of ecg_question')
    parser.add_argument('encoding', type=str, choices=['natural_language', 'statistical_summary', 'visual'], help='The type of data encoding.')
    parser.add_argument('model_name', type=str, choices=['gemini-1.0'], help='The LLM or multimodal model used.')
    parser.add_argument('--data_path', type=str, required=True, help='Path of output/ptbxl/valid/*.json')
    parser.add_argument('--output_path', type=str, help='Path to save results to')
    args = parser.parse_args()

    PROJECT_ID = "urop-1"  
    LOCATION = "us-central1"  
    vertexai.init(project=PROJECT_ID, location=LOCATION)

    if not args.output_path:
        current_time = datetime.now()
        folder_name = 'results_' + question_type + "_" + encoding + "_" + model_name + "_" + current_time.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(folder_name)
        output_path = folder_name
    else:
        output_path = args.output_path
    print('processed output path')

    if 'single' in args.question_type:
        run_ecg_qa_single_verify(args.question_type, args.encoding, args.model_name, args.data_path, output_path)
    else:
        run_ecg_qa_comparison_verify(args.question_type, args.encoding, args.model_name, args.data_path, output_path)


def run_ecg_qa_single_verify(question_type, encoding, model_name, data_path, output_path):
    samples = load_ecg_data(question_type, data_path)
    client = OpenAI()

    results = []
    preds = []
    answers = []
    for sample in samples:
        question = sample['question']
        answer = sample['answer'][0]
        options = ['no', 'not sure', 'yes']
        ecg_path = sample["ecg_path"][0]
        ecg, _ = wfdb.rdsamp(ecg_path)
        instruction = """You are an intelligent healthcare agent, and your task is to answer a medical question. Please first think step-by-step and then output the answer. 
        Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str(answer)}. 
        Your responses will be used for research purposes only, so please have a definite answer."""
        if encoding == "natural_language":
            ecg_data = "\n".join(f"{lead}: {ecg.T[i][start_length: start_length + sample_length]}" for i, lead in enumerate(leads))
            prompt = f"""Given this 12-Lead ECG sequence, please answer the following question. 
                    ECG Sequence: {ecg_data}
                    Question: {question}
                    Options: {options}
                    """
            if model_name == 'gemini-1.0':
                model = GenerativeModel("gemini-1.0-pro")
                response = model.generate_content([instruction, prompt], stream=False).text
            elif model_name == 'gpt-4o':
                completion = client.chat.completions.create(
                    model = 'gpt-4o',
                    messages = [
                        {'role': 'system', 'content': instruction},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                response = completion.choices[0].message.content
        elif encoding == "statistical_summary":
            ecg_data = "\n".join(f"{lead}: {compute_statistics(ecg.T[i][start_length: start_length + sample_length])}" for i, lead in enumerate(leads))
            prompt = f"""Given a statistical summary of this 12-Lead ECG sequence, please answer the following question. 
                    ECG Sequence: {ecg_data}
                    Question: {question}
                    Options: {options}
                    """
            if model_name == 'gemini-1.0':
                model = GenerativeModel("gemini-1.0-pro")
                response = model.generate_content([instruction, prompt], stream=False).text
            elif model_name == 'gpt-4o':
                completion = client.chat.completions.create(
                    model = 'gpt-4o',
                    messages = [
                        {'role': 'system', 'content': instruction},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                response = completion.choices[0].message.content
        else:
            prompt_part1 = f"""Given the 12-Lead ECG sequence plotted below in the order of "I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6", please answer the following question."""
            prompt_part2 = f"""Question: {question}
                               Options: {options}"""
            draw_all_leads_ecg(ecg.T, output_path + '/ecg.png')
            image = Image.load_from_file(output_path + '/ecg.png')
            if model_name == 'gemini-1.0':
                inp = [instruction, prompt_part1, image, prompt_part2]
                model = GenerativeModel("gemini-1.0-pro-vision")
                response = model.generate_content(inp, stream=False).text
            elif model_name == 'gpt-4o':
                base64_image = encode_image(image)
                completion = client.chat.completions.create(
                    model = 'gpt-4o',
                    messages = [
                        {'role': 'system', 'content': instruction},
                        {'role': 'user', 'content': [
                                            {'type': 'text', 'text': prompt_part1},
                                            {'type': 'image_url', 'image_url': f"data:image/jpeg;base64,{base64_image}"},
                                            {'type': 'text', 'text': prompt_part2}
                                        ]
                        }
                    ]
                )
                response = completion.choices[0].message.content

        
        response_parsed = parse_response(response)
        pred = parse_pred(response_parsed, question_type)
        # print('prompt', prompt)
        # print('response', response)
        # print('pred', pred)

        preds.append(pred)
        answers.append(answer)
        results.append([sample['sample_id'], question, answer, pred])
    
    compute_accuracy(answers, preds)
    write_results_to_file(results, output_path)


def run_ecg_qa_comparison_verify(question_type, encoding, model_name, data_path, output_path):
    samples = load_ecg_data(question_type, data_path)
    client = OpenAI()

    results = []
    preds = []
    answers = []
    for sample in samples:
        question = sample['question']
        answer = sample['answer'][0]
        options = ['no', 'yes']
        ecg_path_1 = sample["ecg_path"][0]
        ecg_path_2 = sample["ecg_path"][1]
        ecg_1, _ = wfdb.rdsamp(ecg_path_1)
        ecg_2, _ = wfdb.rdsamp(ecg_path_2)
        instruction = """You are an intelligent healthcare agent, and your task is to answer a medical question. Please first think step-by-step and then output the answer. 
        Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Str(answer)}. 
        Your responses will be used for research purposes only, so please have a definite answer."""
            
        if encoding == "natural_language":
            ecg_1_data = "\n".join(f"{lead}: {ecg_1.T[i][start_length: start_length + sample_length]}" for i, lead in enumerate(leads))
            ecg_2_data = "\n".join(f"{lead}: {ecg_2.T[i][start_length: start_length + sample_length]}" for i, lead in enumerate(leads))
            prompt = f"""Given these two 12-Lead ECG sequence, please answer the following question. 
                        First ECG: {ecg_1_data}
                        Second ECG: {ecg_2_data}
                        Question: {question}
                        Options: {options}
                        """
            if model_name == 'gemini-1.0':
                model = GenerativeModel("gemini-1.0-pro")
                response = model.generate_content([instruction, prompt], stream=False).text
            elif model_name == 'gpt-4o':
                completion = client.chat.completions.create(
                    model = 'gpt-4o',
                    messages = [
                        {'role': 'system', 'content': instruction},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                response = completion.choices[0].message.content
        elif encoding == "statistical_summary":
            ecg_1_data = "\n".join(f"{lead}: {compute_statistics(ecg_1.T[i][start_length: start_length + sample_length])}" for i, lead in enumerate(leads))
            ecg_2_data = "\n".join(f"{lead}: {compute_statistics(ecg_2.T[i][start_length: start_length + sample_length])}" for i, lead in enumerate(leads))
            prompt = f"""Given a statistical summary of these two 12-Lead ECG sequence2, please answer the following question. 
                    First ECG: {ecg_1_data}
                    Second ECG: {ecg_2_data}
                    Question: {question}
                    Options: {options}
                    """
            if model_name == 'gemini':
                model = GenerativeModel("gemini-1.0-pro")
                response = model.generate_content([instruction, prompt], stream=False).text
            elif model_name == 'gpt-4o':
                completion = client.chat.completions.create(
                    model = 'gpt-4o',
                    messages = [
                        {'role': 'system', 'content': instruction},
                        {'role': 'user', 'content': prompt}
                    ]
                )
                response = completion.choices[0].message.content
        else:
            model = GenerativeModel("gemini-1.0-pro-vision")
            prompt_part1 = f"""Given the two 12-Lead ECG sequences plotted below, please answer the following question."""
            prompt_part2 = f"""Question: {question}
                               Options: {options}"""
            draw_all_leads_ecg(ecg_1.T, output_path + '/ecg1.png')
            draw_all_leads_ecg(ecg_2.T, output_path + '/ecg2.png')
            image_1 = Image.load_from_file(output_path + '/ecg1.png')
            image_2 = Image.load_from_file(output_path + '/ecg2.png')

            if model_name == 'gemini-1.0':
                inp = [instruction, prompt_part1, image_1, image_2, prompt_part2]
                model = GenerativeModel("gemini-1.0-pro-vision")
                response = model.generate_content(inp, stream=False).text
            elif model_name == 'gpt-4o':
                base64_image_1 = encode_image(image_1)
                base64_image_2 = encode_image(image_2)
                completion = client.chat.completions.create(
                    model = 'gpt-4o',
                    messages = [
                        {'role': 'system', 'content': instruction},
                        {'role': 'user', 'content': [
                                        {'type': 'text', 'text': prompt_part1},
                                        {'type:': 'image_url', 'image_url': f"data:image/jpeg;base64,{base64_image_1}"},
                                        {'type:': 'image_url', 'image_url': f"data:image/jpeg;base64,{base64_image_2}"},
                                        {'type': 'text', 'text': prompt_part2}]
                        }
                    ]
                )
                response = completion.choices[0].message.content
        
        response_parsed = parse_answer(response)
        pred = parse_pred(response_parsed)
        # print('prompt', prompt)
        # print('response', response)
        # print('pred', pred)

        preds.append(pred)
        answers.append(answer)
        results.append([sample['sample_id'], question, answer, response, pred])

    compute_accuracy(answers, preds)
    write_results_to_file(results, output_path)

            
if __name__ == "__main__":
    run_ecg_qa()
