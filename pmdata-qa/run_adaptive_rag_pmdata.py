import argparse
import os
from datetime import datetime
import pandas as pd
import numpy as np
from src.medrag import MedRAG
from adaptive_rag_helper import *
from template_pmdata import *
from vertexai.generative_models import GenerativeModel, Image


def predict_fatigue_and_readiness():
    parser = argparse.ArgumentParser()
    parser.add_argument('method', type=str, choices=['TAARE', 'COT', 'RAG'], help= 'The type of retrieval method.')
    parser.add_argument('target_var', type=str, choices=['readiness', 'fatigue'])
    parser.add_argument('encoding', type=str, choices=['natural_language', 'statistical_summary', 'visual'], help='The type of data encoding.') 
    parser.add_argument('context_length', type=int)
    parser.add_argument('--retriever', type=str, default='Contriever', choices=['BM25', 'Contriever', 'SPECTER', 'MedCPT'], help= 'The retriever for RAG and TAARE.')
    parser.add_argument('--corpus', type=str, default='Textbooks', choices=['Textbooks', 'StatPearls', 'Wikipedia', 'MedCorp', 'PubMed'], help= 'The corpus for RAG and TAARE.')
    parser.add_argument('--data_path', type=str, default = 'pmdata', help='Path to pmdata')
    parser.add_argument('--output_path', type=str, help='Path to save results to')
    args = parser.parse_args()

    fatigue_templates = {"statistical_summary": fatigue_question_stats, "visual": fatigue_question_visual,
                    "natural_language": fatigue_question_NL}

    readiness_templates = {"statistical_summary": readiness_question_stats, "visual": readiness_question_visual,
                    "natural_language": readiness_question_NL}

    if args.target_var == 'fatigue':
        prompt = fatigue_templates[args.encoding]
    else:
        prompt = readiness_templates[args.encoding]
    
    if not args.output_path:
        current_time = datetime.now()
        folder_name = 'results_' + args.method + "_" + args.target_var + "_" + args.encoding + "_" + current_time.strftime("%Y-%m-%d_%H-%M-%S")
        os.makedirs(folder_name)
        output_path = folder_name
    else:
        output_path = args.output_path

    answers = []
    preds = []
    bins = [-1, 1, 4, 7, 10]
    retrieval_count = 0
    non_retrieval_count = 0

    for item in os.listdir(args.data_path):
        if item != "participant-overview.xlsx" and item != ".DS_Store":
            item_path = os.path.join(args.data_path, item)
            df = load_data(item_path + "/", False)
            
            if args.target_var == 'readiness':
                indices = balance_samples(df, 'readiness', 5, bins)
            else:
                indices = balance_samples(df, 'fatigue', 4)
            
            answer, pred, retrieval, nonretrieval = adpative_retrieval(args.method, df, indices, args.context_length, prompt, args.target_var, args.encoding, args.retriever, args.corpus, output_path)
            answers.extend(answer)
            preds.extend(pred)
            retrieval_count += retrieval
            non_retrieval_count += nonretrieval

    print('fraction retrieved', retrieval_count / (non_retrieval_count + retrieval_count))
    if args.target_var == 'readiness':
        mae, mape = score_regression(answers, preds)
    else:
        acc, macro_f1, micro_f1 = score_classification(answers, preds)

    
def adpative_retrieval(method, df, sampled_indices, context_length, prompt, target_var, encoding, retriever, corpus, output_path):
    today = datetime.today().strftime('%Y-%m-%d')
    fewshot_examples = (
          "Question: what is the water boiling point?\nAnswer: [No].\n\n"
          "Question: As of today, name the 5 NFL teams that have never actually played in a super bowl?\nAnswer: [Yes].\n\n"
          "Question: What is the capital of France?\nAnswer: [No].\n\n"
          "Question: What is Walter de la Pole's occupation?\nAnswer: [Yes].\n\n"
    )
    model = GenerativeModel("gemini-1.0-pro")

    cot = MedRAG(llm_name="gemini", rag=False)
    medrag = MedRAG(llm_name="gemini", rag=True, retriever_name=retriever, corpus_name=corpus)

    answers = []
    preds = []
    retrieval_count = 0
    nonretrieval_count = 0
 
    for i in sampled_indices:
        if (not pd.isna(df[target_var][i])) and (i-context_length+1 >= 0):
            steps = list(df['steps'][i-context_length+1:i+1])
            calories = list(df['calories'][i-context_length+1:i+1])
            heart_rate = list(df['heartRate'][i-context_length+1:i+1])
            sleep = list(df['minutesAsleep'][i-context_length+1:i+1])
            
            if encoding == "natural_language":
                question = prompt.render(context_length = context_length, steps = str(steps), calories = str(calories), heart_rate = str(heart_rate), sleep = str(sleep), mood = str(df['mood'][i]))
            elif encoding == 'statistical_summary':
                question = prompt.render(context_length = context_length, steps = str(compute_statistics(steps)), calories = str(compute_statistics(calories)), heart_rate = str(compute_statistics(heart_rate)), sleep = str(compute_statistics(sleep)), mood = str(df['mood'][i]))
            elif encoding == "visual":
                question = prompt.render(context_length = context_length, mood = str(df['mood'][i]))
                create_multiple_plots(np.array(steps), np.array(heart_rate), np.array(calories), np.array(sleep), output_path)
                steps_image = Image.load_from_file(output_path + "steps.png")
                heart_rate_image = Image.load_from_file(output_path + "heart_rate.png")
                calories_image = Image.load_from_file(output_path + "calories.png")
                sleep_image = Image.load_from_file(output_path + "sleep.png")
            
            if method == "TAARE":
                prompt = """Today is {}. Given a question, determine whether you need to retrieve external resources, such as real-time search engines, Wikipedia, or databases, to answer the question correctly. Only answer \"[Yes]\" or \"[No]\".\n\n"
                    Here are some examples:\n\n
                    {}\n\n
                    Question: {}\n
                    Answer: """.format(today, fewshot_examples, question)
                responses = model.generate_content([prompt], stream=False)
                retrieval = responses.text
            elif method == "RAG":
                retrieval = "yes"
            elif method == "COT":
                retrieval = "no"
   
            if check_string_exist(retrieval) == 0:
                nonretrieval_count += 1
                if encoding == "visual":
                    pred, _, _ = cot.answer(question=[question, steps_image, heart_rate_image, calories_image, sleep_image])
                else:
                    pred, _, _ = cot.answer(question=question)
            else:
                retrieval_count += 1
                if encoding == "visual":
                    pred, snippets, scores = medrag.answer(question=[question, steps_image, heart_rate_image, calories_image, sleep_image], k=4)
                else:
                    pred, snippets, scores = medrag.answer(question=question, k=4) 

            parsed_pred = parse_score(pred)
            preds.append(parsed_pred)
            answers.append(str(df[target_var][i]))

    return answers, preds, retrieval_count, nonretrieval_count


if __name__ == "__main__":
    predict_fatigue_and_readiness()