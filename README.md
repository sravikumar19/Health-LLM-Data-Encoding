# Health-LLM-Data-Encoding

## Testing time-series encodings and TA-ARE on PMData 
1. Clone MedRAG repo from https://github.com/Teddy-XiongGZ/MedRAG
2. Install requirements for MedRAG
3. Replace MedRAG/src/medrag.py with pmdata-qa/MedRAG/medrag.py and MedRAG/src/template.py with pmdata-qa/MedRAG/template.py
4. Copy pmdata-qa/run_adaptive_rag_pmdata.py, pmdata-qa/adaptive_rag_helper.py, and pmdata-qa/template_pmdata.py into MedRAG
5. python run_adaptive_rag_pmdata.py $method $target_var $encoding $context_length
* $method is 'TAARE', 'RAG', or 'COT'
* $target_var is 'readiness' or 'fatigue'
* $encoding is 'natural_language', 'statistical_summary', or 'visual'
* $context_length is length of time series context
* --output_path is path results will be saved to

## Testing time-series encodings on ECG-QA dataset
1. Clone ECG-QA dataset from https://github.com/Jwoo5/ecg-qa
2. Follow steps 1-3 in quickstart instructions section
3. Copy run_ecg_qa.py and ecg-qa/ecg_qa_helper.py into ecg-qa
4. python run_ecg_qa.py $question_type $encoding $model_name --data_path output/ptbxl/valid/*.json
* $method is TAARE, RAG, COT
* $question_type is 'single-verify', 'comparison_consecutive-verify' or 'comparison_irrelevant-verify'
* $encoding is 'natural_language', 'statistical_summary', or 'visual'
* $model_name is 'gemini-pro-1.0' or 'gpt-4o'
* --data_path is path of output/ptbxl/valid/*.json
* --output_path is path results will be saved to


