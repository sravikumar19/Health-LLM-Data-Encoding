from liquid import Template

general_zero_shot_system = '''You are an intelligent healthcare agent and your task is to answer a medical question.'''

general_zero_shot = Template('''
{{question}}

''')

general_few_shot_system = '''You are a helpful medical expert, and your task is to answer a medical question.'''

general_few_shot = Template('''
Here are some examples:\n\n
{{fewshot_examples}}\n\n

Here is the question:
{{question}}

''')

general_cot_system = '''You are a helpful medical expert, and your task is to answer a medical question. Please first think step-by-step and then output the answer. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Float(score)}. Do not output any additional information. Your responses will be used for research purposes only, so please have a definite answer.'''

general_cot = Template('''
Here is the question:
{{question}}

Please think step-by-step and generate your output in json:
''')

general_medrag_system = '''You are a helpful medical expert, and your task is to answer a medical question using the relevant documents. Please first think step-by-step and then output a score. Organize your output in a json formatted as Dict{"step_by_step_thinking": Str(explanation), "answer_choice": Float(score)}. Your responses will be used for research purposes only, so please have a definite answer.'''

general_medrag = Template('''
Here are the relevant documents:
{{context}}

Here is the question:
{{question}}

Please think step-by-step and generate your output in json:
''')

meditron_cot = Template('''
### User:
Here is the question:
...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Please think step-by-step and generate your output in json.

### Assistant:
''')

meditron_medrag = Template('''
Here are the relevant documents:
{{context}}

### User:
Here is the question:
...

Please think step-by-step and generate your output in json.

### Assistant:
{"step_by_step_thinking": ..., "answer_choice": "X"}

### User:
Here is the question:
{{question}}

Please think step-by-step and generate your output in json.

### Assistant:
''')