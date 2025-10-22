import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

import googletrans

model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')



def question_answer(question, text):
    
    #tokenize question and text in ids as a pair
    input_ids = tokenizer.encode(question, text)
    
    #string version of tokenized ids
    tokens = tokenizer.convert_ids_to_tokens(input_ids)
    
    #segment IDs
    #first occurence of [SEP] token
    sep_idx = input_ids.index(tokenizer.sep_token_id)

    #number of tokens in segment A - question
    num_seg_a = sep_idx+1

    #number of tokens in segment B - text
    num_seg_b = len(input_ids) - num_seg_a
    
    #list of 0s and 1s
    segment_ids = [0]*num_seg_a + [1]*num_seg_b
    
    assert len(segment_ids) == len(input_ids)
    
    #model output using input_ids and segment_ids
    output = model(torch.tensor([input_ids]), token_type_ids=torch.tensor([segment_ids]))
    
    # print(output)

    #reconstructing the answer
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits)

    print(f"start: {answer_start}, end: {answer_end}")


    if answer_end >= answer_start:
        answer = tokens[answer_start]
        for i in range(answer_start+1, answer_end+1):
            if tokens[i][0:2] == "##":
                answer += tokens[i][2:]
            else:
                answer += " " + tokens[i]
    else:
        answer = "[CLS]"
                
    if answer.startswith("[CLS]"):
        answer = "Unable to find the answer to your question."
    
    # print("\nAnswer:\n{}".format(answer.capitalize()))
    return answer


# system_prompt = "Frankfurt, officially Frankfurt am Main (Literally \"Frankfurt on the Main\", ), is a metropolis and the largest city in the German state of Hesse and the fifth-largest city in Germany, with a 2015 population of 732,688 within its administrative boundaries, and 2.3\u00a0million in its urban area. The city is at the centre of the larger Frankfurt Rhine-Main Metropolitan Region, which has a population of 5.5\u00a0million and is Germany's second-largest metropolitan region after Rhine-Ruhr. Since the enlargement of the European Union in 2013, the geographic centre of the EU is about to the east of Frankfurt's CBD, the Bankenviertel. Frankfurt is culturally and ethnically diverse, with around half of the population, and a majority of young people, having a migration background. A quarter of the population are foreign nationals, including many expatriates. \n\nFrankfurt is an alpha world city and a global hub for commerce, culture, education, tourism and transportation. It's the site of many global and European corporate headquarters. Frankfurt Airport is among the world's busiest. Frankfurt is the major financial centre of the European continent, with the HQs of the European Central Bank, German Federal Bank, Frankfurt Stock Exchange, Deutsche Bank, Commerzbank, DZ Bank, KfW, several cloud and fintech startups and other institutes. Automotive, technology and research, services, consulting, media and creative industries complement the economic base. Frankfurt's DE-CIX is the world's largest internet exchange point. Messe Frankfurt is one of the world's largest trade fairs. Major fairs include the Frankfurt Motor Show, the world's largest motor show, the Music Fair, and the Frankfurt Book Fair, the world's largest book fair."
# user_prompt = "What is the official name of Frankfurt?"

system_prompt = "Maria are mere. Gheorghe vine și cere. Maria nu se îndură. Gheorghe vine și fură."
user_prompt = "Cine are mere?"

from gpytranslate import SyncTranslator
translator = SyncTranslator()


system_prompt = translator.translate(str(system_prompt), dest= 'en').text
user_prompt = translator.translate(str(user_prompt), dest='en').text

print(f"translated system: {system_prompt}")
print(f"translated user: {user_prompt}")

answer = question_answer(system_prompt, user_prompt)

answer = translator.translate(answer, dest='ro')
print(answer)