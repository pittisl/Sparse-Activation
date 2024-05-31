###################################
'''
The code is for truthful_qa labels generation
'''
###################################
import torch
import transformers
from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoTokenizer, PhiConfig, PhiForCausalLM, AutoModelForCausalLM
from  datasets  import  load_dataset
import model.modeling_phi as modeling_phi
from generation.utils_normal import GenerationMixin
import json
from tqdm import tqdm

torch.set_default_device("cuda")
configuration = PhiConfig.from_json_file("phi/config.json")
configuration.num_key_value_heads=configuration.num_attention_heads
model= modeling_phi.PhiForCausalLM.from_pretrained("microsoft/phi-2", config=configuration)
tokenizer = AutoTokenizer.from_pretrained("phi", trust_remote_code=True)
    
#load dataset
dataset_name = "truthful_qa"
truthful_dataset = load_dataset(dataset_name, "generation")["validation"]
text_input_list = truthful_dataset["question"]
text_solution_list= truthful_dataset["best_answer"]

# Set a fixed seed
torch.manual_seed(42)

#set the hyperparameters
base=0
num_data=100 #number of the data samples
label_list=[]

#obtain the output of each sample
for sample_idx in tqdm(range(num_data), desc="Generating labels"):
    text_input="Instruct: "+text_input_list[sample_idx]+"\nOutput:"
    text_solution=text_solution_list[sample_idx]
    solution = tokenizer(text_solution, return_tensors="pt", return_attention_mask=False)
    inputs = tokenizer(text_input, return_tensors="pt", return_attention_mask=False)
    inputs_ids=inputs.input_ids
    solution = solution.input_ids    #solution.input_ids[:,199:]

    outputs = GenerationMixin.greedy_search(model, inputs_ids, solution, stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=inputs_ids.size(1)+solution.size(1))]), return_dict_in_generate=True, output_scores=True, baseline=base, mask_layer_list=None, heads_mask=None, pad_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)

    output_ids = outputs.sequences[:, inputs_ids.size(1):]  # 从输入部分之后的第一个token开始取
    text = tokenizer.batch_decode(output_ids)[0]

    label_list.append(text)


# save the labels in the file label_generation
#with open('labels2.json', 'w') as file:
    #json.dump(label_list, file)