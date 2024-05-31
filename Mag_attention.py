###################################
'''
The code is for truthful_qa magnitude of attention layer generation
'''
###################################
import torch
import transformers
from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoTokenizer, PhiConfig, PhiForCausalLM, AutoModelForCausalLM
from  datasets  import  load_dataset
import model.modeling_phi as modeling_phi
from generation.utils_normal import GenerationMixin
from tqdm import tqdm
import json

#torch.set_safetensors_enabled(True)

torch.set_default_device("cuda")

configuration = PhiConfig.from_json_file("phi/config.json")
configuration.num_key_value_heads=configuration.num_attention_heads
model= modeling_phi.PhiForCausalLM.from_pretrained("microsoft/phi-2", config=configuration)
tokenizer = AutoTokenizer.from_pretrained("phi", trust_remote_code=True)

# set the hyperparameters
number_layers=32
features_out_hook = []
magnitude_list=[]

#use hook to get the output of the 
def hook(module, fea_in, fea_out):
    # Only take the forward pass outputs
    features_out_hook.append(fea_out)
    return None


# Register hooks for all attention_layers
for mask_layer_idx in range(number_layers):
    for name, module in model.named_modules():
        if name == "model.layers."+str(mask_layer_idx)+".self_attn.head_layer":
             module.register_forward_hook(hook=hook)


#load dataset
dataset_name = "truthful_qa"

truthful_piqa_dataset = load_dataset(dataset_name, "generation")["validation"]
text_input_list = truthful_piqa_dataset["question"]
with open('labels.json', 'r') as file:
    text_solution_list = json.load(file)

# Set a fixed seed
torch.manual_seed(42)
base=0
num_data=100 #len(text_input_list)

for sample_idx in tqdm(range(num_data), desc="Obtaining magnitude"):
    features_out_hook=[]
    
    text_input="Instruct: "+text_input_list[sample_idx]+"\nOutput:"
    text_solution=text_solution_list[sample_idx]

    solution = tokenizer(text_solution, return_tensors="pt", return_attention_mask=False)
    inputs = tokenizer(text_input, return_tensors="pt", return_attention_mask=False)
    inputs_ids=inputs.input_ids
    solution = solution.input_ids

    outputs = GenerationMixin.greedy_search(model, inputs_ids, solution, stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=inputs_ids.size(1)+solution.size(1))]), return_dict_in_generate=True, output_scores=True, baseline=base, mask_layer_list=None, heads_mask=None, pad_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)

    output_ids = outputs.sequences[:, inputs_ids.size(1):]  # Start from the first token after the input part.
    text = tokenizer.batch_decode(output_ids)[0]

    #reshape the output and get the output magnitude
    mean_tensors = [torch.norm(tensor[:, :, -1, :], p=2, dim=2)[0] for tensor in features_out_hook]
    
    #stack the magnitude of each token
    stacked_tensor = torch.stack(mean_tensors)

    # reshape the list into tensor with size number_token*number_layer*number_head 
    reshaped_tensor = stacked_tensor.view(output_ids.size(1), number_layers, 32)
    
    # add to the list of each sample
    magnitude_list.append(reshaped_tensor.detach())


#save the magnitude list as a file of .mat
torch.save(magnitude_list, 'result/truthfulqa/magnitude/magnitude_attention_list.pt')