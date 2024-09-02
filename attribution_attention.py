###################################
'''
The code is based on inference for IG calculation of attention layer
'''
###################################
import argparse
import torch
import transformers
from transformers import StoppingCriteriaList, MaxLengthCriteria
from transformers import AutoTokenizer, PhiConfig, PhiForCausalLM, AutoModelForCausalLM
from datasets import load_dataset
import sacrebleu
import model.modeling_phi as modeling_phi
from tqdm import tqdm
import json

def main(args):
    torch.set_default_device("cuda")

    configuration = PhiConfig.from_json_file("phi/config.json")
    configuration.num_key_value_heads = configuration.num_attention_heads
    model = modeling_phi.PhiForCausalLM.from_pretrained(args.model_name, config=configuration)
    tokenizer = AutoTokenizer.from_pretrained("phi", trust_remote_code=True)

    if args.metric == 'gradient':
        from generation.utils_ghead import GenerationMixin
    else:
        from generation.utils_ighead import GenerationMixin

    def set_seed(seed):
        torch.manual_seed(seed)

    def generate_outputs(model, inputs_ids, solution, n_steps):
        combine_ig_list = []
        num_layer = 32

        # IG scores require per layer calculation
        for layer_idx in range(num_layer):
            outputs = GenerationMixin.greedy_search(
                model, inputs_ids, solution, 
                stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=inputs_ids.size(1) + solution.size(1))]), 
                heads_mask=None, return_dict_in_generate=True, output_scores=True, 
                baseline=base, mask_layer_list=[layer_idx], n_steps=n_steps, 
                pad_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id
            )
            combine_ig_list.append(outputs.ig_tensor)

        # Stack the ig_tensor for each layer
        ig_tensor = torch.stack(combine_ig_list, dim=1)[:, :, 0, :]
        return ig_tensor.detach()

    def preprocess(num_data, text_input_list, text_solution_list, tokenizer, model, base, n_steps):
        tensor_list = []
        for sample_idx in tqdm(range(num_data), desc="Processing"):
            text_input = "Instruct: " + text_input_list[sample_idx] + "\nOutput:"
            text_solution = text_solution_list[sample_idx]
            solution = tokenizer(text_solution, return_tensors="pt", return_attention_mask=False)
            inputs = tokenizer(text_input, return_tensors="pt", return_attention_mask=False)
            inputs_ids = inputs.input_ids
            solution = solution.input_ids

            ig_tensor = generate_outputs(model, inputs_ids, solution, n_steps)

            tensor_list.append(ig_tensor)

        return tensor_list

    set_seed(42)
    base = 0

    # Load dataset
    dataset_name = "truthful_qa"
    truthful_piqa_dataset = load_dataset(dataset_name, "generation")["validation"]
    text_input_list = truthful_piqa_dataset["question"]
    with open('labels.json', 'r') as file:
        text_solution_list = json.load(file)

    num_data = args.num_data

    if args.metric == 'ig':
        n_steps = args.n_steps
    elif args.metric in ['gxo', 'gradient']:
        n_steps = 1
    else:
        raise ValueError("Invalid metric. Choose either 'ig', 'gxo', or 'gradient'.")

    tensor_list = preprocess(num_data, text_input_list, text_solution_list, tokenizer, model, base, n_steps)
    output_path = f'result/truthfulqa/ig/{args.metric}_attention_list.pt'
    torch.save(tensor_list, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2', help='Name of the model')
    parser.add_argument('--num_data', type=int, default=100, help='Number of data samples to process')
    parser.add_argument('--metric', type=str, choices=['gxo', 'ig', 'gradient'], default='ig', help='Choose between gxo, ig, and gradient')
    parser.add_argument('--n_steps', type=int, default=20, help='Number of steps for IG calculation')

    args = parser.parse_args()

    main(args)
