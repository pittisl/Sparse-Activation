import argparse
import torch
import transformers
from transformers import AutoTokenizer, PhiConfig, PhiForCausalLM, AutoModelForCausalLM
from datasets import load_dataset
import model.modeling_phi_mask as modeling_phi
from generation.utils_mask import GenerationMixin
from tqdm import tqdm
import json
import sacrebleu
import scipy.io as scio
import matplotlib.pyplot as plt
import os

def main(args):
    torch.set_default_device("cuda")
    # Set a fixed seed
    torch.manual_seed(67)

    configuration = PhiConfig.from_json_file("phi/config.json")
    configuration.num_key_value_heads = configuration.num_attention_heads
    model = modeling_phi.PhiForCausalLM.from_pretrained(args.model_name, config=configuration)
    tokenizer = AutoTokenizer.from_pretrained("phi", trust_remote_code=True)

    # Load magnitude and attribution scores (IG or GxO, Gradient)
    mask_tensor_head1 = torch.load('result/truthfulqa/magnitude/magnitude_attention_list.pt', map_location=torch.device('cuda'))
    mask_tensor_head2 = torch.load(args.mask_tensor_head2_path, map_location=torch.device('cuda'))
    mask_tensor_head3 = torch.load('result/truthfulqa/ig/gradient_attention_list.pt', map_location=torch.device('cuda'))

    mask_tensor_mlp1 = torch.load('result/truthfulqa/magnitude/magnitude_mlp_list.pt', map_location=torch.device('cuda'))
    mask_tensor_mlp2 = torch.load(args.mask_tensor_mlp2_path, map_location=torch.device('cuda'))
    mask_tensor_mlp3 = torch.load('result/truthfulqa/ig/gradient_mlp_list.pt', map_location=torch.device('cuda'))

    # Load dataset
    dataset_name = "truthful_qa"
    truthful_piqa_dataset = load_dataset(dataset_name, "generation")["validation"]
    text_input_list = truthful_piqa_dataset["question"]
    with open('labels.json', 'r') as file:
        text_solution_list = json.load(file)

    base = 0
    bleu_all = 0
    num_data = args.num_data

    ml_list = [i for i in range(24)]

    bleu_res = []
    zero_res = []
    for per in tqdm(range(10), desc="Deactivation-ratio"):
        percentage = per * 0.1
        zero_ratio_list, bleu_all = preprocess(args.metric_name, percentage, mask_tensor_mlp1, mask_tensor_mlp2, mask_tensor_mlp3, mask_tensor_head1, mask_tensor_head2, mask_tensor_head3, text_input_list, text_solution_list, tokenizer, num_data, model, base, ml_list)
        bleu_res.append(bleu_all / num_data)
        zero_res.append(sum(zero_ratio_list) / len(zero_ratio_list))

    scio.savemat('result/truthfulqa/res/both/bleu_res_cor_gxo_both.mat', {"bleu_cor_gxo_both": bleu_res})
    scio.savemat('result/truthfulqa/res/both/zero_res_cor_gxo_both.mat', {"zero_cor_gxo_both": zero_res})

    # Plotting the curve
    plt.figure(figsize=(10, 6))
    plt.plot([1 - x for x in zero_res], bleu_res, linestyle='-', color='r', linewidth=5, marker='.', markersize=24, label='Cor-GxO')

    # Setting grid
    plt.grid(True)

    # Adding title and labels
    plt.xlim((0, 1.1))
    plt.ylim((0, 110))

    # Set x and y ticks
    plt.xticks([0.2, 0.4, 0.6, 0.8, 1.0], fontsize=20)
    plt.yticks([0, 20, 40, 60, 80, 100], fontsize=20)

    # Set axis labels
    plt.xlabel('Activation Ratio', fontsize=20)
    plt.ylabel('BLEU', fontsize=20)

    # Set legend
    plt.legend(fontsize=16, loc='lower right', bbox_to_anchor=(0.95, 0.05), ncol=2, borderaxespad=0.)

    # Save the image
    output_path = os.path.join("result/truthfulqa/res/both", "bleu_vs_zero.pdf")
    plt.savefig(output_path)

def compute_threshold(mask_tensor, percentile):
    a, b, c = mask_tensor.shape
    threshold_indices = torch.zeros((a, b), dtype=torch.float)
    for i in range(a):
        for j in range(b):
            sorted_tensor, _ = torch.sort(mask_tensor[i, j])
            idx = int(percentile * c)
            threshold_indices[i, j] = sorted_tensor[idx]
    return threshold_indices

def apply_threshold(mask_tensor, threshold_indices):
    a, b, c = mask_tensor.shape
    mask = torch.ones_like(mask_tensor)
    for i in range(a):
        for j in range(b):
            threshold = threshold_indices[i, j]
            mask[i, j] = (mask_tensor[i, j] >= threshold).float()
    masked_tensor = mask
    return masked_tensor

def compute_zero_ratio(mask_tensor):
    total_elements = mask_tensor.numel()
    num_zeros = torch.sum(mask_tensor == 0).item()
    zero_ratio = num_zeros / total_elements
    return zero_ratio

def generate_outputs(model, inputs_ids, solution, mask_mlp, mask_head, base, ml_list, tokenizer):
    outputs = GenerationMixin.greedy_search(model, inputs_ids, solution, baseline=base, max_length=inputs_ids.size(1) + solution.size(1), heads_mask=mask_head, mlp_mask=mask_mlp, return_dict_in_generate=True, output_scores=True, mask_layer_list=ml_list, pad_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id)
    return outputs

def preprocess(metric_name, percentage, mask_tensor_mlp1, mask_tensor_mlp2, mask_tensor_mlp3, mask_tensor_head1, mask_tensor_head2, mask_tensor_head3, text_input_list, text_solution_list, tokenizer, num_data, model, base, ml_list):
    zero_ratio_list = []
    bleu_all = 0
    for sample_idx in range(num_data):
        threshold1 = compute_threshold(abs(mask_tensor_mlp1[sample_idx]), percentage)
        mask_mlp1 = apply_threshold(abs(mask_tensor_mlp1[sample_idx]), threshold1)
        threshold2 = compute_threshold(-mask_tensor_mlp2[sample_idx], percentage)
        mask_mlp2 = apply_threshold(-mask_tensor_mlp2[sample_idx], threshold2)
        threshold3 = compute_threshold(-mask_tensor_mlp3[sample_idx], percentage)
        mask_mlp3 = apply_threshold(-mask_tensor_mlp3[sample_idx], threshold3) 
        
        threshold_head1 = compute_threshold(abs(mask_tensor_head1[sample_idx]), percentage)
        mask_head1 = apply_threshold(abs(mask_tensor_head1[sample_idx]), threshold_head1)
        threshold_head2 = compute_threshold(-mask_tensor_head2[sample_idx], percentage)
        mask_head2 = apply_threshold(-mask_tensor_head2[sample_idx], threshold_head2)
        threshold_head3 = compute_threshold(-mask_tensor_head3[sample_idx], percentage)
        mask_head3 = apply_threshold(-mask_tensor_head3[sample_idx], threshold_head2)

        if metric_name == "magnitude":
            mask_mlp = mask_mlp1
            mask_head = mask_head1
        elif metric_name == "gradient":
            mask_mlp = mask_mlp3
            mask_head = mask_head3
        elif metric_name == "gxo":
            mask_mlp = mask_mlp2
            mask_head = mask_head2
        elif metric_name == "snip":
            #calculate the mask based on snip scores
            threshold_snip = compute_threshold(abs(mask_tensor_mlp2[sample_idx]), percentage)
            mask_snip = apply_threshold(abs(mask_tensor_mlp2[sample_idx]), threshold_snip)
            
            threshold_head_snip = compute_threshold(abs(mask_tensor_head2[sample_idx]), percentage)
            mask_head_snip = apply_threshold(abs(mask_tensor_head2[sample_idx]), threshold_head_snip)
            
            mask_mlp = mask_snip
            mask_head = mask_head_snip
            
        elif metric_name == "cor_gxo":
            for i_correction in range(mask_tensor_head_correction.shape[0]):
                for j_correction in range(mask_tensor_head_correction.shape[1]):
                    #print(torch.sum(heads_mask_tensor_mlp2[sample_idx][i_correction,j_correction,:]))
                    mask_tensor_head_correction[i_correction,j_correction,:]=-(mask_tensor_head2[sample_idx][i_correction,j_correction,:])+0.5*abs(mask_tensor_head1[sample_idx]    [i_correction,j_correction,:])*torch.norm(mask_tensor_head3[sample_idx][i_correction,j_correction,:], p=2)  
                    mask_tensor_mlp_correction[i_correction,j_correction,:]=-mask_tensor_mlp2[sample_idx][i_correction,j_correction,:]+0.5*abs(mask_tensor_mlp1[sample_idx]    [i_correction,j_correction,:])*torch.norm(mask_tensor_mlp3[sample_idx][i_correction,j_correction,:], p=2)  
                
            threshold_head_correction = compute_threshold(mask_tensor_head_correction, percentage)
            mask_head_correction = apply_threshold(mask_tensor_head_correction, threshold_head_correction)

            threshold_mlp_correction = compute_threshold(mask_tensor_mlp_correction, percentage)
            mask_mlp_correction = apply_threshold(mask_tensor_mlp_correction, threshold_mlp_correction)
            
            mask_mlp = mask_mlp_correction
            mask_head = mask_head_correction
            
        else:
            print('no such metric')

        
        zero_ratio_attention = compute_zero_ratio(mask_head)
        zero_ratio_mlp = compute_zero_ratio(mask_mlp)
        zero_ratio_list.append(zero_ratio_attention / 3 + zero_ratio_mlp * 2 / 3)
        text_input = "Instruct: " + text_input_list[sample_idx] + "\nOutput:"
        text_solution = text_solution_list[sample_idx]
        solution = tokenizer(text_solution, return_tensors="pt", return_attention_mask=False)
        inputs = tokenizer(text_input, return_tensors="pt", return_attention_mask=False)
        inputs_ids = inputs.input_ids
        solution = solution.input_ids
        outputs = generate_outputs(model, inputs_ids, solution, mask_mlp, mask_head, base, ml_list, tokenizer)
        output_ids = outputs.sequences[:, inputs_ids.size(1):]
        text = tokenizer.batch_decode(output_ids)[0]
        candidate = [text]
        reference = [tokenizer.batch_decode(solution)[0]]
        bleu = sacrebleu.corpus_bleu(candidate, [reference])
        bleu_all += bleu.score
    return zero_ratio_list, bleu_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('--mask_tensor_head2_path', type=str, default='result/truthfulqa/magnitude/magnitude_attention_list.pt', help='Path to the mask tensor head2 file') #default='result/truthfulqa/ig/gxo_attention_list.pt'
    parser.add_argument('--mask_tensor_mlp2_path', type=str, default='result/truthfulqa/magnitude/magnitude_mlp_list.pt', help='Path to the mask tensor 2 file') #'result/truthfulqa/ig/gxo_mlp_list.pt'
    parser.add_argument('--model_name', type=str, default='microsoft/phi-2', help='Name of the model')
    parser.add_argument('--num_data', type=int, default=100, help='Number of data samples to process')
    parser.add_argument('--metric_name', type=str, default='magnitude', help='magnitude, gradient, gxo, snip, cor_gxo')    

    args = parser.parse_args()

    main(args)
