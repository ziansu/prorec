import evaluate
import json
import os
import pickle
import re
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    'codellama/CodeLlama-7b-Instruct-hf',
    cache_dir='../save/.cache'
)

# bleu_score = evaluate.load('bleu', tokenizer=tokenizer, smooth=True)
# rouge_score = evaluate.load('rouge', tokenizer=tokenizer)
meteor_score = evaluate.load('meteor', tokenizer=tokenizer)
chrf_score = evaluate.load('chrf')


BOP_LEN = len('Start of Purpose\n')
EOP_LEN = len('End of Purpose\n')
def extract_summary_old(summary):
    pos = summary.find('[Start of Purpose]')
    bop_len = BOP_LEN + 2
    eop_len = EOP_LEN + 2
    if pos == -1:
        pos = summary.find('### Start of Purpose')
        bop_len = BOP_LEN + 4
        eop_len = EOP_LEN + 4
    if pos == -1:
        pos = summary.find('---\nStart of Purpose\n---')
        bop_len = BOP_LEN + 8
        eop_len = EOP_LEN + 8
    if pos == -1:
        pos = summary.find('**Purpose:**\n')
        bop_len = len('**Purpose:**\n')
        eop_len = 0
    if pos == -1:
        # print(summary)
        pos = -500    # cannot handle and fall back to fixed-length response
        pass
    summary = summary[pos:][bop_len: -eop_len]
    return summary


def extract_summary(response):
    # pos = response.find('**Purpose**:\n')
    pos = response.find('**Purpose**:')
    if pos == -1:
        pos = response.find('**Purpose:**')
    if pos == -1:
        pos = response.find('**Purpose**')
        return response[pos:][12:]
    if pos == -1:
        print(response)
        assert 0
    summary = response[pos:][13:]
    return summary

def extract_summary_for_gpt4evaluator(response):
    pos = response.find('**Purpose**:')
    if pos == -1:
        pos = response.find('**Purpose:**')
    if pos == -1:
        pos = response.find('**Purpose**')
        return response[pos:][12:]
    if pos == -1:
        print(response)
        assert 0
    summary = response[pos:][13:]
    return summary
    # pos = response.find('**Summary**:')
    # if pos == -1:
    #     pos = response.find('**Summary:**')
    # if pos == -1:
    #     print(response)
    #     assert 0
    # summary = response[pos:]
    # return summary


def parse_raw_results(dir, model_name, return_raw=False):
    with open(os.path.join(dir, f'srcsum_{model_name}_results.pkl'), 'rb') as f:
        srcsum_results = pickle.load(f)
    with open(os.path.join(dir, f'decsum_{model_name}_results.pkl'), 'rb') as f:
        decsum_results = pickle.load(f)
    with open(os.path.join(dir, f'parsum_{model_name}_results.pkl'), 'rb') as f:
        parsum_results = pickle.load(f)
    with open(os.path.join(dir, f'ragsum_{model_name}_results.pkl'), 'rb') as f:
        ragsum_results = pickle.load(f)
    # with open(os.path.join(dir, 'gpt4eval_par_vs_rag_results.pkl'), 'rb') as f:
    #     gpt4eval_results = pickle.load(f)

    agg_results = []
    for srcsum_res, decsum_res, parsum_res, ragsum_res in \
        zip(srcsum_results, decsum_results, parsum_results, ragsum_results):
        ref = extract_summary(srcsum_res.choices[0].message.content)
        pred_dec = extract_summary(decsum_res.choices[0].message.content)
        pred_pro = extract_summary(parsum_res.choices[0].message.content)
        pred_rag = extract_summary(ragsum_res.choices[0].message.content)
        agg_results.append(
            {
                'srcsum': ref,
                'decsum': pred_dec,
                'prosum': pred_pro,
                'ragsum': pred_rag,
            }
        )
    if return_raw:
        return agg_results, \
            {'srcsum': srcsum_results, 'decsum': decsum_results, 'parsum': parsum_results, 'ragsum': ragsum_results}
    else:
        return agg_results


prefix = 'The purpose of the function '
def post_process_binsum(summarization: str):
    # return summarization.split('\n')[0]
    # return summarization
    return summarization[len(prefix):]


gpt4eval_regex = re.compile(r"{'Q-A': [0-9], 'Q-B': [0-9]}")
gpt4eval_regex_qA = re.compile(r"{'Q-A': [0-9]}")
gpt4eval_regex_qB = re.compile(r"{'Q-B': [0-9]}")
gpt4eval_regex_qA_star = re.compile(r"\*\*Q-A\*\*: [0-9]")
gpt4eval_regex_qB_star = re.compile(r"\*\*Q-B\*\*: [0-9]")
def parse_gpt4eval(response):
    result = gpt4eval_regex.search(response)
    if result is not None:
        return eval(result.group(0))
    resultA = gpt4eval_regex_qA.search(response)
    resultB = gpt4eval_regex_qB.search(response)
    if resultA is not None and resultB is not None:
        return {
            'Q-A': eval(resultA.group(0))['Q-A'],
            'Q-B': eval(resultB.group(0))['Q-B']
        }
    resultA = gpt4eval_regex_qA_star.search(response)
    resultB = gpt4eval_regex_qB_star.search(response)
    if resultA is not None and resultB is not None:
        return {
            'Q-A': int(resultA.group(0).split(' ')[-1]),
            'Q-B': int(resultB.group(0).split(' ')[-1])
        }
    else:
        print(response)
        raise ValueError("Failed to parse GPT4 evaluation results.")


def evaluate_automatic_metrics(
    references,
    **predictions
):
    bleu_references = []
    roug_references = []
    for refsum in references:
        ref = post_process_binsum(refsum)
        bleu_references.append([ref])
        roug_references.append(ref)

    processed_predictions = {k: [] for k in predictions.keys()}
    for k, preds in predictions.items():
        for pred in preds:
            processed_predictions[k].append(
                post_process_binsum(pred))

    results = {k: {} for k in predictions.keys()}
    for k in predictions.keys():
        # results[k]['smoothed_bleu_score'] = bleu_score.compute(
        #     predictions=processed_predictions[k],
        #     references=bleu_references
        # )
        # results[k]['rouge_score'] = rouge_score.compute(
        #     predictions=processed_predictions[k],
        #     references=roug_references
        # )
        results[k]['meteor_score'] = meteor_score.compute(
            predictions=processed_predictions[k],
            references=roug_references
        )
        results[k]['chrf_score'] = chrf_score.compute(
            predictions=processed_predictions[k],
            references=bleu_references
        )
    print(json.dumps(results, indent=2))


def evaluate_file(filename):
    with open(filename, 'r') as f:
        results = json.load(f)
    evaluate_automatic_metrics(
        [res['srcsum'] for res in results],
        decsum=[res['decsum'] for res in results],
        prosum=[res['prosum'] for res in results],
        ragsum=[res['ragsum'] for res in results]
    )


def evaluate_detailed_automatic_metrics(
    references,
    **predictions
):
    bleu_references = []
    roug_references = []
    for refsum in references:
        ref = post_process_binsum(refsum)
        bleu_references.append([ref])
        roug_references.append(ref)

    processed_predictions = {k: [] for k in predictions.keys()}
    for k, preds in predictions.items():
        for pred in preds:
            processed_predictions[k].append(
                post_process_binsum(pred))

    results = []
    for i in range(len(bleu_references)):
        bleu_ref = [bleu_references[i]]
        roug_ref = [roug_references[i]]
        sp_pred = {k: [v[i]] for k, v in processed_predictions.items()}
        single_result = {k: {} for k in predictions.keys()}
        try:
            for k, preds in sp_pred.items():
                # single_result[k]['smoothed_bleu_score'] = bleu_score.compute(
                #     predictions=preds,
                #     references=bleu_ref
                # )
                # single_result[k]['rouge_score'] = rouge_score.compute(
                #     predictions=preds,
                #     references=roug_ref
                # )
                single_result[k]['meteor_score'] = meteor_score.compute(
                    predictions=preds,
                    references=roug_ref
                )
                single_result[k]['chrf_score'] = chrf_score.compute(
                    predictions=preds,
                    references=roug_ref
                )
            results.append(single_result)
        except ZeroDivisionError:
            print(k, preds)

    return results


if __name__ == '__main__':
    pass