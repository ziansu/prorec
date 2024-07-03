import evaluate

import json
import re

import sentencepiece as spm
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet
import nltk
from .utils import preprocess, lemmatize_name_tokens
from tqdm import tqdm
from typing import List, Dict


def camel_to_snake(name):
    """
    Convert a camelCase name to snake_case.
    """
    name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)  # Handles transitions like camelCase
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()  # Handles transitions like IDInCamel


fn_regex = re.compile(r'`[_A-Za-z0-9]+`')
fn_regex_p = re.compile(r'`[_A-Za-z0-9]+\(\)`')
def extract_function_name(response):
    "extract function name from LLM reponse"
    fn = fn_regex.findall(response)
    if fn:
        return fn[-1][1:-1]
    if '**Function Name**:' in response:
        fn = fn_regex_p.findall(response[response.find('**Function Name**:'):])
        print(fn)
        print(fn[0][1:-3])
        # print(response)
        # assert 0
        return fn[0][1:-3]
    if response.startswith('```'):
        fn = response.split('(')[0].split(' ')[-1]
        return fn
    print(response)
    assert 0


def extract_src_func_name(src_func):
    src_func_name = re.split(' |\n|\t', src_func.split('(')[0])
    if src_func_name[-1] == '':  # ... Member (long AllocSize)
        src_func_name = src_func_name[-2]
    else:
        src_func_name = src_func_name[-1]
    return src_func_name


sp = spm.SentencePieceProcessor()
# sp.load("models/dirty-vocab-all.model")
sp.load("evaluator/symlm_related/segmentation.model")

lem = WordNetLemmatizer()


word_cluster = json.load(open("evaluator/symlm_related/word_cluster.json", "r"))


word_cluster_fast = {}
for word, cluster in word_cluster.items():
    word_cluster_fast[word] = set()
    for idx in cluster:
        word_cluster_fast[word].add(idx)


def tokenize_name(name):
    preprocessed_name = preprocess(name)
    name_tokens = preprocessed_name.split()
    result_name_tokens = lemmatize_name_tokens(name_tokens)
    split = sp.encode_as_pieces(" ".join(result_name_tokens))    
    ret = []
    for w in split:
        if w.startswith("\u2581"):
            w = w[1:]
        ret.append(w)
    return ret


def _same_token(gt, pred):
    if gt == pred:
        return True
    shorter = gt if len(gt) < len(pred) else pred
    longer = gt if len(gt) >= len(pred) else pred
    if len(shorter) >= 2 and longer.startswith(shorter):
        return True
    if len(shorter) >= 2 and longer.endswith(shorter):
        return True    
    
    if gt in word_cluster_fast and pred in word_cluster_fast:
        return len(word_cluster_fast[gt] & word_cluster_fast[pred]) > 0
    return False


def score_name(gt, pred, dbg=False):
    # return precision and recall
    gt = gt.lower()
    pred = pred.lower()
    if gt == pred:
        if dbg:
            return 1, 1, "exact match"
        else:
            return 1, 1
    gt_tokens = tokenize_name(gt)
    if len(gt_tokens) == 0:
        print("gt_tokens is empty for gt = {}".format(gt))
        return 0, 0
                
    pred_tokens = tokenize_name(pred)
    if len(pred_tokens) == 0:
        print("pred_token is empty, pred = %s, gt = %s" % (pred, gt))
        return 0, 0
    matched_gt = set()
    matched_pred = set()
    for gt_t in gt_tokens:
        for pred_t in pred_tokens:
            if _same_token(gt_t, pred_t):
                matched_gt.add(gt_t)
                matched_pred.add(pred_t)
    precision = len(matched_pred) / len(pred_tokens)
    recall = len(matched_gt) / len(gt_tokens)
    if dbg:
        return precision, recall, {
                'matched_gt': matched_gt,
                'matched_pred': matched_pred,
                'gt_tokens': gt_tokens,
                'pred_tokens': pred_tokens
            }
    else:
        return precision, recall
            

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0:
                dp[i][j] = j
            elif j == 0:
                dp[i][j] = i
            elif s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
    return dp[m][n]


def _same_token_ori(gt, pred):
    if gt == pred:
        return True
    shorter = gt if len(gt) < len(pred) else pred
    longer = gt if len(gt) >= len(pred) else pred
    if longer.startswith(shorter):
        return True
    
    if gt in word_cluster_fast and pred in word_cluster_fast:
        return len(word_cluster_fast[gt] & word_cluster_fast[pred]) > 0
    
    if len(shorter) > 1 and shorter[0] == longer[0]:
        d = edit_distance(shorter, longer)
        dr = d / len(longer)
        if dr < 1/3:
            return True

    return False     


def score_name_ori(gt, pred, dbg=False):
    # return precision and recall
    gt = gt.lower()
    pred = pred.lower()
    if gt == pred:
        if dbg:
            return 1, 1, "exact match"
        else:
            return 1, 1
    gt_tokens = tokenize_name(gt)
    if len(gt_tokens) == 0:
        print("gt_tokens is empty for gt = {}".format(gt))
        return 0, 0
                
    pred_tokens = tokenize_name(pred)
    if len(pred_tokens) == 0:
        print("pred_token is empty, pred = %s, gt = %s" % (pred, gt))
        return 0, 0
    matched_gt = set()
    matched_pred = set()
    for gt_t in gt_tokens:
        for pred_t in pred_tokens:
            if _same_token_ori(gt_t, pred_t):
                matched_gt.add(gt_t)
                matched_pred.add(pred_t)
    precision = len(matched_pred) / len(pred_tokens)
    recall = len(matched_gt) / len(gt_tokens)
    if dbg:
        return precision, recall, {
                'matched_gt': matched_gt,
                'matched_pred': matched_pred,
                'gt_tokens': gt_tokens,
                'pred_tokens': pred_tokens
            }
    else:
        return precision, recall


def evaluate_func_names(
    references,
    **predictions
):
    """
    evaluate only underscore notation function names 
    (convert using `camel_to_snake`)
    """

    def _evaluate(refs, preds):
        precisions = []
        recalls = []
        f1s = []

        for ref, pred in zip(refs, preds):
            precision, recall = score_name(ref, pred)
            # precision, recall = score_name(ref, pred)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall > 0) else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        return {
            'precision': sum(precisions) / len(precisions),
            'recall': sum(recalls) / len(recalls),
            'f1': sum(f1s) / len(f1s),
        }
        

    results = {k: {} for k in predictions.keys()}
    for k in predictions.keys():
        results[k] = _evaluate(references, predictions[k])

    print(json.dumps(results, indent=2))


def evaluate_detailed_func_names(
    references,
    **predictions
):
    """
    evaluate only underscore notation function names 
    (convert using `camel_to_snake`)
    """

    def _evaluate(refs, preds):
        precisions = []
        recalls = []
        f1s = []

        for ref, pred in zip(refs, preds):
            precision, recall = score_name(ref, pred)
            # precision, recall = score_name(ref, pred)
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall > 0) else 0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)
        
        return {
            'precision': precisions,
            'recall': recalls,
            'f1': f1s,
        }
        

    results = {k: {} for k in predictions.keys()}
    for k in predictions.keys():
        results[k] = _evaluate(references, predictions[k])

    return results