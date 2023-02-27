from typing import *
import jieba
from tqdm import tqdm

def maximum_match_cut(text: str, vocab: set, max_size: int=4) -> List[Tuple[int, int]]:
    # maximum matching algo
    # Args:
    #   text: str, input text to be parsed
    #   vocab: set, word set
    #   max_size: considered maximum length of words
    # Returns:
    #   result: List[tuple], list of index pair indicating parsed words, e.g. [(0, 3), (3, 5), ...]
    result = []
    start, end = 0, 0
    while start < len(text):
        for end in range(min(len(text), start + max_size), start, -1):
            if text[start:end] in vocab:
                break
        result.append((start, end))
        start = end
    return result

def get_final_result(backward_result: List[Tuple], forward_result: List[Tuple]):
    # return final result given backward matching result and forward matching result
    # Args:
    #   backward_result: List[Tuple]
    #   forward_result: List[Tuple]
    # Returns:
    #   result: List[Tuple]
    result = (backward_result if (get_single := lambda x: len([i for i in x if i[1] - i[0] <= 1]))(backward_result) < get_single(forward_result) else forward_result) if len(forward_result) == len(backward_result) else (forward_result if len(forward_result) < len(backward_result) else backward_result)
    return result

def jieba_cut(valid_text: List[str]):
    # use jieba to cut
    # Args:
    #   valid_text: List[str]
    # Returns:
    #   jieba_result: List[List[Tuple]]
    jieba_words = [jieba.lcut(sent, cut_all=False) for sent in tqdm(valid_text)]
    jieba_result = []
    for words in jieba_words:
        jieba_result.append([])
        index = 0
        for word in words:
            jieba_result[-1].append((index, index + len(word)))
            index += len(word)
    return jieba_result


def evaluate(prediction: List[List[tuple]], target: List[List[tuple]]):
    # Span-level metric calculation, return precision, recall, and f1 
    # Args:
    #   prediction: List[List[tuple]], each tuple is an index pair indicating one parsed word
    #   target: List[List[tuple]], same as above
    # Returns:
    #   precision: float
    #   recall: float
    #   f1: float
    assert len(prediction) == len(target)
    prediction_size, target_size, correct_size = 0, 0, 0
    for i in range(len(prediction)):
        prediction_size += len(prediction[i])
        target_size += len(target[i])
        correct_size += len(set(prediction[i]) & set(target[i]))
    precision, recall = correct_size / prediction_size, correct_size / target_size
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1