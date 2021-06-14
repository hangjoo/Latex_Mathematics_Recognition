import editdistance
import numpy as np


def word_error_rate(predicted_outputs, ground_truths):
    sum_wer = 0.0
    for output, ground_truth in zip(predicted_outputs, ground_truths):
        output = output.split(" ")
        ground_truth = ground_truth.split(" ")
        distance = editdistance.eval(output, ground_truth)
        length = max(len(output), len(ground_truth))
        sum_wer += distance / length

    return sum_wer / len(predicted_outputs)


def sentence_acc(predicted_outputs, ground_truths):
    correct_sentences = 0
    for output, ground_truth in zip(predicted_outputs, ground_truths):
        if np.array_equal(output, ground_truth):
            correct_sentences += 1

    return correct_sentences / len(predicted_outputs)


def get_symbol_acc(predicted_outputs, ground_truths):
    correct_symbols = 0
    total_symbols = 0
    for output, ground_truth in zip(predicted_outputs, ground_truths):
        if len(output) < len(ground_truth):
            correct_symbols += sum([True if pr_ == gt_ else False for pr_, gt_ in zip(output, ground_truth[: len(output)])])
        else:
            correct_symbols += sum([True if pr_ == gt_ else False for pr_, gt_ in zip(output[: len(ground_truth)], ground_truth)])
        total_symbols += max(len(output), len(ground_truth))

    return correct_symbols / total_symbols
