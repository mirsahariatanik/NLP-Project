def compute_spans(seq):
    """Compute the spans in a BIO tag sequence.

    Parameters:
    - seq (list of str): the sequence of labels

    Returns:
    - list of (int, int): the list of span positions

    The span positions are Python-style, i.e., a span (X, i, j) means
    seq[i:j] is a span of type X.
    """
    spans = []
    state = i = None

    def flush():
        nonlocal state, i
        if state is not None:
            spans.append((state, i, j))
        state = i = None
    
    for j, label in enumerate(seq):
        if label.startswith('B-'):
            flush()
            state = label[2:]
            i = j
            
        elif label.startswith('I-'):
            if label[2:] == state:
                pass
            else:
                # This is not allowed, but we do our best to make sense of it
                flush()
                state = label[2:]
                i = j
                
        else:
            flush()
            state = None
    flush()
    return spans

def compute_f1(predict, correct):
    """Compute F1 score for BIO tag sequences.

    Parameters:
    - predict (list of list of str): the label sequences to score
    - correct (list of list of str): the correct label sequences

    It should be the case that len(predict) == len(correct), and for
    all i, len(predict[i]) == len(correct[i]).

    A span begins with label B-type and continues with one or more
    I-type labels, where type can be any string.

    All other labels are ignored.

    """

    if len(predict) != len(correct):
        raise ValueError(f'different number of predicted and correct label sequences ({len(predict)} != {len(correct)})')
    for li, (predict_seq, correct_seq) in enumerate(zip(predict, correct)):
        if len(predict_seq) != len(correct_seq):
            raise ValueError(f'line {li+1} has different number of words in predicted file and correct file')
        
    m = p = c = 0
    for predict_seq, correct_seq in zip(predict, correct):
        predict_spans = set(compute_spans(predict_seq))
        correct_spans = set(compute_spans(correct_seq))
        m += len(predict_spans & correct_spans)
        p += len(predict_spans)
        c += len(correct_spans)
    if m > 0:
        precision = m/p
        recall = m/c
        return 1/(((1/precision)+(1/recall))/2)
    else:
        return 0.

def read_labels(file):
    """Read words and labels from file.

    Parameters:
    - file: file object (not filename!) to read from

    The format of the file should be one sentence per line. Each line
    is of the form

    word1:label1 word2:label2 ...
    """
    
    ret = []
    for line in file:
        words = []
        labels = []
        for wordlabel in line.split():
            try:
                word, label = wordlabel.rsplit(':', 1)
            except ValueError:
                raise ValueError(f'invalid token {wordlabel}')
            words.append(word)
            labels.append(label)
        ret.append((words, labels))
    return ret

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print('usage: labels.py <predict> <correct>', file=sys.stderr)
        exit(1)
    predict_filename = sys.argv[1]
    correct_filename = sys.argv[2]

    predict = [labels for words, labels in read_labels(open(predict_filename))]
    correct = [labels for words, labels in read_labels(open(correct_filename))]

    print(compute_f1(predict, correct))
    
