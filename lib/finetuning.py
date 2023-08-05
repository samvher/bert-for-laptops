import torch

from datasets import load_dataset


def encode_sentence(sentence, bpe):
    """Take a string sentence and turn it into a list of BPE tokens."""
    encoded = []
    for atom in atomize(clean_string(sentence)):
        if atom.isalpha():
            encoded += [tok for tok in bpe.encode('_' + atom)]
        else:
            encoded.append(atom)
    return encoded

def prep_data(left_sentences, right_sentences, targets, bpe, length = 128, classification_target = True):
    """
    Take two lists of string sentences and a list of targets and generate Torch matrices for training.
    If the targets are not categorical (i.e. we're regressing), set classification_target = False.
    """
    assert len(left_sentences) == len(right_sentences) == len(targets)
    num_samples = len(left_sentences)
    tok2idx = bpe.token_mapping()
    xs = []
    ys = []
    skipped = 0
    for i in range(num_samples):
        left_encoded = encode_sentence(left_sentences[i], bpe)
        right_encoded = encode_sentence(right_sentences[i], bpe)
        x = ([tok2idx["[CLS]"]] + 
             [tok2idx[e] for e in left_encoded] +
             [tok2idx["[SEP]"]] +
             [tok2idx[e] for e in right_encoded] +
             [tok2idx["[PAD]"]] * (length - len(left_encoded) - len(right_encoded) - 2))
        if len(x) == length:
            xs.append(x)
            ys.append(targets[i])
        else:
            print(f"WARNING: Skipping sample of length {len(x)} at index {i}")
            skipped += 1
    print(f"Skipped {skipped} samples ({skipped/num_samples * 100}%)")
    joint = list(zip(xs, ys))
    random.shuffle(joint)
    xs, ys = zip(*joint)
    xs = torch.LongTensor(xs).to(device)
    if classification_target:
        ys = torch.LongTensor(ys).to(device)
    else:
        ys = torch.tensor(ys, device = device)
    return xs, ys

def finetune(bert, head, xs, ys):
    """
    Fairly simple training procedure going through xs and ys for 5 epochs.
    Batch size is constant, learning rate is warmed up and decayed but is constant per epoch.
    `bert` and `head` are modified in-place (you might not want to do this at home), this function does not return anything.
    """
    batch_size = 16
    total_samples = xs.shape[0]
    
    param_groups = [{'params': [p for p in list(bert.parameters()) + list(head.parameters()) if p.dim() >= 2], 'weight_decay': 0.01},
                    {'params': [p for p in list(bert.parameters()) + list(head.parameters()) if p.dim() < 2], 'weight_decay': 0}]
    optimizer = optim.AdamW(param_groups, lr = 4e-5, betas = (0.9, 0.98), eps = 1e-12, fused = True)
    scaler = GradScaler()
    
    # Poor man's warmup and decay
    lrs = [1e-5, 4e-5, 4e-5, 2e-5, 1e-5]
    
    for epoch in tqdm(range(5)):
        
        for g in optimizer.param_groups:
            g['lr'] = lrs[epoch]
                        
        i = 0
        while i < total_samples:

            batch_xs = xs[i:min(i+batch_size, total_samples), :]
            batch_ys = ys[i:min(i+batch_size, total_samples)]

            optimizer.zero_grad(set_to_none = True)

            with autocast(device_type='cuda', dtype=torch.float16):
                _, loss = head(bert(batch_xs), batch_ys)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            i += batch_size

def cls_predict(bert, cls_head, xs):
    """Take a trained BERT and CLSHead and generate predictions for the inputs xs."""
    pred = []
    for i in tqdm(range(xs.shape[0])):
        with torch.no_grad():
            logits, _ = cls_head(bert(xs[i:i+1]))
        pred.append(torch.argmax(logits))
    return torch.LongTensor(pred).to(device)

def reg_predict(bert, reg_head, xs):
    """Take a trained BERT and RegHead and generate predictions for the inputs xs."""
    pred = []
    for i in tqdm(range(xs.shape[0])):
        with torch.no_grad():
            y_hat, _ = reg_head(bert(xs[i:i+1]))
        pred.append(y_hat)
    return torch.tensor(pred, device = device)

def accuracy(pred, true):
    """Calculate accuracy from predictions and ground truth."""
    return (torch.sum(pred == true) / pred.shape[0]).item()

def f1(pred, true):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(pred)):
        if pred[i] and true[i]:
            tp += 1
        elif pred[i] and not true[i]:
            fp += 1
        elif not pred[i] and true[i]:
            fn += 1
        elif not pred[i] and not true[i]:
            tn += 1
    return 2 * tp / (2 * tp + fp + fn)

def mcc(pred, true):
    """Matthew's Correlation Coefficient"""
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(pred)):
        if pred[i] and true[i]:
            tp += 1
        elif pred[i] and not true[i]:
            fp += 1
        elif not pred[i] and true[i]:
            fn += 1
        elif not pred[i] and not true[i]:
            tn += 1
    return (tp*tn - fp*fn) / math.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))

def spearman(pred, true):
    """Return Spearman correlation for predictions and ground truth."""
    return scipy.stats.spearmanr(np.array(pred.cpu()), np.array(true.cpu())).correlation

def eval_rte(bert, bpe, length = 320):
    cls_head_rte = CLSHead(config, 2).to(device)
    
    rte_train = load_dataset("glue", "rte", split = "train")
    rte_train_xs, rte_train_ys = prep_data([s['sentence1'] for s in rte_train],
                                           [s['sentence2'] for s in rte_train],
                                           [s['label'] for s in rte_train],
                                           bpe,
                                           length = length) # Need this to accommodate the dataset

    finetune(bert, cls_head_rte, rte_train_xs, rte_train_ys)

    rte_val = load_dataset("glue", "rte", split = "validation")
    rte_val_xs, rte_val_ys = prep_data([s['sentence1'] for s in rte_val],
                                       [s['sentence2'] for s in rte_val],
                                       [s['label'] for s in rte_val],
                                       bpe,
                                       length = length)

    return accuracy(cls_predict(bert, cls_head_rte, rte_val_xs), rte_val_ys)

def eval_mrpc(bert, bpe):
    cls_head_mrpc = CLSHead(config, 2).to(device)
    
    mrpc_train = load_dataset("glue", "mrpc", split = "train")
    mrpc_train_xs, mrpc_train_ys = prep_data([s['sentence1'] for s in mrpc_train],
                                             [s['sentence2'] for s in mrpc_train],
                                             [s['label'] for s in mrpc_train],
                                             bpe)

    finetune(bert, cls_head_mrpc, mrpc_train_xs, mrpc_train_ys)

    mrpc_val = load_dataset("glue", "mrpc", split = "validation")
    mrpc_val_xs, mrpc_val_ys = prep_data([s['sentence1'] for s in mrpc_val],
                                         [s['sentence2'] for s in mrpc_val],
                                         [s['label'] for s in mrpc_val],
                                         bpe)

    return f1(cls_predict(bert, cls_head_mrpc, mrpc_val_xs), mrpc_val_ys)

def eval_stsb(bert, bpe, length = 192):
    """Take a pre-trained BERT, finetune on STS-B, and return performance."""
    reg_head_stsb = RegHead(config).to(device)
    
    stsb_train = load_dataset("glue", "stsb", split = "train")
    stsb_train_xs, stsb_train_ys = prep_data([s['sentence1'] for s in stsb_train],
                                             [s['sentence2'] for s in stsb_train],
                                             [s['label'] for s in stsb_train],
                                             bpe,
                                             length = length,
                                             classification_target = False)

    finetune(bert, reg_head_stsb, stsb_train_xs, stsb_train_ys)

    stsb_val = load_dataset("glue", "stsb", split = "validation")
    stsb_val_xs, stsb_val_ys = prep_data([s['sentence1'] for s in stsb_val],
                                         [s['sentence2'] for s in stsb_val],
                                         [s['label'] for s in stsb_val],
                                         bpe,
                                         length = length,
                                         classification_target = False)

    return spearman(reg_predict(bert, reg_head_stsb, stsb_val_xs), stsb_val_ys)

def eval_cola(bert, bpe):
    cls_head_cola = CLSHead(config, 2).to(device)
    
    cola_train = load_dataset("glue", "cola", split = "train")
    cola_train_xs, cola_train_ys = prep_data([s['sentence'] for s in cola_train],
                                             ['' for s in cola_train],
                                             [s['label'] for s in cola_train],
                                             bpe)

    finetune(bert, cls_head_cola, cola_train_xs, cola_train_ys)

    cola_val = load_dataset("glue", "cola", split = "validation")
    cola_val_xs, cola_val_ys = prep_data([s['sentence'] for s in cola_val],
                                         ['' for s in cola_val],
                                         [s['label'] for s in cola_val],
                                         bpe)

    return mcc(cls_predict(bert, cls_head_cola, cola_val_xs), cola_val_ys)

def eval_sst2(bert, bpe):
    """Take a pre-trained BERT, finetune on SST2, and return performance."""
    cls_head_sst2 = CLSHead(config, 2).to(device)
    
    sst2_train = load_dataset("glue", "sst2", split = "train")
    sst2_train_xs, sst2_train_ys = prep_data([s['sentence'] for s in sst2_train],
                                             ['' for s in sst2_train],
                                             [s['label'] for s in sst2_train],
                                             bpe)

    finetune(bert, cls_head_sst2, sst2_train_xs, sst2_train_ys)

    sst2_val = load_dataset("glue", "sst2", split = "validation")
    sst2_val_xs, sst2_val_ys = prep_data([s['sentence'] for s in sst2_val],
                                         ['' for s in sst2_val],
                                         [s['label'] for s in sst2_val],
                                         bpe)

    return accuracy(cls_predict(bert, cls_head_sst2, sst2_val_xs), sst2_val_ys)

def eval_qnli(bert, bpe, length = 256):
    cls_head_qnli = CLSHead(config, 2).to(device)
    
    qnli_train = load_dataset("glue", "qnli", split = "train")
    qnli_train_xs, qnli_train_ys = prep_data([s['sentence'] for s in qnli_train],
                                             [s['question'] for s in qnli_train],
                                             [s['label'] for s in qnli_train],
                                             bpe,
                                             length = length)

    finetune(bert, cls_head_qnli, qnli_train_xs, qnli_train_ys)

    qnli_val = load_dataset("glue", "qnli", split = "validation")
    qnli_val_xs, qnli_val_ys = prep_data([s['sentence'] for s in qnli_val],
                                         [s['question'] for s in qnli_val],
                                         [s['label'] for s in qnli_val],
                                         bpe,
                                         length = length)

    return accuracy(cls_predict(bert, cls_head_qnli, qnli_val_xs), qnli_val_ys)

def eval_qqp(bert, bpe, length = 192):
    cls_head_qqp = CLSHead(config, 2).to(device)
    
    qqp_train = load_dataset("glue", "qqp", split = "train")
    qqp_train_xs, qqp_train_ys = prep_data([s['question1'] for s in qqp_train],
                                           [s['question2'] for s in qqp_train],
                                           [s['label'] for s in qqp_train],
                                           bpe,
                                           length = length)

    finetune(bert, cls_head_qqp, qqp_train_xs, qqp_train_ys)

    qqp_val = load_dataset("glue", "qqp", split = "validation")
    qqp_val_xs, qqp_val_ys = prep_data([s['question1'] for s in qqp_val],
                                       [s['question2'] for s in qqp_val],
                                       [s['label'] for s in qqp_val],
                                       bpe,
                                       length = length)

    return f1(cls_predict(bert, cls_head_qqp, qqp_val_xs), qqp_val_ys)

def eval_mnli(bert, bpe, length = 256):
    cls_head_mnli = CLSHead(config, 3).to(device)
    
    mnli_train = load_dataset("glue", "mnli", split = "train")
    mnli_train_xs, mnli_train_ys = prep_data([s['premise'] for s in mnli_train],
                                             [s['hypothesis'] for s in mnli_train],
                                             [s['label'] for s in mnli_train],
                                             bpe,
                                             length = length)

    finetune(bert, cls_head_mnli, mnli_train_xs, mnli_train_ys)

    mnli_val_m = load_dataset("glue", "mnli_matched", split = "validation")
    mnli_val_m_xs, mnli_val_m_ys = prep_data([s['premise'] for s in mnli_val_m],
                                             [s['hypothesis'] for s in mnli_val_m],
                                             [s['label'] for s in mnli_val_m],
                                             bpe,
                                             length = length)

    mnli_val_mm = load_dataset("glue", "mnli_mismatched", split = "validation")
    mnli_val_mm_xs, mnli_val_mm_ys = prep_data([s['premise'] for s in mnli_val_mm],
                                               [s['hypothesis'] for s in mnli_val_mm],
                                               [s['label'] for s in mnli_val_mm],
                                               bpe,
                                               length = length)

    return (accuracy(cls_predict(bert, cls_head_mnli, mnli_val_m_xs), mnli_val_m_ys),
            accuracy(cls_predict(bert, cls_head_mnli, mnli_val_mm_xs), mnli_val_mm_ys))