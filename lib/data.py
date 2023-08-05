import pickle
import random
import re

from collections import Counter

import numpy as np

from unidecode import unidecode


### LOADING AND CLEANING

def clean_string(s):
    s = unidecode(s)             # Make sure we have only ASCII characters
    s = s.lower()                # Lowercase
    s = re.sub('[ \t]+', ' ', s) # Replace tabs and sequences of spaces with a single space
    s = s.replace('\n', '\\n')   # Escape newlines
    return s.strip()             # Remove leading and trailing whitespace

def preprocess_dataset(d, tag):
    c = Counter() # Will keep track of word counts
    
    # Save clean data to a local text file
    with open(f"{tag}.txt", "w") as f:
        for sample in d:
            s_clean = clean_string(sample['text'])
            f.write(s_clean + '\n')
            words = re.findall(r'[a-zA-Z]+', s_clean.replace('\\n', ' ')) # avoid capturing the 'n's of newlines
            c.update(words)
    
    # Pickle counts
    with open(f"{tag}_counts.pkl", "wb") as f:
        pickle.dump(c, f)

        
### BYTE PAIR ENCODING
        
alphabet = (["[CLS]", "[MASK]", "[SEP]", "[PAD]"] +
            [c for c in string.ascii_lowercase] +
            [f"_{c}" for c in string.ascii_lowercase] +
            [symbol for symbol in '0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'] +
            ["\\n"])

class BPEncoder:
    
    def __init__(self, alphabet, merge_rules, bpe_cache = dict()):
        """
        alphabet: a list of strings
        merge_rules: a list of string pairs to be merged
        """
        self.alphabet = alphabet
        self.merge_rules = merge_rules
        self.bpe_cache = bpe_cache
    
    def total_tokens(self):
        return len(self.alphabet) + len(self.merge_rules)
    
    def all_tokens(self):
        return self.alphabet + [a + b for a, b in self.merge_rules]

    def token_mapping(self):
        tokens = self.all_tokens()
        return {tok: i for i, tok in enumerate(tokens)}
    
    def add_merge_rule(self, merge_rule):
        self.merge_rules.append(merge_rule)
        
    def split_seq(self, s):
        """Split s into units from the alphabet."""
        t = sorted([a for a in alphabet if s.startswith(a)], key = lambda x: -len(x))[0]
        if len(t) < len(s):
            return [t] + self.split_seq(s[len(t):])
        else:
            return [t]
    
    def apply_merge_rule(self, merge_rule, bpe_seq):
        ret = []
        delta_dict = Counter()
        i = 0
        while i < len(bpe_seq) - 1:
            if merge_rule == (bpe_seq[i], bpe_seq[i+1]):
                ret.append(bpe_seq[i] + bpe_seq[i+1])
                
                # This part is a bit hairy and only really necessary for training the encoder (done further down).
                # It's essentially accounting logic to keep track of the occurrence of
                # sequential pairs: when we apply a merge rule, the merged pair disappears
                # everywhere in the sequence, but new pairs also appear. Keeping track of
                # that change this way is a bit more efficient than just re-counting all pairs.
                
                # Example:
                #   We have the sequence [t1, t2, t3, t4] and the merge rule (t2, t3).
                #   The pair (t2, t3) disappears, and pairs (t1, t2+t3) and (t2+t3, t4) appear.
                
                delta_dict.update({merge_rule: -1})
                if i > 0:
                    delta_dict.update({(ret[-2], bpe_seq[i]): -1})
                    delta_dict.update({(ret[-2], bpe_seq[i] + bpe_seq[i+1]): 1})
                if i < len(bpe_seq) - 2:
                    delta_dict.update({(bpe_seq[i+1], bpe_seq[i+2]): -1})
                    delta_dict.update({(bpe_seq[i] + bpe_seq[i+1], bpe_seq[i+2]): 1})
                    
                i += 2
            else:
                ret.append(bpe_seq[i])
                i += 1
        if i == len(bpe_seq) - 1:
            ret.append(bpe_seq[i])
        return ret, delta_dict
        
    def encode(self, s):
        """
        Apply BPE to s.
        This implementation is very slow for encodings with many merge rules.
        In our case that doesn't matter much, we will cache encodings.
        """
        if s in self.bpe_cache:
            return self.bpe_cache[s]
        else:
            ret = self.split_seq(s)
            for mr in self.merge_rules:
                ret, _ = self.apply_merge_rule(mr, ret)
            self.bpe_cache[s] = ret
            return ret

        
class BPETrainer:
    
    def __init__(self, word_counts, alphabet):
        self.bpe = BPEncoder(alphabet, [])
        
        self.data = [] # Will hold the encoded version of the words from our data with its count
        self.pair_counts = Counter() # Will hold occurrence frequencies of token pairs
        self.token_word_index = {token: [] for token in self.bpe.all_tokens()} # Maps tokens to the words in which they occur
        for i in range(len(word_counts)):
            word, count = word_counts[i]
            word_enc = self.bpe.split_seq('_' + word) # Prepend underscore to differentiate leading tokens
            self.data.append((word_enc, count))
            for j in range(0, len(word_enc) - 1):
                self.pair_counts.update({(word_enc[j], word_enc[j+1]): count})
            for tok in set(word_enc):
                self.token_word_index[tok].append(i)
    
    def add_merge_rule(self, t1, t2):
        """Adds the rule to merge t1 and t2 to the BPE and updates internal statistics."""
        
        # Add the new (merged) token to the word mapping
        self.token_word_index[t1 + t2] = []
            
        # The below code finds words that contain *both* t1 and t2 in a somewhat efficient way.
        # It relies on the fact that the list values in self.token_word_index are in sorted order.
        i = 0
        j = 0
        while i < len(self.token_word_index[t1]) and j < len(self.token_word_index[t2]):

            if self.token_word_index[t1][i] < self.token_word_index[t2][j]:
                i += 1

            elif self.token_word_index[t2][j] < self.token_word_index[t1][i]:
                j += 1

            else:
                # This word contains both t1 and t2: we might need to merge pairs here.
                
                word_idx = self.token_word_index[t1][i]
                word_enc, count = self.data[word_idx]

                # Get the encoded word after applying the merge rule, and the changes
                # that we need to make to our `pair_counts`.
                word_enc_post, delta = self.bpe.apply_merge_rule((t1, t2), word_enc)
                self.data[word_idx] = (word_enc_post, count)
                self.pair_counts.update({pair: d*count for pair, d in delta.items()})
                
                # Update the word index
                if t1 not in word_enc_post:
                    del self.token_word_index[t1][i]
                else:
                    i += 1
                if t2 not in word_enc_post:
                    if t2 != t1:
                        del self.token_word_index[t2][j]
                else:
                    j += 1
                if t1 + t2 in word_enc_post:
                    self.token_word_index[t1 + t2].append(word_idx)

        # Update the BPE to include the new merge rule
        self.bpe.add_merge_rule((t1, t2))
    
    def find_merge_rules(self, token_limit, verbose = False):
        """Add merge rules to the BPE until token_limit is reached."""
        
        while self.bpe.total_tokens() < token_limit:
            
            # Find the most frequent pair.
            # This call could be sped up with a better data structure.
            t1, t2 = max(self.pair_counts, key = self.pair_counts.get)
            count = self.pair_counts.get((t1, t2))
            
            if count == 0:
                print(f"No more tokens to add, every word has its own token already.")
                break
            
            if verbose:
                print(f"{self.bpe.total_tokens()}: {t1} + {t2} -> {t1}{t2} (count = {self.pair_counts.get((t1, t2))})")
            
            # Add the most frequent pair as a merge rule.
            self.add_merge_rule(t1, t2)
            

### CHUNKING

def atomize(s):
    """Break down a sample into symbols and words."""
    atom_re = r'(\[(CLS|SEP|PAD|MASK)\]|[a-z]+|\\n|[0-9]|\\|[!#$%&\'()*+,-./:;<=>?@[\]^_`{|}~])'
    return [m[0] for m in re.findall(atom_re, s)]
    
def chunks(fname, bpe, max_length, merge_lines = False):
    ret_list = []
    ret_tok_len = 0
        
    with open(fname, "r") as f:
        for line in f:
            atoms = atomize(line)
            for atom in atoms:
                if atom.isalpha():
                    # Deal with some weird sequences in the training data
                    if len(bpe.encode('_' + atom)) > max_length:
                        continue
                        
                    if ret_tok_len + len(bpe.encode('_' + atom)) > max_length:
                        yield ' '.join(ret_list)
                        ret_list = []
                        ret_tok_len = 0
                    ret_list.append(atom)
                    ret_tok_len += len(bpe.encode('_' + atom))
                else:
                    if ret_tok_len == max_length:
                        yield ' '.join(ret_list)
                        ret_list = []
                        ret_tok_len = 0
                    ret_list.append(atom)
                    ret_tok_len += 1
            if not merge_lines:
                yield ' '.join(ret_list)
                ret_list = []
                ret_tok_len = 0
    yield ' '.join(ret_list)


### SAMPLE GENERATION

def encode_sample(sample, bpe):
    encoded = []
    for item in sample.strip().split(' '):
        if item.isalpha():
            encoded += [tok for tok in bpe.encode('_' + item)]
        else:
            encoded.append(item)
    return encoded

def samples_and_masks(fname, length, bpe):
    """Assumes that all samples in fname have been sized not to exceed `length`."""
    tok2idx = bpe.token_mapping()
    
    with open(fname, "r") as f:
        for sample in f:
            
            # Apply BPE to the sample
            encoded = [tok2idx[e] for e in encode_sample(sample, bpe)]
            total_tokens = len(encoded)
            
            # Generate mask in the shape of the sample
            mask_count = math.ceil(0.15 * total_tokens)
            mask = [1] * mask_count + [0] * (total_tokens - mask_count)
            random.shuffle(mask)
            
            # Generate ground truth and mask in matching shape
            training_output = [tok2idx["[CLS]"]] + encoded + [tok2idx["[SEP]"]] + [tok2idx["[PAD]"]] * (length - total_tokens - 2)
            training_mask = [0] + mask + [0] + [0] * (length - total_tokens - 2)
            
            # Generate input data
            training_input = [t for t in training_output]
            for i in range(length):
                if training_mask[i] == 1: # Mask this token
                    r = random.random()
                    if r < 0.8:   # Regular masking
                        training_input[i] = tok2idx["[MASK]"]
                    elif r < 0.9: # Random other token instead of [MASK]
                        training_input[i] = random.randrange(bpe.total_tokens())
                    else:         # Feed the original token as input, untouched
                        pass
            
            yield [training_input, training_output, training_mask]

def samples_and_masks_spanbert(fname, length, bpe):
    """Assumes that all samples in fname have been sized not to exceed `length`."""
    tok2idx = bpe.token_mapping()
    
    with open(fname, "r") as f:
        for sample in f:
            
            # Apply BPE to the sample, here we don't flatten the list
            encoded = [bpe.encode('_' + item) if item.isalpha() else [item]
                       for item in sample.strip().split(' ')]
            total_items = len(encoded)
            total_tokens = sum([len(e) for e in encoded])
            
            # Generate span mask
            mask = [[0] * len(e) for e in encoded]
            total_masked = 0
            while total_masked < 0.15 * total_tokens:
                # Determine mask location
                # Masks can overlap, we'll take that into account later
                # That _does_ mean we stray slightly from the geometric distribution
                mask_location = np.random.randint(0, len(encoded))
                
                # Determine mask length
                mask_length = np.random.geometric(0.2)
                # Re-sample if too high (paper says clipping, but Figure 2 suggests this is the approach)
                while mask_length > min(len(encoded) - mask_location, 10):
                    mask_length = np.random.geometric(0.2)
                
                # Apply mask
                for i in range(mask_location, mask_location + mask_length):
                    if mask[i][0] != 1: # Masks can overlap/collide, correct for that
                        total_masked += len(encoded[i])
                    mask[i] = [1] * len(encoded[i])
            
            encoded = [tok2idx[t] for e in encoded for t in e]
            mask = [b for m in mask for b in m]
            
            # Generate ground truth and mask in matching shape
            training_output = [tok2idx["[CLS]"]] + encoded + [tok2idx["[SEP]"]] + [tok2idx["[PAD]"]] * (length - total_tokens - 2)
            training_mask = [0] + mask + [0] + [0] * (length - total_tokens - 2)
            
            # Generate input data
            training_input = [t for t in training_output]
            for i in range(length):
                if training_mask[i] == 1: # Mask this token
                    if i == 0 or training_mask[i-1] == 0: # Use the same strategy throughout the span
                        r = random.random()
                    if r < 0.8:   # Regular masking
                        training_input[i] = tok2idx["[MASK]"]
                    elif r < 0.9: # Random other token instead of [MASK]
                        training_input[i] = random.randrange(bpe.total_tokens())
                    else:         # Feed the original token as input, untouched
                        pass
            
            yield [training_input, training_output, training_mask]