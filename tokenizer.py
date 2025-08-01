import torch
import numpy as np
import torch.nn as nn
import re
from g2p_en import G2p
from oz2.text import text_to_sequence

class Tokenizer(nn.Module):
    def __init__(self, lexicon_path=None, cleaners=["english_cleaners"]):
        super(Tokenizer, self).__init__()
        if lexicon_path is None:
            lexicon_path = "/cm/archive/nghiahnh/OZSpeech2/oz2/lexicon/librispeech-lexicon.txt"

        self.lexicon = self.load_lexicon(lexicon_path)
        self.cleaners = cleaners
        self.g2p = G2p()
    
    def load_lexicon(self, lexicon_path):
        lexicon = {}
        with open(lexicon_path) as f:
            for line in f:
                temp = re.split(r"\s+", line.strip("\n"))
                word = temp[0]
                phones = temp[1:]
                if word.lower() not in lexicon:
                    lexicon[word.lower()] = phones
        return lexicon

    def forward(self, text):
        phones = []
        words = re.split(r"([,;.\-\?\!\s+])", text)
        for w in words:
            if w.lower() in self.lexicon:
                phones += self.lexicon[w.lower()]
            else:
                phones += list(filter(lambda p: p != " ", self.g2p(w)))
        phones = "{sp " + " ".join(phones) + "}"
        phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
        phones = phones.replace("}{", " ")
        sequence = np.array(text_to_sequence(phones, self.cleaners))
        sequence = torch.from_numpy(sequence).unsqueeze(0)
        return sequence

    def tokenize(self, text):
        return self.forward(text)