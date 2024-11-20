# Use model.pth to summarize new text and return the summary
# Usage: python summarizer.py -m model.pth -t "text to summarize"

import argparse
import torch
from transformers import BertTokenizer


def summarize(model, tokenizer, text):
    input_ids = tokenizer.encode(text, return_tensors='pt')
    output = model.generate(input_ids, max_length=100, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(output[0], skip_special_tokens=True)
    return summary




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True, help='Path to model.pth')
    parser.add_argument('-t', '--text', required=True, help='Text to summarize')
    args = parser.parse_args()

    model = torch.load(args.model)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    summary = summarize(model, tokenizer, args.text)
    print(summary)
