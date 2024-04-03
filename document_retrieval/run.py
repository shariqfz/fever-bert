#!/usr/bin/env python3

# Referenced from https://github.com/UKPLab/fever-2018-team-athene/blob/master/src/athene/retrieval/document/docment_retrieval.py


import argparse
import json
import os
import sys
sys.path.append(os.getcwd())
import re
import time
import unicodedata
from multiprocessing.pool import ThreadPool

import nltk
import spacy
from tqdm import tqdm

from utils.doc_db import WikiDB


def processed_line(method, line):
    nps, pages = method.exact_match(line)
    line["noun_phrases"] = nps
    line["predicted_pages"] = pages
    return line


def process_line_with_progress(method, line, progress=None):
    if progress is not None and line["id"] in progress:
        return progress[line["id"]]
    else:
        return processed_line(method, line)


class Doc_Retrieval:
    def __init__(self, database_path, add_claim=False, max_pages_per_query=None):
        self.db = WikiDB(database_path)
        self.add_claim = add_claim
        self.max_pages_per_query = max_pages_per_query
        self.proter_stemm = nltk.PorterStemmer()
        self.tokenizer = nltk.word_tokenize
        self.nlp = spacy.load('en_core_web_sm')
    
    def get_noun_phrases(self, line):
        claim = line["claim"]
        doc = self.nlp(claim)
        noun_phrases = [chunk.text for chunk in doc.noun_chunks if len(chunk.text) > 1]
        named_entities = [entity.text for entity in doc.ents if len(entity.text) > 1]
        vbz = [token.text for token in doc if token.tag_ == 'VBZ' if len(token.text) > 1]
        if self.add_claim:
            noun_phrases.append(claim)
        return list(set(noun_phrases + named_entities))


    def np_conc(self, noun_phrases):
        noun_phrases = set(noun_phrases)
        predicted_pages = []
        for np in noun_phrases:
            page = np.replace("( ", "-LRB-")
            page = page.replace(" )", "-RRB-")
            page = page.replace(" - ", "-")
            page = page.replace(" :", "-COLON-")
            page = page.replace(" ,", ",")
            page = page.replace(" 's", "'s")
            page = page.replace(" ", "_")

            if len(page) < 1:
                continue
            doc_lines = self.db.get_doc_lines(page)
            if doc_lines is not None:
                predicted_pages.append(page)
        return predicted_pages

    def exact_match(self, line):
        noun_phrases = self.get_noun_phrases(line)

        claim = unicodedata.normalize("NFD", line["claim"])
        claim = claim.replace(".", "")
        claim = claim.replace("-", " ")
        words = [self.proter_stemm.stem(word.lower()) for word in self.tokenizer(claim)]
        words = set(words)
        predicted_pages = self.np_conc(noun_phrases)
        predicted_pages = list(set(predicted_pages))
        
        return noun_phrases, predicted_pages


def get_map_function(parallel, p=None):
    assert (
        not parallel or p is not None
    ), "A ThreadPool object should be given if parallel is True"
    return p.imap_unordered if parallel else map


def main(db_file, max_pages_per_query, in_file, out_file, add_claim=True, parallel=True):
    method = Doc_Retrieval(
        database_path=db_file, add_claim=add_claim, max_pages_per_query=max_pages_per_query
    )
    processed = dict()
    path = os.getcwd()
    lines = []
    with open(os.path.join(path, in_file), "r") as f:
        lines = [json.loads(line) for line in f.readlines()]
    if os.path.isfile(os.path.join(path, out_file + ".progress")):
        with open(os.path.join(path, out_file + ".progress"), "rb") as f_progress:
            import pickle

            progress = pickle.load(f_progress)
            print(
                os.path.join(path, out_file + ".progress")
                + " exists. Load it as progress file."
            )
    else:
        progress = dict()

    try:
        with ThreadPool(processes=4 if parallel else None) as p:
            for line in tqdm(
                get_map_function(parallel, p)(
                    lambda l: process_line_with_progress(method, l, progress), lines
                ),
                total=len(lines),
            ):
                processed[line["id"]] = line
                progress[line["id"]] = line
               
        with open(os.path.join(path, out_file), "w+") as f2:
            for line in lines:
                f2.write(json.dumps(processed[line["id"]]) + "\n")
    finally:
        with open(os.path.join(path, out_file + ".progress"), "wb") as f_progress:
            import pickle

            pickle.dump(progress, f_progress, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-file", type=str,
                        help="database file which contains wiki pages")
    parser.add_argument("--in-file", type=str, help="input dataset")
    parser.add_argument("--out-file", type=str,
                        help="path to save output dataset")
    parser.add_argument("--max-pages-per-query", type=int,
                        help="first k pages for wiki search")
    parser.add_argument("--parallel", type=bool, default=True)
    parser.add_argument("--add-claim", type=bool, default=True)
    args = parser.parse_args()

    nltk.download("punkt", quiet=True)

    main(
        args.db_file,
        args.max_pages_per_query,
        args.in_file,
        args.out_file,
        args.add_claim,
        args.parallel,
    )
