import argparse
import glob
import os
import random

import io
import sys
import json
from dateutil.parser import parse as parse_date

import csv
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from code.ScienceParse.Paper import Paper
from code.ScienceParse.ScienceParseReader import ScienceParseReader

from PeerRead_hand_features import get_PeerRead_hand_features

rng = random.Random(0)


def process_json_paper(paper_json_filename, scienceparse_dir):
    paper = Paper.from_json(paper_json_filename)
    paper.SCIENCEPARSE = ScienceParseReader.read_science_parse(paper.ID, paper.TITLE, paper.ABSTRACT,
                                                               scienceparse_dir)

    # Will handle tokenization form hugging face
    try:
        title_tokens = paper.TITLE
    except ValueError:  # missing titles are quite common sciparse
        print("Missing title for " + paper_json_filename)
        title_tokens = None

    abstract_tokens = paper.ABSTRACT

    text_features = {'title': title_tokens,
                     'abstract': abstract_tokens}

    context_features = {'authors': paper.AUTHORS,
                        'accepted': paper.ACCEPTED,
                        'name': paper.ID}
    # add hand crafted features from PeerRead
    pr_hand_features = get_PeerRead_hand_features(paper)
    context_features.update(pr_hand_features)

    return text_features, context_features


venues = {'acl': 1,
          'conll': 2,
          'iclr': 3,
          'nips': 4,
          'icml': 5,
          'emnlp': 6,
          'aaai': 7,
          'hlt-naacl': 8,
          'arxiv': 0}


def _venues(venue_name):
    if venue_name.lower() in venues:
        return venues[venue_name.lower()]
    else:
        return -1


def _arxiv_subject(subjects):
    subject = subjects[0]
    if 'lg' in subject.lower():
        return 0
    elif 'cl' in subject.lower():
        return 1
    elif 'ai' in subject.lower():
        return 2
    else:
        raise Exception("arxiv subject not recognized")


def clean_PeerRead_dataset(review_json_dir, parsedpdf_json_dir,
                           venue, year,
                           out_dir, out_file,
                           max_abs_len,
                           default_accept=1,
                           is_arxiv = False):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    print('Reading reviews from...', review_json_dir)
    paper_json_filenames = sorted(glob.glob('{}/*.json'.format(review_json_dir)))
    
    with open(out_dir + "/" + out_file+ ".csv", 'w') as csv_file:
            csv_writer = csv.writer(csv_file)
        
            for idx, paper_json_filename in enumerate(paper_json_filenames):
                text_features, context_features = process_json_paper(paper_json_filename, parsedpdf_json_dir)

                if context_features['accepted'] is None:
                    context_features['accepted'] = default_accept
                print("here")
                many_split = rng.randint(0, 100)  # useful for easy data splitting later

                # other context features
                arxiv = -1
                try:
                    if is_arxiv:
                        with io.open(paper_json_filename) as json_file:
                            loaded = json.load(json_file)
                        year = parse_date(loaded['DATE_OF_SUBMISSION']).year
                        venue = _venues(loaded['conference'])
                        arxiv = _arxiv_subject([loaded['SUBJECTS']])
                    extra_context = {'id': idx, 'venue': venue, 'year': year, 'many_split': many_split,
                                        'arxiv': arxiv}
                    context_features.update(extra_context)
                    # csv writer
                    if idx == 0:
                        header = ['title', 'abstract']
                        header.extend(context_features.keys())
                        csv_writer.writerow(header)

                    # now context_features and text_features are dictionaries
                    # save it in csv files with keys as column names and values as values
                    new_text_features = [text_features['title'], text_features['abstract']]
                    # use for loop fo new_context_features
                    new_context_features = [context_features[k] for k in header[2:]]
                    paper_ex = new_text_features + new_context_features

                    csv_writer.writerow(paper_ex)
                    print("done")
                except:
                    print("buggy")

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--review-json-dir', type=str, default='../data/all/train/reviews/')
    parser.add_argument('--parsedpdf-json-dir', type=str, default='../data/all/train/parsed_pdfs/')
    parser.add_argument('--out-dir', type=str, default='../dat/PeerRead/proc')
    parser.add_argument('--out-file', type=str, default='arxiv-all')
    #parser.add_argument('--vocab-file', type=str, default='../../bert/pre-trained/uncased_L-12_H-768_A-12/vocab.txt')
    parser.add_argument('--max-abs-len', type=int, default=250)
    parser.add_argument('--venue', type=int, default=0)
    parser.add_argument('--year', type=int, default=2017)


    args = parser.parse_args()

    clean_PeerRead_dataset(args.review_json_dir, args.parsedpdf_json_dir,
                           args.venue, args.year,
                           args.out_dir, args.out_file,
                           args.max_abs_len, is_arxiv=True)


if __name__ == "__main__":
    main()