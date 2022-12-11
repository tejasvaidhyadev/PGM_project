import os
import json
import csv
import argparse

import pandas as pd
import numpy as np


def convert_to_int_columns(df, exclude=['post_text', 'response_text', 'score', 'controversiality', 'gilded',
                                        'created_utc']):
    df = df.astype({'score': np.int64, 'controversiality': np.int64, 'gilded': np.int64, 'created_utc': np.int64})

    for col in df.columns.tolist():
        if col in exclude:
            continue
        df[col] = pd.Categorical(df[col]).codes

    return df


def load_reddit(data_dir='data'):
    with open(os.path.join(data_dir, '2018.json'), 'r') as f:
        record_dicts = []
        for line in f.readlines():
            record = json.loads(line)
            reply_list = record['reply']
            earliest_reply_text = None
            for reply_dict in sorted(reply_list, key=lambda x: x['created_utc']):
                if reply_dict['body'] != '[deleted]':
                    earliest_reply_text = reply_dict['body']
                if earliest_reply_text:
                    break

            if earliest_reply_text:
                record.pop('reply')
                record['response_text'] = earliest_reply_text
                record_dicts.append(record)

    reddit_df = pd.DataFrame(record_dicts)
    reddit_df = reddit_df[reddit_df.body != '[deleted]']
    reddit_df = reddit_df.rename(columns={'body': 'post_text'})
    reddit_df = convert_to_int_columns(reddit_df)
    reddit_df = reddit_df.reset_index(drop=True)

    return reddit_df


def process_row_record(row_dict, random_response=None):
    context_features = {}

    if random_response:
        response_text = random_response
        context_features['has_random_resp'] = 1
    else:
        response_text = row_dict['response_text']
        context_features['has_random_resp'] = 0

    text_features = {'op_text': row_dict['post_text'],
                     'resp_text': response_text}

    for key in row_dict:
        if key not in {'post_text', 'response_text'}:
            context_features[key] = row_dict[key]

    return text_features, context_features


def process_reddit_dataset(data_dir, out_dir, out_file, subsample):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    reddit_df = load_reddit(data_dir)

    # add persistent record of the index of the data examples
    reddit_df['index'] = reddit_df.index

    reddit_records = reddit_df.to_dict('records')
    random_example_indices = np.arange(len(reddit_records))
    np.random.shuffle(random_example_indices)
    random_response_mask = np.random.randint(0, 2, len(reddit_records))

    with open(out_dir + "/" + out_file+ ".csv", 'w') as csv_file:
        csv_writer = csv.writer(csv_file)
        for idx, row_dict in enumerate(reddit_records):

            print("Processing example %d/%d" % (idx, len(reddit_records)))
            if subsample and idx >= subsample:
                break

            if (random_response_mask[idx]) and (random_example_indices[idx] != idx):
                random_response = reddit_records[random_example_indices[idx]]['response_text']
                text_features, context_features = process_row_record(row_dict,
                                                                     random_response=random_response)
            else:
                text_features, context_features = process_row_record(row_dict)

            if text_features and context_features:
                many_split = np.random.randint(0, 100)  # useful for easy data splitting later
                extra_context = {'many_split': many_split}
                context_features.update(extra_context)

                if idx == 0:
                    header = ['op_text', 'resp_text']
                    header.extend(context_features.keys())
                    csv_writer.writerow(header)

                row_ex = [text_features['op_text'], text_features['resp_text']]
                row_ex += [context_features[k] for k in header[2:]]
                csv_writer.writerow(row_ex)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--out_dir', type=str, default='data')
    parser.add_argument('--out_file', type=str, default='reddit_processed')
    parser.add_argument('--subsample', type=int, default=None)
    args = parser.parse_args()
    

    process_reddit_dataset(args.data_dir, args.out_dir, args.out_file, args.subsample)

    print('Done!')