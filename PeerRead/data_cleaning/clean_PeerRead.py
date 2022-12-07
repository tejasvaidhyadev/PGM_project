
import argparse
import os

from process_PeerRead_abstracts import clean_PeerRead_dataset

dataset_names = ['acl_2017',
                  'arxiv.cs.ai_2007-2017',
                  'arxiv.cs.cl_2007-2017',
                  'arxiv.cs.lg_2007-2017',
                  'conll_2016',
                  'iclr_2017',
                  'nips_2013',
                  'nips_2014',
                  'nips_2015',
                  'nips_2016',
                  'nips_2017'
                  ]

dataset_paths = ['acl_2017',
                  'arxiv.cs.ai_2007-2017',
                  'arxiv.cs.cl_2007-2017',
                  'arxiv.cs.lg_2007-2017',
                  'conll_2016',
                  'iclr_2017',
                  'nips_2013-2017/2013',
                  'nips_2013-2017/2014',
                  'nips_2013-2017/2015',
                  'nips_2013-2017/2016',
                  'nips_2013-2017/2017'
                  ]

dataset_paths = dict(zip(dataset_names, dataset_paths))

dataset_years = {'acl_2017': 2017,
                  'conll_2016': 2016,
                  'iclr_2017': 2017,
                  'arxiv.cs.ai_2007-2017': None,
                  'arxiv.cs.cl_2007-2017': None,
                  'arxiv.cs.lg_2007-2017': None,
                  'nips_2013': 2013,
                  'nips_2014': 2014,
                  'nips_2015': 2015,
                  'nips_2016': 2016,
                  'nips_2017': 2017}

# dataset_venues = {k: v for v,k in enumerate(dataset_names)}

dataset_venues = {'acl_2017': 0,
                  'conll_2016': 1,
                  'iclr_2017': 2,
                  'nips_2013': 3,
                  'nips_2014': 3,
                  'nips_2015': 3,
                  'nips_2016': 3,
                  'nips_2017': 3,
                  'arxiv.cs.ai_2007-2017': 4,
                  'arxiv.cs.cl_2007-2017': 5,
                  'arxiv.cs.lg_2007-2017': 6,
                  }

def main():
    # todo basically generating dataset from the process readme