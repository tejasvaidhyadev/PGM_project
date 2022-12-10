# script to create semi-sythetic data for the project
## Only use for the generation of data of PEERREAD dataset


# read the above csv file in pandas 
import pandas as pd
import numpy as np
from numpy.random import normal, uniform
import argparse

parser = argparse.ArgumentParser()
#parser.add_argument('--exp_name', type=str, default='experiment')
parser.add_argument("--raw_csv", default="process_data/arxiv-all.tf_record.csv", type=str, required=False,
                    help="raw csv containing all the information about data")
parser.add_argument("--save_csv", default="process_data/", type=str, required=False,
                    help="semi-Synthetic CSV file")

parser.add_argument("--treat_strength", default=0.25, type=int, required=False,
                    help="beta0 Treat_strength.")
parser.add_argument("--con_strength", default=5.0, type=int, required=False,
                    help="beta1 Confounding str ength. Also, refer to as gamma in the paper")

parser.add_argument("--noise_level", default=1.0, type=int, required=False,
                    help="gamma in the paper")

args = parser.parse_args()
def make_buzzy_based_simulated_labeler(treat_strength, con_strength, noise_level, setting="simple", seed=0):
    # hardcode probability of theorem given buzzy / not_buzzy
    theorem_given_buzzy_probs = np.array([0.27, 0.07], dtype=np.float32)

    np.random.seed(seed)
    all_noise = np.array(normal(0, 1, 12000), dtype=np.float32)
    all_threshholds = np.array(uniform(0, 1, 12000), dtype=np.float32)

    def labeler(data):
        buzzy = data['buzzy_title']
        index = data['index']
        treatment = data['theorem_referenced']
        treatment = np.float32(treatment)
        confounding = 3.0 * (theorem_given_buzzy_probs[buzzy] - 0.25)

        noise = all_noise[index]

        y, y0, y1 = _outcome_sim(treat_strength, con_strength, noise_level, treatment, confounding, noise,
                                 setting=setting)
        # generating binary 
        simulated_prob = 1 / (1 + np.exp(-y))

        y0 = 1 / (1 + np.exp(-y0))
        y1 = 1 / (1 + np.exp(-y1))
        threshold = all_threshholds[index]
        simulated_outcome = np.int32(simulated_prob > threshold)

        return {**data, 'outcome': simulated_outcome, 'y0': y0, 'y1': y1, 'treatment': treatment}

    return labeler

def _outcome_sim(beta0, beta1, gamma, treatment, confounding, noise, setting="simple"):
    # same as descirbe in paper
    if setting == "simple":
        y0 = beta1 * confounding
        y1 = beta0 + y0
        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    elif setting == "multiplicative":
        y0 = beta1 * confounding
        y1 = beta0 * y0

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    elif setting == "interaction":
        # required to distinguish ATT and ATE
        y0 = beta1 * confounding
        y1 = y0 + beta0 * tf.math.square(confounding)

        simulated_score = (1. - treatment) * y0 + treatment * y1 + gamma * noise

    else:
        raise Exception('setting argument to make_simulated_labeler not recognized')

    return simulated_score, y0, y1

def make_extra_feature_cleaning(df):
    # cleaning of extra features from csv file
    df['num_authors'] = np.minimum(df['num_authors'], 6) - 1
    df['year'] = df['year'] - 2007

    # some extras
    equation_referenced = np.minimum(df['num_ref_to_equations'], 1)
    theorem_referenced = np.minimum(df['num_ref_to_theorems'], 1)

    # buzzy title
    any_buzz = df["title_contains_deep"] + df["title_contains_neural"] + \
                df["title_contains_embedding"] + df["title_contains_gan"]
    buzzy_title = np.not_equal(any_buzz, 0).astype(int)

    buzzy_title = np.not_equal(any_buzz, 0).astype(int)
    # add equation_referenced, theorem_referenced, buzzy_title to df
    df['equation_referenced'] = equation_referenced
    df['theorem_referenced'] = theorem_referenced
    df['buzzy_title'] = buzzy_title
    df['index'] = df['id']
    return df
def dict_to_csv(dic, output_dir):
    # save the data
    df = pd.DataFrame.from_dict(dic)
    df.to_csv(output_dir, index=False)
    print("saved data to {}".format(output_dir))
    
def main(filename, output_dir):
    # get the data
    df = pd.read_csv(filename)

    df = make_extra_feature_cleaning(df)
    # get the labeler
    labeler = make_buzzy_based_simulated_labeler(treat_strength=args.treat_strength, con_strength=args.con_strength, noise_level=args.noise_level, setting="simple", seed=0)
    # apply the labeler
    # here df will be dict
    df = labeler(df)
    dict_to_csv(df, output_dir)
    # save the data

if __name__ == "__main__":
    # data is generated using PeerRead Library 
    raw_data = args.raw_csv
    save_data = args.save_csv + "beta0_" +str(args.treat_strength) + "beta1_" +str(args.con_strength) + "gamma_" + str(args.noise_level) + ".csv"
    main(raw_data, save_data)
