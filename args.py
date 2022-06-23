from argparse import ArgumentParser
import numpy as np
parser = ArgumentParser(description='GRACE: Generating Concise and Informative Contrastive Sample')
parser.add_argument('--csv', type=str, default="Covid.csv")
parser.add_argument('--n_estimators', type=int, default=600)
parser.add_argument('--max_depth', type=int, default=3)
parser.add_argument('--pre_scaler', type=int, default=1)
parser.add_argument('--model_scaler', type=int, default=1)

parser.add_argument('--gen_max_features', type=int, default=3)
parser.add_argument('--gen_gamma', type=float, default=0.5)
parser.add_argument('--gen_overshoot', type=int, default=0.0001)
parser.add_argument('--gen_max_iter', type=int, default=50)

parser.add_argument('--num_normal_feat', type=int, default=3)
parser.add_argument('--explain_table', type=int, default=1)
parser.add_argument('--explain_text', type=int, default=1)
parser.add_argument('--explain_units', type=str, default="%")
parser.add_argument('--num_feat', type=int, default=16)
parser.add_argument('--action_bound', type=int, default=5)
parser.add_argument('--noise_std', type=int, default=1.8)
parser.add_argument('--rule', type=int, default={1:["U",0],2:["L",0],9:["U",0],10:["L",0],11:["L",0],12:["U",0]}
)
parser.add_argument('--verbose_threshold', type=int, default=50)
parser.add_argument('--model_temp_path', type=str, default="./model_temp.pkl")

###MDQN



parser.add_argument('--seed', type=int, default=66)
args = parser.parse_args()