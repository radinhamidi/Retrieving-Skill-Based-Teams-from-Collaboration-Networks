import csv
import numpy as np
import eval.ranking as rk
from dal.graph_util import *
import ml_metrics as metrics
import eval.evaluator as dblp_eval

user_HIndex = dblp.get_preprocessed_user_HIndex()
user_skill_dict = dblp.get_user_skill_dict(
    dblp.load_preprocessed_dataset(file_path='../dataset/dblp_preprocessed_dataset.pkl'))
foldIDsampleID_strata_dict = dblp.get_foldIDsampleID_stata_dict(
    data=dblp.load_preprocessed_dataset(file_path='../dataset/dblp_preprocessed_dataset.pkl'),
    train_test_indices=dblp.load_train_test_indices(file_path='../dataset/Train_Test_indices.pkl'),
    kfold=10)
preprocessed_authorNameId_dict = dblp.get_preprocessed_authorNameID_dict()
graph_handler = DBLPGraph()
graph_handler.load_files()
graph_handler.read_graph()

SVAEO = '../output/predictions/S_VAE_O_output.csv'
Sapienza = '../output/predictions/Sapienza_output.csv'
SVDpp = '../output/predictions/SVDpp_output.csv'
RRN = '../output/predictions/RRN_output.csv'
BL2009 = '../output/predictions/BL2009_output.csv'
BL2017 = '../output/predictions/BL2017_output.csv'
ME_M2V_SVAEO = '../output/predictions/ME_M2V_S_VAE_O_output.csv'
MET_M2V_SVAEO = '../output/predictions/MET_M2V_S_VAE_O_output.csv'


file_names = [MET_M2V_SVAEO, ME_M2V_SVAEO, BL2017, BL2009, RRN, SVDpp, Sapienza, SVAEO]

for file_name in file_names:
    method_name, pred_indices, true_indices, _, calc_skill_time, k_fold, k_max = \
        dblp_eval.load_output_file(file_name, foldIDsampleID_strata_dict)
    k_max = 10
    # eval settings
    evaluation_k_set = np.arange(1, k_max + 1, 1)
    fold_set = np.arange(1, k_fold + 1, 1)
    # Initializing metric holders
    Coverage = dblp_eval.init_eval_holder(evaluation_k_set)
    Sensitivity = dblp_eval.init_eval_holder(evaluation_k_set)
    nDCG = dblp_eval.init_eval_holder(evaluation_k_set)
    MAP = dblp_eval.init_eval_holder(evaluation_k_set)
    MRR = dblp_eval.init_eval_holder(evaluation_k_set)
    Quality = dblp_eval.init_eval_holder(evaluation_k_set)
    team_personal = dblp_eval.init_eval_holder(evaluation_k_set)
    # writing output file
    result_output_name = "../output/eval_results/{}.csv".format(method_name)
    with open(result_output_name, 'w') as file:
        writer = csv.writer(file)

        writer.writerow(['@K',
                         'Coverage Mean', 'Coverage STDev',
                         'nDCG Mean', 'nDCG STDev',
                         'MAP Mean', 'MAP STDev',
                         'MRR Mean', 'MRR STDev',
                         'Quality Mean', 'Quality STDev',
                         'Team Personal Mean', 'Team Personal STDev'])

        for i in fold_set:
            truth = true_indices[i]
            pred = pred_indices[i]
            for j in evaluation_k_set:
                print('{}, fold {}, @ {}'.format(method_name, i, j))
                coverage_overall, _ = dblp_eval.r_at_k(pred, truth, k=j)
                Coverage[j].append(coverage_overall)
                nDCG[j].append(rk.ndcg_at(pred, truth, k=j))
                MAP[j].append(metrics.mapk(truth, pred, k=j))
                MRR[j].append(dblp_eval.mean_reciprocal_rank(dblp_eval.cal_relevance_score(pred, truth, k=j)))
                Quality[j].append(dblp_eval.team_formation_feasibility(pred, truth, user_skill_dict, k=j))
                team_personal[j].append(dblp_eval.team_formation_feasibility(pred, truth,
                                                                             user_x_dict=user_HIndex,
                                                                             mode='personal', k=j))

        for j in evaluation_k_set:
            Coverage_mean = np.mean(Coverage[j])
            Coverage_std = np.std(Coverage[j])

            nDCG_mean = np.mean(nDCG[j])
            nDCG_std = np.std(nDCG[j])

            MAP_mean = np.mean(MAP[j])
            MAP_std = np.std(MAP[j])

            MRR_mean = np.mean(MRR[j])
            MRR_std = np.std(MRR[j])

            Quality_mean = np.mean(Quality[j])
            Quality_std = np.std(Quality[j])

            team_personal_mean = np.mean(team_personal[j])
            team_personal_std = np.std(team_personal[j])

            writer.writerow([j,
                             Coverage_mean, Coverage_std,
                             nDCG_mean, nDCG_std,
                             MAP_mean, MAP_std,
                             MRR_mean, MRR_std,
                             Quality_mean, Quality_std,
                             team_personal_mean, team_personal_std
                             ])

        file.close()
