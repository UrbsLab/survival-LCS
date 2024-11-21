import sys
import pandas as pd
from tqdm import tqdm
from importGametes import importGametes
from survival_data_simulator import genSurvSim
from cvPartitioner import cvPartitioner
from sim_utils import make_folder, make_folder_structure, get_parameters

homedir = "/home/bandheyh/common/survival-LCS-telo"
sys.path.append(homedir)

HPC = True
DEBUG = False

outputdir = homedir + "/pipeline"
model_list = ['me', 'epi', 'het', 'add']
nfeat_list = ['f100', 'f1000', 'f10000']
maf_list = ['maf0.2', 'maf0.4']
censor_list = [0.1, 0.4, 0.8]

time_label = "eventTime"
status_label = "eventStatus"
instance_label="inst"
T = 100
knots = 8

iterations = 200000
random_state = 42

cv_count = 5
pmethod = "random"
isContinuous = True
nu = 1
rulepop = 2000

if DEBUG:
    outputdir = homedir + "/test"
    model_list = ['me']
    censor_list = [0.1, 0.4, 0.8]
    nfeat_list = ['f100']
    maf_list = ['maf0.2']
    iterations = 1000
    cv_count = 3

### Create empty brier score DataFrame
brier_df = pd.DataFrame()
cox_brier_df = pd.DataFrame()

make_folder_structure(outputdir, model_list)

pbar = tqdm(total=len(model_list)*len(nfeat_list)*len(maf_list)*len(censor_list))
for i in range(0,len(model_list)):
    for j in range(0,len(nfeat_list)):
        for k in range(0,len(maf_list)):
            g, mtype, d, m, o, e, m0_path, m0_type, m1_path, m1_type = get_parameters(homedir, outputdir, 
                                                                                      model_list, nfeat_list, maf_list, 
                                                                                      i, j, k)
            gametes_data_path = g
            gametes_model_path_0 = m0_path
            gametes_model_path_1 = m1_path
            data_path = d
            model_path = m
            output_path = o
            experiment_name = e
            model0_type = m0_type
            model1_type = m1_type
            model_type = mtype

            make_folder(data_path)
            make_folder(output_path)
            # make_folder(model_path)

            import_Gametes = importGametes(gametes_data_path,gametes_model_path_0, model0_type,model1_type,model_type,gametes_model_path_1)
            P, P0, P1 = import_Gametes.parseModelFile(gametes_model_path_0,gametes_model_path_1,model0_type,model1_type,model_type)
            test_genSurvSim = genSurvSim(import_Gametes, T, model_type)

            for l in range(0, len(censor_list)):
                survData,survDists = test_genSurvSim.generate_time(test_genSurvSim.X, import_Gametes.names, P, P0, P1, model_type, censor_list[l]) #names is from importGametes, will this import properly?

                # weird error in heterogeneous datasets, added extra column "Unnamed...", this removes that
                survData = survData.loc[:, ~survData.columns.str.startswith('Unnamed')]
                # need to save data to one of the output locations
                survData.to_csv(output_path+'/' + str(model_type) + '_cens'+ str(censor_list[l]) + '_surv', index = False, sep = '\t')
                if gametes_model_path_0 is not None:
                    cv_part = cvPartitioner(output_path+'/' + str(model_type) +'_cens'+str(censor_list[l])+'_surv',data_path, cv_count,pmethod,time_label)
                else:
                    cv_part = cvPartitioner(output_path+'/' + str(model_type)+'_surv',data_path, cv_count,pmethod,time_label)
                pbar.update(1)
