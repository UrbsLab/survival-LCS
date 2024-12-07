import os
import shutil
import dask
from dask.distributed import Client
from dask_jobqueue import SLURMCluster, LSFCluster, SGECluster

def make_folder(path, overwrite=False):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        if overwrite:
            shutil.rmtree(path)
            os.makedirs(path)

def make_folder_structure(outputdir, models, overwrite=True):
    if overwrite==True:
        make_folder(outputdir+'/cv_sim_data/')
        make_folder(outputdir+'/pickled_cv_models/')
        make_folder(outputdir+'/sim_lcs_output/')
        for model in models:
            make_folder(outputdir+'/cv_sim_data/cv_' + model, overwrite=overwrite)
            make_folder(outputdir+'/pickled_cv_models/' + model, overwrite=overwrite)
            make_folder(outputdir+'/sim_lcs_output/' + model, overwrite=overwrite)
    else:
        raise NotImplementedError


def get_cluster(cluster_type='SLURM', output_path=".", queue='defq', memory=16):
    client = None
    try:
        if cluster_type == 'SLURM':
            cluster = SLURMCluster(queue=queue,
                                   cores=1,
                                   memory=str(memory) + "G",
                                   walltime="24:00:00",
                                   log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=100)
        elif cluster_type == "LSF":
            cluster = LSFCluster(queue=queue,
                                 cores=1,
                                 mem=memory * 1000000000,
                                 memory=str(memory) + "G",
                                 walltime="24:00",
                                 log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=100)
        elif cluster_type == 'UGE':
            cluster = SGECluster(queue=queue,
                                 cores=1,
                                 memory=str(memory) + "G",
                                 resource_spec="mem_free=" + str(memory) + "G",
                                 walltime="24:00:00",
                                 log_directory=output_path + "/dask_logs/")
            cluster.adapt(maximum_jobs=100)
        elif cluster_type == 'Local':
            c = Client()
            cluster = c.cluster
        else:
            raise Exception("Unknown or Unsupported Cluster Type")
        client = Client(cluster)
    except Exception as e:
        print(e)
        raise Exception("Exception: Unknown Exception")
    print("Running dask-cluster")
    print(client.scheduler_info())
    return client

def get_parameters(homedir, outputdir, models, nfeat, maf, i, j, k):

    g = homedir + '/' + 'simulated_datasets/' + \
        'EDM-1/'+str(models[i]) + \
        '_' + str(nfeat[j]) + '_' + str(maf[k]) + '_' + 'EDM-1_01.txt'
    dtype = str(models[i]) + '_' + str(nfeat[j]) + '_' + str(maf[k])
    # print(g)

    d = outputdir + '/' + 'cv_sim_data/cv_' + str(models[i]) + '/' + dtype
    m = outputdir + '/' + 'pickled_cv_models/' + str(models[i]) + '/' + dtype
    o = outputdir + '/' + 'sim_lcs_output/' + str(models[i]) + '/' + dtype

    ### Set m0_path
    if models[i] in ['me','add','het']:
        m0_path = homedir+'/'+'simulated_datasets/'+'EDM-1/model_files/me_h0.2_'+str(maf[k])+'_Models.txt'
    else:
        m0_path = homedir+'/'+'simulated_datasets/'+'EDM-1/model_files/epi_h0.2_'+str(maf[k])+'_Models.txt'

    ### Set m1_path
    if models[i] in ['me','epi']:
        m1_path = None
    else:
        m1_path = homedir+'/'+'simulated_datasets/'+'EDM-1/model_files/epi_h0.2_'+str(maf[k])+'_Models.txt'

    ### Set m0_type
    if models[i] in ['me','add','het']:
        m0_type = 'main_effect'
    else:
        m0_type = '2way_epistasis'

    ### Set m1_type
    if models[i] in ['me', 'epi']:
        m1_type = None
    else:
        m1_type = '2way_epistasis'

    ### Set mtype
    if models[i] == 'me':
        mtype = 'main_effect'
    elif models[i] == 'epi':
        mtype = '2way_epistasis'
    elif models[i] == 'add':
        mtype = 'additive'
    else:
        mtype = 'heterogeneous'


    e = "testallsims"
    # print(str(models[i])+'_'+str(nfeat[j])+'_'+str(maf[k]))

    # self.gametes_data_path = g
    # self.gametes_model_path_0 = m0_path
    # self.gametes_model_path_1 = m1_path
    # self.data_path = d
    # self.model_path = m
    # self.output_path = o
    # self.experiment_name = e
    # self.model0_type = m0_type
    # self.model1_type = m1_type
    # self.model_type = mtype #add parameter with name of original dataset

    return g, mtype, d, m, o, e, m0_path, m0_type, m1_path, m1_type


def run_parellel(model):
    try:
        return model.run()
    except Exception as e:
        return e
