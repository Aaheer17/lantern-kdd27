"""
This code is  partially adopted from CaloChallenge github page. Here is the link for it
https://github.com/CaloChallenge/homepage/blob/main/code/

"""




import numpy as np
import h5py
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
import HighLevelFeatures as HLF
#from utils import *
import re
from matplotlib import gridspec
from scipy.stats import wasserstein_distance
from evaluate_plotting_helper import *
import configargparse
import jetnet

def file_read(file_name):
    with h5py.File(file_name, "r") as h5f:
        e = h5f['incident_energies'][::].astype(np.float32)  
        shower = h5f['showers'][::].astype(np.float32)
        #print(e.shape)
        #print(shower.shape)
    return e, shower

def write_dict_to_txt(dictionary, filename):
    # helper function to write a dictionary to a .txt file
    with open(filename, 'w') as file:
        for key, value in dictionary.items():
            file.write(f"{key}: {value}\n")
            
def extract_name_part(file_name):
    # Use regular expression to extract the desired part of the filename
    match = re.search(r'_([^_]+)\.h5(?:df5)?$', file_name)
    if match:
        return match.group(1)
    else:
        match = re.search(r'_([^_]+)\.hdf5$', file_name)
        return match.group(1)
    

def iterate_files(directory):
    Es=[]
    Showers=[]
    model_names=[]
    files=[]
    for filename in os.listdir(directory):
        if filename.endswith(".h5") or filename.endswith(".hdf5"):
            file_path = os.path.join(directory, filename)
            e,shower=file_read(file_path)
            Es.append(e)
            Showers.append(shower)
            name_part = extract_name_part(filename)
            print(f'name part: {name_part}')
            files.append(file_path)
            
            if name_part:
                #print("Extracted part from filename:", name_part)
                model_names.append(name_part)
    return model_names,Es,Showers,files

def save_reference(ref_hlf, fname):
    """ Saves high-level features class to file """
    print("Saving file with high-level features.")
    #filename = os.path.splitext(os.path.basename(ref_name))[0] + '.pkl'
    with open(fname, 'wb') as file:
        pickle.dump(ref_hlf, file)
    print("Saving file with high-level features DONE.")
    

            
def check_file(given_file, dataset, which=None):
    """ checks if the provided file has the expected structure based on the dataset """
    print("Checking if {} file has the correct form ...".format(
        which if which is not None else 'provided'))
    num_features = {'1-photons': 368, '1-pions': 533, '2': 6480, '3': 40500}[dataset]
    num_events = given_file['incident_energies'].shape[0]
    assert given_file['showers'].shape[0] == num_events, \
        ("Number of energies provided does not match number of showers, {} != {}".format(
            num_events, given_file['showers'].shape[0]))
    assert given_file['showers'].shape[1] == num_features, \
        ("Showers have wrong shape, expected {}, got {}".format(
            num_features, given_file['showers'].shape[1]))

    print("Found {} events in the file.".format(num_events))
    print("Checking if {} file has the correct form: DONE \n".format(
        which if which is not None else 'provided'))
def extract_shower_and_energy(given_file, which):
    """ reads .hdf5 file and returns samples and their energy """
    print("Extracting showers from {} file ...".format(which))
    shower = given_file['showers'][:]
    energy = given_file['incident_energies'][:]
    print("Extracting showers from {} file: DONE.\n".format(which))
    return shower, energy
def initialize_HLFs(path,particle,binning_file):
    Es=[]
    Showers=[]
    
    HLFs=[]
    model_names,Es, Showers, files=iterate_files(path)

    for i in range(len(model_names)):
        hlf=HLF.HighLevelFeatures(particle,binning_file)
        hlf.Einc=Es[i]
        hlf.CalculateFeatures(Showers[i])
        HLFs.append(hlf)
    return Es, Showers, HLFs, model_names,files


def prepare_high_data_for_classifier(hdf5_file, hlf_class, label):
    """ takes hdf5_file, extracts high-level features, appends label, returns array """
    voxel, E_inc = extract_shower_and_energy(hdf5_file, label)
    nan_indices = np.where(np.isnan(voxel))
    # print("in voxel: ", len(nan_indices[0]))
    # print("in voxel: ", len(nan_indices[1]))
          
    #print("voxel shape: ",voxel.shape)
    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
  
    E_layer = np.concatenate(E_layer, axis=1)
  
    EC_etas = np.concatenate(EC_etas, axis=1)
   
    EC_phis = np.concatenate(EC_phis, axis=1)
  
    Width_etas = np.concatenate(Width_etas, axis=1)

    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate([np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                          Width_etas/1e2, Width_phis/1e2, label*np.ones_like(E_inc)], axis=1)
    return ret

def parse_arguments():
    parser = configargparse.ArgumentParser(default_config_files=[])
    parser.add_argument('--dataset_path', type=str, required=False, 
                        default='/project/bi_dsc_community/calorimeter/calorimeter_evaluation_data/dataset_2/',
                        help='path to generated h5/hdf5 files are stored')
    parser.add_argument('--output_dir',type=str,
                        default='results/')
    parser.add_argument('--binning_file', type=str, required=False, 
                        default='binning_dataset_2.xml',
                        help='path to binning file')
    parser.add_argument('--particle_type', type=str, required=True,default='electron',
                        help='type of the particle being evaluated e.g., photon, pion, electron')
    parser.add_argument('--dataset_num', type=int, required=True, default=2,
                        help='dataset number e.g., 1, 2, 3')
    parser.add_argument('--title_size', type=int, required=False, default=48,
                        help='size of plot title')
    parser.add_argument('--xlabel_size', type=int, required=False, default=44,
                        help='size of xlabels')
    parser.add_argument('--ylabel_size', type=int, required=False, default=44,
                        help='size of ylabels')
    parser.add_argument('--xtick_size', type=int, required=False, default=30,
                        help='size of xtick')
    parser.add_argument('--ytick_size', type=int, required=False, default=30,
                        help='size of ytikc')
    parser.add_argument('--legend_size',type=int,required=False,default=30,
                       help='legend size')
    parser.add_argument('--row',type=int,default=3,
                       help='row size for subplot, for dataset 2 and 3 it should be 3, for pion it is 2 and photon it is 1')
    parser.add_argument('--col',type=int,default=3,
                       help='column size for subplot, for dataset 2 and 3 it should be 3, for pion and photon it is 2')
    parser.add_argument('--metrics',type=str,required=False,default='all',
                        help='type of metrics to evaluate'+\
                        'all--is used for all high level features\' histograms such as layer dist, shower width, center of energy, sparsity, voxel dist, E_ratio'+\
                        'sep-- is used to generate separation power graph'+\
                        'emd-- is used to generate emd graph'+\
                        'fpd-kpd-- is used to generate fpd,kpd values'
                       )
    parser.add_argument('--sep_file_path',type=str, help='path to file of separation and emd score')

    args = parser.parse_args()
    return args
    
def evaluate_metrics_ds_2_3(Es, Showers, HLFs, model_names, files, args):
    """Plot histograms for dataset 2 and 3 """
    #sparsity
    plot_sparsity_group(HLFs, args.dataset_num,args.output_dir,args.particle_type, model_names, width=4,height=3,
                        TITLE_SIZE=args.title_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                                LEGEND_SIZE=args.legend_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    min_energy=0.5e-3/0.033
    #voxel energy dist
    plot_cell_dist(Showers,min_energy,args.dataset_num,args.output_dir, args.particle_type, model_names, 
                      ratio = False,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    #E_ratio
    plot_Etot_Einc_new(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                      row=1,col=1,height=6,width=8,YMAX=20,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    #center of energy in eta direction
    plot_ECEtas_group(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                      ratio = False,row=3,col=3,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)

    #center of energy in phi direction
    plot_ECPhis_group(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                      ratio = False,row=3,col=3,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    #shower width in eta
    plot_SW_etas_group(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                      ratio = False,row=3,col=3,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    #shower width in phi
    plot_SW_Phis_group(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                      ratio = False,row=3,col=3,height=6,width=8,YMAX=100,LEGEND_SIZE=args.legend_size,
                      XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    
def evaluate_metrics_ds_1(Es, Showers, HLFs, model_names, files, args):
    """ Plot histograms for dataset 1"""
    plot_Etot_Einc_new(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                              row=1,col=1,height=6,width=8,YMAX=25,LEGEND_SIZE=args.legend_size,
                              XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                              TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    min_energy=10

    plot_cell_dist(Showers,min_energy,args.dataset_num,args.output_dir,args.particle_type, model_names, width=7,height=4,
                       TITLE_SIZE=args.title_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
           LEGEND_SIZE=args.legend_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size,YMAX=3,ratio = False)
    plot_ECEtas(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                ratio = False,row=args.row,col=args.col,height=4,width=6,

                YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    plot_ECPhis(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                ratio = False,row=args.row,col=args.col,height=4,width=6,

                YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                      TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)

    plot_SW_Phis(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                    row=args.row,col=args.col,height=4,width=6,

                    YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                          TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)

    if args.particle_type=='photon':
        plot_SW_Etas(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                    ratio = False,row=args.row,col=args.col,height=4,width=6,

                    YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                          TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)
    elif args.particle_type=='pion':
        plot_SW_Etas_pion(HLFs, args.dataset_num,args.output_dir, args.particle_type, model_names, 
                    ratio = False,row=args.row,col=args.col,height=4,width=6,

                    YMAX=100,LEGEND_SIZE=args.legend_size,XLABEL_SIZE=args.xlabel_size,YLABEL_SIZE=args.ylabel_size,
                          TITLE_SIZE=args.title_size,XTICK_SIZE=args.xtick_size,YTICK_SIZE=args.ytick_size)

def evaluate_fpd_kpd(Es,Showers,HLFs,model_names,files,args):
    """Calculates FPD and KPD scores """
    fpd_vals={}
    kpd_vals={}
    fpd_errs={}
    kpd_errs={}
    g_index=model_names.index('Geant4')
            
    reference_HLF=HLFs[g_index]
    reference_file = h5py.File(files[g_index],'r')
    reference_array = prepare_high_data_for_classifier(reference_file, reference_HLF,1)
    reference_array=reference_array[:, :-1]

    for j in range(len(model_names)):
        if j!=g_index:
            source_file=h5py.File(files[j],'r')
            source_array= prepare_high_data_for_classifier(source_file, HLFs[j],0)
            source_array=source_array[:, :-1]
            fpd_val, fpd_err = jetnet.evaluation.fpd(reference_array, source_array)
            kpd_val, kpd_err = jetnet.evaluation.kpd(reference_array, source_array)
            name="dataset_"+str(args.dataset_num)+"_particle_"+args.particle_type+"_model_names"+model_names[j]
            fpd_vals[name]=fpd_val
            kpd_vals[name]=kpd_val
            fpd_errs[name]=fpd_err
            kpd_errs[name]=kpd_err
        print("done with:", model_names[j])

    write_dict_to_txt(fpd_vals,"fpd_val_"+str(args.dataset_num)+"_"+args.particle_type+".txt")
    write_dict_to_txt(kpd_vals,"kpd_val_"+str(args.dataset_num)+"_"+args.particle_type+".txt")

    write_dict_to_txt(fpd_errs,"fpd_errs_"+str(args.dataset_num)+"_"+args.particle_type+".txt")
    write_dict_to_txt(kpd_errs,"kpd_errs_"+str(args.dataset_num)+"_"+args.particle_type+".txt")
    
def main():
    
    ### ......input arguments....
    args = parse_arguments()
    print("printing all arguments.....\n")
    print(args)
    
   
    
    if args.metrics=='all':
         ## this returns incident energy, showers, HLFs object for each sample dataset and their order in the folder
        Es,Showers,HLFs,model_names,files=initialize_HLFs(args.dataset_path,args.particle_type,args.binning_file)
    
        if args.dataset_num==2 or args.dataset_num== 3:
            evaluate_metrics_ds_2_3(Es, Showers, HLFs, model_names, files, args)
            
        elif args.dataset_num==1:
            evaluate_metrics_ds_1(Es, Showers, HLFs, model_names, files, args)
        else:
            print(f"Error in {args.dataset_num}.")
            
    elif args.metrics=='sep':
        taskname='separation_power'
        plot_sep_emd(args.sep_file_path, args.output_dir, args.dataset_num, args.particle_type,width=7,height=5,taskname=taskname)
    elif args.metrics=='emd':
        taskname='emd_score'
        plot_sep_emd(args.sep_file_path, args.output_dir, args.dataset_num, args.particle_type,width=7,height=5,taskname=taskname)
        
    elif args.metrics=='fpd-kpd':
        
        Es,Showers,HLFs,model_names,files=initialize_HLFs(args.dataset_path,args.particle_type,args.binning_file)
        evaluate_fpd_kpd(Es,Showers,HLFs,model_names,files,args)
        
    else:
        print(f"Error! {args.metrics} is not implemented.")
        
        


if __name__ == "__main__":
    main()