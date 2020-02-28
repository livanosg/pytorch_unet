import os
import random
from glob import glob
from shutil import copy, move, copytree, rmtree
import cv2

from config import paths, DATASET_DIR

"""
Module to rename, rearrange the files from CHAOS dataset and change MR ground label to binary for easier handling
Change the file tree from
Init---> CT ---> PatientID ---> DICOManon ---> *.dcm
                           ---> Ground    ---> *.png
    ---> MR ---> PatientID ---> T1DUAL ---> DICOManon ---> InPhase  ---> *.dcm
                                                      ---> OutPhase ---> *.dcm
                                       ---> Ground ---> *.png
                           ---> T2SPIR ---> DICOManon ---> *.dcm
                                       ---> Ground ---> *.png
to
Init---> CT ---> PatientID ---> DICOManon ---> *.dcm
                           ---> Ground    ---> *.png
    ---> MR ---> PatientID  ---> DICOManon ---> *.dcm  files renamed to T1_In*, T1_Out*, T2_*
                            ---> Ground ---> *.png
"""


def remove_empty_folders(path, removeroot=True):
    """Function to remove empty folders"""
    if not os.path.isdir(path):
        return
    # remove empty subfolders
    files = os.listdir(path)
    if len(files):
        for f in files:
            fullpath = os.path.join(path, f)
            if os.path.isdir(fullpath):
                remove_empty_folders(fullpath)
    # if folder empty, delete it
    files = os.listdir(path)
    if len(files) == 0 and removeroot:
        print("Removing empty folder:", path)
        os.rmdir(path)


def check_save_png(image, new_path):
    if os.path.exists(new_path):
        print('File exist: {}'.format(new_path))
    else:
        print('Writing: {}'.format(new_path))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        cv2.imwrite(new_path, image)


def check_save_dcm(old_path, new_path):
    if os.path.exists(new_path):
        print('File exist: {}'.format(new_path))
    else:
        print('Writing: {}'.format(new_path))
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        copy(old_path, new_path)


def reconstruct_file_system():
    if not os.path.isdir(paths['chaos']):
        raise ValueError("{} is not a directory".format(paths['chaos']))
    dicoms_list = glob(paths['chaos'] + '/**/*.dcm', recursive=True)
    grounds_list = glob(paths['chaos'] + '/**/*.png', recursive=True)
    for dicom_file in dicoms_list:
        new_dicom_file = dicom_file.replace(paths['chaos'], paths['base'])
        if 'T1DUAL/DICOM_anon/InPhase' in new_dicom_file:
            new_dicom_file = new_dicom_file.replace('T1DUAL/DICOM_anon/InPhase/', 'DICOM_anon/T1_In_')
            check_save_dcm(dicom_file, new_dicom_file)
        elif 'T1DUAL/DICOM_anon/OutPhase' in new_dicom_file:
            new_dicom_file = new_dicom_file.replace('T1DUAL/DICOM_anon/OutPhase/', 'DICOM_anon/T1_Out_')
            check_save_dcm(dicom_file, new_dicom_file)
        elif 'T2SPIR/DICOM_anon' in new_dicom_file:
            new_dicom_file = new_dicom_file.replace('T2SPIR/DICOM_anon/', 'DICOM_anon/T2_')
            check_save_dcm(dicom_file, new_dicom_file)
        else:
            check_save_dcm(dicom_file, new_dicom_file)

    for ground_file in grounds_list:
        ground_image = cv2.imread(ground_file, 0)  # Ensure label are binary
        ground_image[ground_image == 255] = 63
        ground_image[ground_image != 63] = 0
        ground_image[ground_image == 63] = 255
        new_ground_file = ground_file.replace(paths['chaos'], paths['base'])
        if 'T1DUAL/Ground' in new_ground_file:
            for i in ['Ground/T1_In_', 'Ground/T1_Out_']:
                new_ground_file_ = new_ground_file.replace('T1DUAL/Ground/', i)
                check_save_png(image=ground_image, new_path=new_ground_file_)
        elif 'T2SPIR/Ground' in new_ground_file:
            new_ground_file = new_ground_file.replace('T2SPIR/Ground/', 'Ground/T2_')
            check_save_png(image=ground_image, new_path=new_ground_file)
        else:
            check_save_png(image=ground_image, new_path=new_ground_file)
    remove_empty_folders(DATASET_DIR)


def split_evaluation():
    """Split evaluation set from  data dir. Gets also the processed files of the evaluation set."""
    print('Split evaluation set from  data dir.')
    if not os.path.exists(paths['eval']):  # check if Eval_set dont exist, else make folder
        os.mkdir(paths['eval'])
    evaluation_data = glob(paths['eval'] + '/**/*', recursive=False)  # Check if eval data exists
    if evaluation_data:
        print("Directory is not empty")
        print(evaluation_data)
        for eval_patients in evaluation_data:  # Remove existing patient files from Eval_Set
            if 'CT' in eval_patients or 'MR' in eval_patients:
                print('Removing {} from {}'.format(os.path.basename(eval_patients), os.path.dirname(eval_patients)))
                rmtree(eval_patients)
    print('Setting up  evaluation data')
    data = glob(paths['base'] + '/**/')  # Get modal paths from Base folder
    for modalities in data:
        print(modalities)
        if 'CT' in modalities or 'MR' in modalities:  # For each modality gets randomly 2 patients to place in Eval_sets
            patients = sorted(glob(modalities + '/**/'))
            print('{0}\nEvaluation patients\n{0}'.format('*' * 20))
            eval_patients = random.sample(patients, k=2)
            for patient in eval_patients:
                patients.pop(patients.index(patient))  # pop patients from list
                patient_train = patient.replace(paths['base'], paths['train'])
                patient_eval = patient.replace(paths['base'], paths['eval'])
                if os.path.exists(patient_train):  # If patient folder exists in Train_sets folder move it to Eval_Sets
                    print('** Move from :{} **\n** to : {} **'.format(patient_train, patient_eval))
                    move(patient_train, patient_eval)
                else:
                    print('Copy from: {} \n to : {}'.format(patient, patient_eval))  # Copy from Base folder.
                    copytree(patient, patient_eval)
            print('{0}\nTraining patients\n{0}'.format('*' * 20))
            for patient in patients:
                patient_train = patient.replace(paths['base'], paths['train'])
                if os.path.exists(patient_train):
                    print('** Already exist : {} **\nPASS'.format(patient_train))
                else:
                    print('Copy from: {} \n to : {}'.format(patient, patient_train))  # If folder does not exist,
                    copytree(patient, patient.replace(paths['base'], paths['train']))  # copy it from base folder
    remove_empty_folders(DATASET_DIR)


if __name__ == '__main__':
    reconstruct_file_system()
    split_evaluation()
    exit()
