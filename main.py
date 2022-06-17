import functions as f
import os
import glob
import re
import numpy as np


if __name__ == '__main__':
    folder = '/home/maryskal/Documents/SITK/Vessel_stencils'
    # Saco todos los files y les elimino la carpeta
    files = [re.split(f'{folder}/', file)[1] for file in glob.glob(f"{folder}/*.nrrd")]
    # Me quedo con los 4 primeras letras, que son el paciente
    patients = list(np.unique([file[0:24] for file in files]))
    # Saco todos los pacientes
    for patient in patients:
        patient_folders = f.extract_patient_data(patient, files)
        ct_path = os.path.join(folder,patient_folders['ct'])
        masks_paths = [os.path.join(folder,mask) for mask in patient_folders['masks']]
        patient_dict = f.paciente_diccionary(patient,ct_path, masks_paths)
        masks = patient_dict['masks']
        f.savePatientMask(patient,masks,[2,3,5],patient_dict['masks']['full'])