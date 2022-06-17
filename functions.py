import logging
import sys
import os
import re
from multiprocessing import Process
from varname import nameof
import SimpleITK as sitk
import numpy as np


# Configura el logging
log_format = '[%(process)d]\t%(asctime)s %(levelname)s: %(message)s'
logging.basicConfig(format=log_format, level=logging.INFO, datefmt="%H:%M:%S",
                    handlers=[logging.StreamHandler(sys.stdout)])


def extract_patient_data(patient_name, all_files):
    '''
    De una lista de archivos extrae todos los que compartan
    el nombre introducido (patient_name)
        :param patient_name (str): nombre comun
        :param all_files (list): lista de files (str)
        :return (dict): diccionario {'ct': ctPath, 'masks': [maskPaths]}
    '''
    patient_files = [file for file in all_files if bool(re.search(patient_name, file))]
    patient_ct = f'{patient_name}.nrrd'
    patient_masks = [file for file in patient_files if file != patient_ct]
    return {'ct': patient_ct, 'masks': patient_masks}

def separar(mask):
    '''
    Coge una mascara en la que estan arteria
    y vena y la separa en dos.
        :param mask (SITK image): mask
        :return (list): con las dos mascaras
    '''
    # vemos los valores diferentes que existen
    arr = sitk.GetArrayFromImage(mask)
    values = np.unique(arr)
    values = values[values != 0]
    # sacamos una mascara por cada valor
    masks = []
    for value in values:
        masks.append(mask == value)
    return masks


def paciente_diccionary(patient, ctPath, maskPaths):
    '''
    Con el path del ct y la lista de paths del
    paciente crea un diccionario con su ctPath
    su mascara, su mascara venosa y su mascara
    arterial.
        :param patient (str): nombre del paciente
        :param ctPath (str):
        :param maskPaths (list): lista de paths de mascaras
        :return (dict): diccionario del paciente:
                        {'patient': nombre del paciente,
                        'path': el path de su ct,
                        'ct': sitk image de su ct,
                        'masks': {maskPaths[i]: sitk image,
                                    masksPath[i]_vena: sitk image,
                                    masksPath[i]_arteria: sitk image}
                        'n_ves': numero de mascaras introducidas}
    Se asume que el numero mas bajo dentro de la mascara es vena y el mas
    alto es arteria, es decir, si tenemos valores de 0,30 y 40, 30 sería vena
    40 sería arteria.
    '''
    art_vein = {0: 'vein',
                1: 'artery'}

    pacienteDict = {}
    pacienteDict['patient'] = patient
    pacienteDict['path'] = ctPath
    pacienteDict['ct'] = sitk.ReadImage(ctPath)
    pacienteDict['masks'] = {}
    for path in maskPaths:
        vesselMask = sitk.ReadImage(path)
        pacienteDict['masks'][path[-9:-5]] = vesselMask
        sep = separar(vesselMask)
        for j, vesel in enumerate(sep):
            pacienteDict['masks'][path[-9:-5] + '_' + art_vein[j]] = vesel
    pacienteDict['n_ves'] = len(maskPaths)
    return pacienteDict


def eliminarVaso(distance, remove):
    '''
    Elimina una mascara sobre otra
        :param distance (sitk Image): mascara original
        :param remove (sitk Image): mascara de eliminacion
        :return (sitk image) sin vasos
    '''
    # A las distancias le restamos la otra mascara binarizada
    new = distance - sitk.BinaryThreshold(remove, 1, 100000, 1)
    # Binarizamos el resultado para que no queden valores negativos
    new = sitk.BinaryThreshold(new, 1, 1, 1)
    logging.info('[F]\teliminarVaso() executed')
    return new


def saveMask(image, path, fileName):
    '''
    Guarda una imagen SITK
        :param image (sitk Image): imagen a guardar
        :param path (string): donde guardar
        :param fileName (string): nombre
    '''
    if not os.path.exists(path):
        os.makedirs(path)
    sitk.WriteImage(image, os.path.join(path, fileName))
    logging.info('[F]\tsaveMask executed in {}, as {}'.format(path, fileName))


def createDistanceMask(vessels, distance, remove, save=False,
                       filename='mask.nrrd', folder='Vessel_stencils/masks'):
    '''
    Crea una mascara de espacio perivascular dada una distancia.
    Además la guarda con el nombre y en la carpeta que se diga.
        :param vesels (sitk Image): mascara de vasos
        :param distance (double): distancia en mm para el espacio peri
        :param save (bool): si quieres guardar la mascara
        :param filename (str): con que nombre guardarlo
        :param folder (string): donde guardarlo
        :return (sitk image) con el espacio perivascular
    '''
    # Se crea el mapa de distancias
    allDistances = sitk.SignedMaurerDistanceMap(vessels,
                                                insideIsPositive=False,
                                                squaredDistance=False,
                                                useImageSpacing=True)
    # Se crea la máscara con el espacio seleccionado
    distanceMask = allDistances > 0 and allDistances < distance
    # Se elimina el vaso de la máscara
    distanceMask = eliminarVaso(distanceMask, remove)
    if save:
        # Se guarda la mascara
        saveMask(distanceMask, folder, filename)
    logging.info('[F]\tcreateDistanceMaks executed, saved {}'.format(save))
    return distanceMask


def savePatientMaskP(patient, masks, distances, path = ''):
    '''
    Si se introduce un paciente, con sus mascaras
    crea una mascara de espacio perivsacular para cada
    mascara vascular y para cada distancia. Lo guarda
    en una carpeta con el nombre del paciente.
        :param patient (str): nombre del paciente
        :param mask (dict): diccionario {'nombre_mascara': sitk imagen}
        :param distance (list of double): distancia de espacios
        :param path (str): carpeta donde queremos guardar
    '''
    # Creo la carpeta de este paciente
    patientFolder = os.path.join(path, patient)
    # Creo un vector de threads
    proceses = []
    # Inicializo el numero de threads
    pNumber = 0

    # Se recorre cada una de las mascaras vasculares
    # y se calcula cada una de las distancias
    for k, mask in masks.items():
        for distance in distances:
            # Creo el nombre del file
            fileName = 'mask_' + k + '_' + str(distance) + '.nrrd'

            # Actualizo el numero de threads
            pNumber += 1

            # Inicio un thread con la funcion createDistanceMask
            process = Process(target=createDistanceMask,
                              args=(mask, distance, True, fileName, patientFolder))
            # Lo añado a la lista
            proceses.append(process)
            # Lo inicializo
            process.start()
            # Imprimo el aviso
            logging.info("[T]\tThread {} started!".format(pNumber))

    # Espera a que los threads finalicen
    for pNumber, process in enumerate(proceses):
        process.join()
        logging.info("[T]\tThread {} joined!".format(pNumber))

    logging.info("[T]\tDONE!")


def savePatientMask(patient, masks, distances, remove):
    '''
    If you introduce vascular masks and distances of a patient
    it create a new mask with each vascular mask and each distance
    and it save them in a folder with the name of the patient.

    Parameters
    ----------
    patient (string): name of the patient
    mask (list of sitk image): vascular mask
    distance (list of double): perivascular distance
    '''
    # Creo la carpeta de este paciente
    patientFolder = os.path.join('Vessel_stencils', patient)

    # Se recorre cada una de las mascaras vasculares
    # y se calcula cada una de las distancias
    for k, mask in masks.items():
        for distance in distances:
            # Creo el nombre del file
            fileName = 'mask_' + k + '_' + str(distance) + '.nrrd'
            # Creo la máscara
            createDistanceMask(mask, distance, remove, True, fileName, patientFolder)
            # Mando el aviso
            logging.info("[F]\tmask {} created".format(fileName))

    logging.info("[F]\tDONE!")