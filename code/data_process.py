import numpy as np
import json
import os, sys

SMISET = {"C": 67, "l": 1, ".": 2, "c": 3, "1": 4, "2": 5, "(": 6,
          "N": 7, "=": 8, "3": 9, ")": 10, "n": 11, "[": 12, "H": 13,
           "]": 14, "O": 15, "@": 16, "s": 17, "+": 18, "/": 19, "S": 20,
            "F": 21, "-": 22, "4": 23, "B": 24, "r": 25, "o": 26, "\\": 27,
             "#": 28, "5": 29, "a": 30, "P": 31, "e": 32, "6": 33, "7": 34,
              "I": 35, "A": 36, "i": 37, "8": 38, "9": 39, "Z": 40, "K": 41,
               "L": 42, "%": 43, "0": 44, "T": 45, "g": 46, "G": 47, "d": 48,
                "M": 49, "b": 50, "u": 51, "t": 52, "R": 53, "p": 54, "m": 55,
                 "W": 56, "Y": 57, "V": 58, "~": 59, "U": 60, "E": 61, "f": 62,
                  "X": 63, "D": 64, "y": 65, "h": 66}

PROTSET = {"A": 1, "R": 2, "N": 3, "D": 4, "C": 5, "Q": 6,
           "E": 7, "G": 8, "H": 9, "I": 10, "L": 11, "K": 12,
           "M": 13, "F": 14, "P": 15, "S": 16, "T": 17, "W": 18,
           "Y": 19, "V": 20}





def one_hot_smiles(line, MAX_SMI_LEN=200):
    X = np.zeros((MAX_SMI_LEN, len(SMISET)))  # +1

    if type(line)!=str:
        print('SMILE format is not str!')
    for i, ch in enumerate(line[:MAX_SMI_LEN]):
        tmp=SMISET.get(ch)
        if tmp:
            X[i, tmp - 1] = 1
        else:
            print(line,'exits not in SMISET character',ch)
    X = X.tolist() 
    return X

def one_hot_protein(line, MAX_SEQ_LEN=1200):
    X = np.zeros((MAX_SEQ_LEN, len(PROTSET)))
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        tmp=PROTSET.get(ch)
        if tmp:
            X[i, tmp - 1] = 1
        else:
            print(line,'exits not in PROTSET character',ch)
    X = X.tolist()
    return X

def label_smiles(line, MAX_SMI_LEN, SMISET):
    X = np.zeros(MAX_SMI_LEN, dtype=int)
    for i, ch in enumerate(line[:MAX_SMI_LEN]):  # x, smiset, y
        X[i] = SMISET[ch]
    X = X.tolist()
    return X

def protein_sequence(line, MAX_SEQ_LEN=550):
    X = np.zeros(MAX_SEQ_LEN)
    for i, ch in enumerate(line[:MAX_SEQ_LEN]):
        X[i] = PROTSET[ch]
    X = X.tolist()
    return X  # .tolist()

def save_txt(dic, fpath,input='c_smiles'):
    dir = os.path.dirname(fpath)
    if input == 'c_smiles':
        path = os.path.join(dir, os.path.basename(fpath).replace('.ism', '.txt'))
    if input == 'c_code':
        path = os.path.join(dir, os.path.basename(fpath).replace('.txt', '') + '-id_coding.txt')
    print(path)
    with open(path, 'w+', encoding='utf-8') as f:
        f.write(json.dumps(dic))

    return path

def main():
    cpd_path = CPD_PATH 
    cpd_label = {}
    ligands = json.load(open(cpd_path))
    for d in ligands.keys():
        cpd_label[d] = label_smiles(ligands[d], 130, SMISET)

    code_path = save_txt(cpd_label, cpd_path, input='c_code')
    print(cpd_label)
    pro_path = PRO_PATH 
    pro_label = {}
    proteins = json.load(open(pro_path))
    for p in proteins.keys():
        pro_label[d] = label_smiles(ligands[d], 130, SMISET)



if __name__ == '__main__':
    main()
