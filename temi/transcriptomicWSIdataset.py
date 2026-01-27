import numpy as np
import pandas as pd
import scipy.io
from torch.utils.data import Dataset, DataLoader
import os
import random
import torch
from sklearn import preprocessing


def resampler(wsi_data, repeat):
    sample_ = [random.choices(list(wsi_data)) for _ in range(repeat)]
    sample = np.squeeze(np.array(sample_))
    return sample

class TranscriptomicWSIDataset(Dataset):
    def __init__(self, PROJECT, MODE, CLASSES, MASK_RATIO, TRANSCRIPTOMIC_ROOT, WSI_ROOT):
        

        TRANSCRIPTOMIC_PATH = "%s/transcriptomic/%s/exp_%s_%s.csv" % (TRANSCRIPTOMIC_ROOT, PROJECT, "_".join(PROJECT.lower().split("-")), MODE.lower())
        transcriptomic_data = pd.read_csv(TRANSCRIPTOMIC_PATH, header=0, index_col=0)

        matfiles = []
        for CLASS in CLASSES:
            WSI_PATH = "%s/postdata/boostrapping-2t-100-200/mat/%s/%s/%s/" % (WSI_ROOT,PROJECT, MODE, CLASS)
            matfiles_ = [WSI_PATH+file.name for file in os.scandir(WSI_PATH) if file.name.endswith(".mat")]
            matfiles += matfiles_
        
        self.transcriptomic_data = transcriptomic_data
        self.matfiles = matfiles
        self.CLASSES = CLASSES
        self.MASK_RATIO = MASK_RATIO


    def __len__(self):
        return len(self.matfiles)

    def __getitem__(self, item):
        matflie = self.matfiles[item]
        mat = scipy.io.loadmat(matflie)

        wsi_data = mat["bstdfeat"]

        if wsi_data.shape[0] < 200:
            wsi_data = np.vstack((wsi_data, np.zeros((200 - wsi_data.shape[0], wsi_data.shape[1]), dtype="f")))


        for i in range(len(self.CLASSES)):
            if self.CLASSES[i] in matflie:
                label = i

        split_matfile = matflie.split("/")[-1]
        patientID = split_matfile.split("_")[0]

        patient_bool = [True if id==patientID else False for id in self.transcriptomic_data.columns]
        if sum(patient_bool) != 1:
            print(patientID)
            raise "WSI and transcriptomic patientIDs are mismatched!!!"
        transcriptomic_expr = self.transcriptomic_data.iloc[:, patient_bool].values.flatten()
        maskID = np.random.permutation(len(transcriptomic_expr))[0:int(len(transcriptomic_expr)*self.MASK_RATIO)]
        mask = np.zeros_like(transcriptomic_expr)
        mask[maskID] = 1
        unmask = 1 - mask
        mask_transcriptomic_exper = mask * transcriptomic_expr
        unmask_transcriptomic_exper = unmask * transcriptomic_expr

        
        return wsi_data, unmask_transcriptomic_exper, mask, label, mask_transcriptomic_exper, patientID, split_matfile

# if __name__ == "__main__":
    
#     PROJECT = "GBM-DX"
#     MODE = "TRAIN"

#     CLASSES = ["Proneural", "Mesenchymal"]
#     MASK_RATIO = 0.0

#     dataset = TranscriptomicWSIDataset(PROJECT, MODE, CLASSES, MASK_RATIO)
   

#     train_iter = DataLoader(dataset, batch_size=2048, shuffle=True, num_workers=8, pin_memory=True)
#     for X, UMZ, M, y, MZ, pid_, _ in train_iter:
#         pid__ = preprocessing.LabelEncoder().fit_transform(pid_)
#         pid___ = torch.as_tensor(pid__).float()
#         print()