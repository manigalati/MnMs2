import pandas as pd
import os
import numpy as np
import random
import pickle
from scipy.ndimage import binary_fill_holes
import nibabel as nib
from skimage.transform import resize
from batchgenerators.augmentations.utils import resize_segmentation
from batchgenerators.augmentations.spatial_transformations import augment_spatial
import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
from medpy.metric import binary

#################
####CONSTANTS####
#################

BATCH_SIZE = 20

EPOCHS = 250

CKPT = "checkpoints/"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

####################
#Data Visualization#
####################

def get_vendor_info(path):
    vendor_info = pd.read_csv(path)
    for id in range(161,201):
        vendor_info = vendor_info.append({"SUBJECT_CODE": id, "VENDOR": None}, ignore_index=True).astype({"SUBJECT_CODE": int})
    vendor_info = pd.concat([vendor_info, vendor_info]).sort_values(by=["SUBJECT_CODE"]).reset_index(drop=True)
    vendor_info["SUBJECT_CODE"] = ["{:03d}_{}".format(id, "SA" if i%2==0 else "LA") for i,id in enumerate(vendor_info["SUBJECT_CODE"])]
    vendor_info["PATH"] = "data/" + vendor_info["SUBJECT_CODE"] + "_{}.nii.gz"
    return vendor_info

####################
##Data Preparation##
####################

def get_splits(dict_path):
  if not os.path.isfile(dict_path):
      splits = {"train": {"lab": {}, "ulab": {}}, "val": {}, "test": {}}
      ids = range(1, 161)
      ids = random.sample(ids, len(ids))
      splits["train"]["lab"] = ids[:int(len(ids)*0.7)]
      splits["train"]["ulab"] = list(range(161,201))
      splits["val"] = ids[int(len(ids)*0.7) : int(len(ids)*0.85)]
      splits["test"] = ids[int(len(ids)*0.85):]
      with open(dict_path,'wb') as f:
          pickle.dump(splits,f)
  else:
      with open(dict_path,'rb') as f:
          splits = pickle.load(f)
  return splits

def crop_image(image):
    nonzero_mask = binary_fill_holes(image != 0)
    mask_voxel_coords = np.stack(np.where(nonzero_mask))
    minidx = np.min(mask_voxel_coords, axis=1)
    maxidx = np.max(mask_voxel_coords, axis=1) + 1
    resizer = tuple([slice(*i) for i in zip(minidx, maxidx)])
    return resizer
  
def generate_patient_info(vendor_info, dict_path):
  patient_info = {}
  for id in vendor_info["SUBJECT_CODE"]:
      patient_info[id] = vendor_info[vendor_info["SUBJECT_CODE"] == id].to_dict()
      patient_info[id] = {k: list(v.values())[0] for k,v in patient_info[id].items()}
      
      image_path = patient_info[id]["PATH"]
      image = nib.load(image_path.format("CINE"))
      patient_info[id]["spacing"] = image.header["pixdim"][[3,2,1]]
      patient_info[id]["header"] = image.header
      patient_info[id]["affine"] = image.affine

      image_ED = nib.load(image_path.format("ED")).get_fdata()
      image_ES = nib.load(image_path.format("ES")).get_fdata()

      patient_info[id]["shape_ED"] = image_ED.shape
      patient_info[id]["shape_ES"] = image_ES.shape
      patient_info[id]["crop_ED"] = crop_image(image_ED)
      patient_info[id]["crop_ES"] = crop_image(image_ES)
  with open(dict_path, 'wb') as f:
      pickle.dump(patient_info, f)
  return patient_info

def preprocess_image(id, image,crop, is_seg, spacing, spacing_target):
    image = image[crop].transpose(2,1,0)
    spacing_target[0] = spacing[0]
    new_shape = np.round(spacing / spacing_target * image.shape).astype(int)
    if not is_seg:
        image = np.stack([resize(slice, new_shape[1:], 3, cval=0, mode='edge', anti_aliasing=False) for slice in image])
        image -= image.mean()
        image /= image.std()+1e-8
    else:
        image = resize_segmentation(image, new_shape, order=1)
    return image

def preprocess(patient_info, spacing_target, folder_out, soft_preprocessing=False):
    if not os.path.isdir(os.path.join(folder_out, "SA")): os.makedirs(os.path.join(folder_out, "SA"))
    if not os.path.isdir(os.path.join(folder_out, "LA")): os.makedirs(os.path.join(folder_out, "LA"))
    for id in patient_info.keys():
        image_path = patient_info[id]["PATH"]
        images = {"data": {}, "gt": {}}
        for i in ["","_gt"]:
            for j in ["ED","ES"]:
              fname = image_path.format(j+i)
              if(not os.path.isfile(fname)):
                  continue
              if(soft_preprocessing and i=="_gt"):
                  image = nib.load(fname).get_fdata()
              else:
                  image=preprocess_image(
                      id, nib.load(fname).get_fdata(), patient_info[id]["crop_{}".format(j)],
                      i=="_gt", patient_info[id]["spacing"], spacing_target
                  )
              images["data" if i=="" else "gt"][j] = image.astype(np.float32)
        np.save(os.path.join(folder_out, "SA" if "SA" in id else "LA", id), images)

def inSplit(id, split):
    return int(id.split("_")[0]) in split

###################
####Dataloaders####
###################

class RandomCropTransform():
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        sample["data"] = sample["data"][None,None,:,:]
        sample["gt"] = sample["gt"][None,None,:,:]
        sample["data"], sample["gt"] = random_crop(sample["data"], sample["gt"], self.crop_size)
        return {"data": sample["data"][0,0], "gt": sample["gt"][0,0]}

class SpatialTransform():
    def __init__(
        self, patch_size, do_elastic_deform=False, alpha=None, sigma=None,
        do_rotation=True, angle_x=(-np.pi/6,np.pi/6), angle_y=None, angle_z=None,
        do_scale=True, scale=(0.7, 1.4), border_mode_data='constant', border_cval_data=0, order_data=3,
        border_mode_seg='constant', border_cval_seg=0, order_seg=0, random_crop=True, p_el_per_sample=1,
        p_scale_per_sample=1, p_rot_per_sample=1,independent_scale_for_each_axis=False, p_rot_per_axis:float=1,
        p_independent_scale_per_axis: int=1
    ):
        self.params = locals()
        self.params.pop("self")
        self.params["patch_center_dist_from_border"] = list(np.array(patch_size)//2)

    def __call__(self, sample):
        sample["data"] = sample["data"][None,None,:,:]
        sample["gt"] = sample["gt"][None,None,:,:]
        sample["data"], sample["gt"] = augment_spatial(sample["data"], sample["gt"], **self.params) 
        return {"data": sample["data"][0,0],"gt": sample["gt"][0,0]}   

class RndTransform():
    def __init__(self, transform, prob=0.5, alternative_transform=None):
        self.transform = transform
        self.prob = prob
        self.alternative_transform = alternative_transform

    def __call__(self, sample):
        if np.random.uniform() < self.prob:
            return self.transform(sample)
        else:
            if self.alternative_transform is not None:
                return self.alternative_transform(sample)
            else:
                return sample

class MirrorTransform():
    def __call__(self,sample):
        if np.random.uniform() < 0.5:
            sample["data"] = np.copy(sample["data"][::-1])
            sample["gt"] = np.copy(sample["gt"][::-1])
        if np.random.uniform() < 0.5:
            sample["data"] = np.copy(sample["data"][:, ::-1])
            sample["gt"] = np.copy(sample["gt"][:, ::-1])
        return sample

class AddPadding():
    def __init__(self, output_size):
        self.output_size = output_size

    def resize_image_by_padding(self,image,new_shape,pad_value=0):
        shape = tuple(list(image.shape))
        new_shape = tuple(np.max(np.concatenate((shape, new_shape)).reshape((2, len(shape))), axis=0))
        res = np.ones(list(new_shape), dtype=image.dtype) * pad_value
        start = np.array(new_shape) / 2. - np.array(shape) / 2.
        res[int(start[0]) : int(start[0]) + int(shape[0]), int(start[1]) : int(start[1]) + int(shape[1])] = image
        return res
  
    def __call__(self, sample):
        sample={k: self.resize_image_by_padding(v, new_shape=self.output_size) for k,v in sample.items()}
        return sample

class CenterCrop():
    def __init__(self, output_size):
        self.output_size = output_size

    def center_crop_2D_image(self,img,center_crop):
        if(all(np.array(img.shape) <= center_crop)):
            return img
        center = np.array(img.shape) / 2.
        return img[int(center[0] - center_crop[0] / 2.) : int(center[0] + center_crop[0] / 2.), int(center[1] - center_crop[1] / 2.) : int(center[1] + center_crop[1] / 2.)]
    
    def __call__(self, sample):
        sample={k: self.center_crop_2D_image(v, center_crop=self.output_size) for k,v in sample.items()}
        return sample

class RandomCrop():
    def __init__(self, output_size):
        self.output_size = output_size

    def random_crop_2D_image(self,img,crop_size):
        if crop_size[0] < img.shape[0]:
            lb_x = np.random.randint(0, img.shape[0] - crop_size[0])
        elif crop_size[0] == img.shape[0]:
            lb_x = 0
        if crop_size[1] < img.shape[1]:
            lb_y = np.random.randint(0, img.shape[1] - crop_size[1])
        elif crop_size[1] == img.shape[1]:
            lb_y = 0
        return img[lb_x:lb_x + crop_size[0], lb_y:lb_y + crop_size[1]]
  
    def __call__(self, sample):
        sample={k: self.random_crop_2D_image(v, crop_size=self.output_size) for k,v in sample.items()}
        return sample

class OneHot():
    def one_hot(self,seg,num_classes=4):
        return np.eye(num_classes)[seg.astype(int)].transpose(2,0,1)
    def __call__(self, sample):
        if("gt" in sample.keys()):
            if(isinstance(sample["gt"], list)):
                sample['gt'] = [self.one_hot(y) for y in sample['gt']]
            else:
                sample['gt'] = self.one_hot(sample['gt'])
        return sample

class ToTensor():
    def __call__(self, sample):
        sample['data'] = torch.from_numpy(sample['data'][None,:,:]).float()
        if("gt" in sample.keys()):
            if(isinstance(sample["gt"], list)):
                sample['gt'] = [torch.from_numpy(y).float() if len(y.shape)==3 else torch.from_numpy(y).float()[None,:,:] for y in sample['gt']]
            elif(len(sample['gt'].shape) == 2):
                sample['gt'] = torch.from_numpy(sample['gt']).float()[None,:,:]
            else:
                sample['gt'] = torch.from_numpy(sample['gt']).float()
        return sample

class Downsample():
    def downsample_seg_for_ds_transform2(self,seg, ds_scales=[[1,1,1]] + [2*[1/(2**n)] for n in range(1,6)], order=0, cval=0, axes=None):
        if axes is None:
            axes = list(range(0, len(seg.shape)))
        output = []
        for s in ds_scales:
            if all([i == 1 for i in s]):
              output.append(seg)
            else:
              new_shape = np.array(seg.shape).astype(float)
              for i, a in enumerate(axes):
                  new_shape[a] *= s[i]
              new_shape = np.round(new_shape).astype(int)
              output.append(resize_segmentation(seg, new_shape, order))
        return output
    def __call__(self, sample):
        if("gt" in sample.keys()):
            sample['gt'] = self.downsample_seg_for_ds_transform2(sample['gt'])
        return sample

transform = torchvision.transforms.Compose([
    AddPadding((256,256)),
    CenterCrop((256,256)),
    OneHot(),
    ToTensor()
])
transform_augmentation = torchvision.transforms.Compose([
    MirrorTransform(),
    SpatialTransform(patch_size=(256,256), angle_x=(-np.pi/6,np.pi/6), scale=(0.7,1.4), random_crop=True),
    OneHot(),
    ToTensor()
])
transform_downsample = torchvision.transforms.Compose([
    AddPadding((256,256)),
    CenterCrop((256,256)),
    Downsample(),
    OneHot(),
    ToTensor()
])
transform_augmentation_downsample = torchvision.transforms.Compose([
    MirrorTransform(),
    SpatialTransform(patch_size=(256,256), angle_x=(-np.pi/6,np.pi/6), scale=(0.7,1.4), random_crop=True),
    Downsample(),
    OneHot(),
    ToTensor()
])

class ACDCDataLoader():
    def __init__(self, root_dir, batch_size, transform=None, transform_gt=True):
        self.root_dir = root_dir
        self.patient_ids = [file.split(".npy")[0] for file in os.listdir(root_dir)]
        self.batch_size = batch_size
        self.patient_loaders = []
        for id in self.patient_ids:
            self.patient_loaders.append(torch.utils.data.DataLoader(
                ACDCPatient(root_dir, id, transform=transform, transform_gt=transform_gt),
                batch_size=batch_size, shuffle=False, num_workers=0
            ))
        self.counter_id = 0

    def __iter__(self):
        self.counter_iter = 0
        return self

    def set_transform(self, transform):
        for loader in self.patient_loaders:
            loader.dataset.transform = transform
    
    def __next__(self):
        if(self.counter_iter == len(self)):
            raise StopIteration
        loader = self.patient_loaders[self.counter_id]
        self.counter_id += 1
        self.counter_iter += 1
        if(self.counter_id % len(self) == 0):
            self.counter_id = 0
        return loader

    def __len__(self):
        return len(self.patient_ids)

    def current_id(self):
        return self.patient_ids[self.counter_id]

class ACDCPatient(torch.utils.data.Dataset):
    def __init__(self, root_dir, patient_id, transform=None, transform_gt=True):
        self.root_dir = root_dir
        self.id = patient_id
        with open("preprocessed/patient_info.pkl", 'rb') as f:
            self.info = pickle.load(f)[patient_id]
        self.data = np.load(os.path.join(self.root_dir, f"{self.id}.npy"), allow_pickle=True).item()
        self.transform = transform
        self.transform_gt = transform_gt

    def __len__(self):
        return self.info["shape_ED"][2] + self.info["shape_ES"][2]
    
    def __getitem__(self, slice_id):
        is_es = slice_id >= len(self)//2
        slice_id = slice_id - len(self)//2 if is_es else slice_id
        sample = {
            "data": self.data["data"]["ED"][slice_id] if not is_es else self.data["data"]["ES"][slice_id]
        }
        if self.transform_gt:
            if self.data["gt"] != {}:
                sample["gt"] = self.data["gt"]["ED"][slice_id] if not is_es else self.data["gt"]["ES"][slice_id]
            if self.transform:
                sample = self.transform(sample)
        else:
            if self.transform:
                sample = self.transform(sample)
            if self.data["gt"] != {}:
                sample["gt"] = self.data["gt"]["ED"][:,:,slice_id] if not is_es else self.data["gt"]["ES"][:,:,slice_id]
        return sample

class ACDCAllPatients(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None, transform_gt=True):
        self.dataloader = ACDCDataLoader(root_dir, 1, transform=transform, transform_gt=transform_gt)
        self.map = [j for i,patient in enumerate(self.dataloader) for j in [i]*len(patient)]

    def __len__(self):
        return sum(len(patient) for patient in self.dataloader)
    
    def __getitem__(self, id):
        patient_id = self.map[id]
        slice_id = id - self.map.index(patient_id)
        return self.dataloader.patient_loaders[patient_id].dataset.__getitem__(slice_id)

class ACDCPatientPhase(torch.utils.data.Dataset):
    def __init__(self, root_dir, patient_id, phase, transform=None, transform_gt=True):
        self.root_dir = root_dir
        self.id = patient_id
        with open("preprocessed/patient_info.pkl",'rb') as f:
            self.info = pickle.load(f)[patient_id]
        self.data = np.load(os.path.join(self.root_dir, f"{self.id}.npy"), allow_pickle=True).item()
        self.phase = phase    
        self.transform = transform
        self.transform_gt = transform_gt

    def __len__(self):
        return self.info[f"shape_{self.phase}"][2]
  
    def __getitem__(self, slice_id):
        sample = {
            "data": self.data["data"][self.phase][slice_id]
        }
        if self.transform_gt:
            if self.data["gt"] != {}:
                sample["gt"] = self.data["gt"][self.phase][slice_id]
            if self.transform:
                sample = self.transform(sample)
        else:
            if self.transform:
                sample = self.transform(sample)
            if self.data["gt"] != {}:
                sample["gt"] = self.data["gt"][self.phase][:,:,slice_id]
        return sample

#####################
#Supervised Training#
#####################

class Validator:
    def __init__(self, m_best):
        self.m_best = m_best
        self.metrics = {}
        self.pseudo_metrics = {}
        self.best_models = {}
        self.best_predictions = {}
    
    def _get_summary(self, metrics, phases):
        if not isinstance(phases, list):
            phases = [phases]
        summary = {}
        for id in metrics:
            for phase in phases:
                for k,v in metrics[id][phase].items():
                    if k not in summary: summary[k] = []
                    summary[k].append(v[-1])
        return {k: np.mean(v) for k,v in summary.items()}
    
    def get_summary(self, domain_id, phases=["ED","ES"]):
        return self._get_summary(self.metrics[domain_id], phases)
    
    def get_pseudo_summary(self, domain_id, phases=["ED", "ES"]):
        return {"p"+k: v for k,v in self._get_summary(self.pseudo_metrics[domain_id], phases).items()}
    
    def process_labPatient(self, model, loader):
        prediction, gt = [], []
        for batch in loader:
            batch["data"] = batch["data"].to(device)
            batch["output"] = model(batch["data"])
            
            if isinstance(batch["output"], list):
                batch["output"] = batch["output"][0]
            prediction = torch.cat([prediction, batch["output"]], dim=0) if len(prediction)>0 else batch["output"]
            
            if isinstance(batch["gt"], list):
                batch["gt"] = batch["gt"][0]
            batch["gt"] = batch["gt"].to(device)
            gt = torch.cat([gt, batch["gt"]], dim=0) if len(gt)>0 else batch["gt"]
        prediction, gt = ({
            "ED": segmentation.cpu().numpy()[:len(segmentation) // 2],
            "ES": segmentation.cpu().numpy()[len(segmentation) // 2:]
        } for segmentation in (prediction, gt))
        return prediction, gt
    
    def process_ulabPatient(self, model, loader, reconstructor):
        prediction, pgt = [], []
        for batch in loader:
            batch["data"] = batch["data"].to(device)
            batch["output"] = model(batch["data"])

            if isinstance(batch["output"], list):
                batch["output"] = batch["output"][0]
            prediction = torch.cat([prediction, batch["output"]], dim=0) if len(prediction)>0 else batch["output"]

            batch["pgt"] = reconstructor(batch["output"])
            batch["pgt"] = nn.functional.one_hot(
                torch.argmax(batch["pgt"], dim=1), num_classes=4
            ).permute(0,3,1,2).float()
            pgt = torch.cat([pgt, batch["pgt"]], dim=0) if len(pgt)>0 else batch["pgt"]
        prediction, pgt = ({
            "ED": segmentation.cpu().numpy()[:len(segmentation) // 2],
            "ES": segmentation.cpu().numpy()[len(segmentation) // 2:]
        } for segmentation in (prediction, pgt))
        return prediction, pgt
    
    def evaluate_metrics(self, model, prediction, target):
        epoch_metrics = {}
        for k,v in {
            **model.Loss(
                torch.tensor(prediction).to(device),
                torch.tensor(target).to(device),
                validation=True
            ),
            **model.Metrics(
                np.argmax(prediction, axis=1),
                np.argmax(target, axis=1)
            )
        }.items():
            epoch_metrics[k] = v
        return epoch_metrics
    
    def patient_evaluation(self, patient_id, model, prediction, target, metrics):
        for phase in ["ED","ES"]:
            epoch_metrics = self.evaluate_metrics(model, prediction[phase], target[phase])
            if patient_id not in metrics:
                metrics[patient_id] = {
                  "ED": {k: [] for k in epoch_metrics},
                  "ES": {k: [] for k in epoch_metrics}
                }
            for k, v in epoch_metrics.items(): metrics[patient_id][phase][k].append(v)

    def update_best_models(self, domain_id, model, checkpoint):
        if domain_id not in self.best_models: 
            self.best_models[domain_id] = [np.inf]
        value = self.get_summary(domain_id)["Total"]
        if value < np.max(self.best_models[domain_id]):
            self.best_models[domain_id].append(value)
            self.best_models[domain_id] = sorted(self.best_models[domain_id])[:self.m_best]
            index = self.best_models[domain_id].index(value)
            for model_id in range(index + 1, self.m_best)[::-1]:
                checkpoint.rename("best_{:03d}".format(model_id - 1), "best_{:03d}".format(model_id))
            checkpoint("best_{:03d}".format(index), model, self)
    
    def get_distribution(self, domain_id, phase, measure):
        if domain_id in self.metrics:
            metrics = self.metrics
        else:
            metrics = self.pseudo_metrics
        assert domain_id in metrics, f"Domain id {domain_id} does not exist."
        metrics = metrics[domain_id]
        distribution = []
        best = np.max if "dc" in measure else np.min
        for patient_id in metrics:
            distribution.append(best(metrics[patient_id][phase][measure]))
        return distribution

    def get_thrs(self, domain_id, phase, measure):
        distribution = self.get_distribution(domain_id, phase, measure)
        Q1 = np.quantile(distribution, 0.25)
        Q3 = np.quantile(distribution, 0.75)
        IQR = Q3 - Q1
        return Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    
    def update_best_predictions(self, domain_id, patient_id, prediction, pseudo_metrics):
        if domain_id not in self.best_predictions:
            self.best_predictions[domain_id] = {}
        if patient_id not in self.best_predictions[domain_id]:
            self.best_predictions[domain_id][patient_id] = {
                "ED": {"RV_dc": None, "RV_hd": None, "ensemble": None},
                "ES": {"RV_dc": None, "RV_hd": None, "ensemble": None}
            }
        for phase in ["ED", "ES"]:
            for measure in ["RV_dc", "RV_hd"]:
                value = pseudo_metrics[phase][measure][-1]
                best = np.max if "dc" in measure else np.min
                best_value = best(pseudo_metrics[phase][measure])
                if value == best_value:
                    self.best_predictions[domain_id][patient_id][phase][measure] = prediction[phase]
            pseudo_dc = pseudo_metrics[phase]["RV_dc"][-1]
            if pseudo_dc == np.max(pseudo_metrics[phase]["RV_dc"]):
                pseudo_hd = pseudo_metrics[phase]["RV_hd"][-1]
                if pseudo_hd < self.get_thrs(domain_id, phase, "RV_hd")[1]:
                    self.best_predictions[domain_id][patient_id][phase]["ensemble"] = prediction[phase]

    def domain_evaluation(self, domain_id, model, loader, checkpoint=None, reconstructor=None):
        for patient in loader:
            if reconstructor is None:
                if domain_id not in self.metrics: self.metrics[domain_id] = {}
                prediction, gt = self.process_labPatient(model, patient)
                self.patient_evaluation(patient.dataset.id, model, prediction, gt, self.metrics[domain_id])
            else:
                if domain_id not in self.pseudo_metrics: self.pseudo_metrics[domain_id] = {}
                prediction, pgt = self.process_ulabPatient(model, patient, reconstructor)
                self.patient_evaluation(patient.dataset.id, model, prediction, pgt, self.pseudo_metrics[domain_id])
                self.update_best_predictions(domain_id, patient.dataset.id, prediction, self.pseudo_metrics[domain_id][patient.dataset.id])
        if reconstructor is None:
            self.update_best_models(domain_id, model, checkpoint)
    
    def get_history(self, domain_id):
        random_patient = list(self.metrics[domain_id].keys())[0]
        epochs = len(self.metrics[domain_id][random_patient]["ED"]["Total"])
        history = []
        for epoch in range(epochs):
            history.append(np.mean([
                self.metrics[domain_id][patient_id][phase]["Total"][epoch]
                for phase in ["ED", "ES"]
                for patient_id in self.metrics[domain_id]
            ]))
        return history
    
    def get_best_prediction(self, domain_id, patient_id):
        best_prediction = {}
        for phase in ["ED", "ES"]:
            best_prediction[phase] = self.best_predictions[domain_id][patient_id][phase]["ensemble"]
            if best_prediction[phase] is None:
                best_prediction[phase] = np.stack([
                    self.best_predictions[domain_id][patient_id][phase][measure]
                    for measure in ["RV_dc", "RV_hd"]
                ])
                best_prediction[phase] = np.mean(best_prediction[phase], axis=0)
        return best_prediction

    def get_anomalies(self, domain_id):
        anomalies = {"ED": [], "ES": []}
        for phase in ["ED", "ES"]:
            for patient_id in self.pseudo_metrics[domain_id]:
                pseudo_dc = np.max(self.pseudo_metrics[domain_id][patient_id][phase]["RV_dc"])
                if pseudo_dc < self.get_thrs(domain_id, phase, "RV_dc")[0]:
                    anomalies[phase].append(patient_id)
                    continue
                pseudo_hd = np.min(self.pseudo_metrics[domain_id][patient_id][phase]["RV_hd"])
                if pseudo_hd > self.get_thrs(domain_id, phase, "RV_hd")[1]:
                    anomalies[phase].append(patient_id)
        return anomalies

def epoch_end(results, epoch="init"):
    print("\033[1mEpoch [{}]\033[0m".format(epoch))
    header, row = "", ""
    for k,v in results.items():
        header += "{:.6}\t".format(k); row += "{:.6}\t".format("{:.4f}".format(v))
    print(header)
    print(row)

class Checkpointer():
    def __init__(self, ckpt_folder):
        if not os.path.isdir(ckpt_folder):
            os.makedirs(ckpt_folder)
        self.ckpt_folder = ckpt_folder
    
    def rename(self, src, dst):
        if not src.endswith(".pth"): src += ".pth"
        if not dst.endswith(".pth"): dst += ".pth"
        if os.path.isfile(os.path.join(self.ckpt_folder, src)):
            os.rename(os.path.join(self.ckpt_folder, src), os.path.join(self.ckpt_folder, dst))
            os.rename(
                os.path.join(self.ckpt_folder, src.replace(".pth", "_val.pkl")),
                os.path.join(self.ckpt_folder, dst.replace(".pth", "_val.pkl"))
            )
    
    def __call__(self, ckpt_name, model, validator):
        if not ckpt_name.endswith(".pth"): ckpt_name += ".pth"
        torch.save(
            {
                "M": model.state_dict(),
                "M_optim": model.optimizer.state_dict()
            }, 
            os.path.join(self.ckpt_folder, ckpt_name)
        )
        with open(os.path.join(self.ckpt_folder, ckpt_name.replace(".pth", "_val.pkl")), 'wb') as f:
            pickle.dump(validator, f)


def supervised_training(model, epochs, train_loader, val_loader, validator, checkpoint):    
    model.eval()
    with torch.no_grad():
        validator.domain_evaluation("val", model, val_loader, checkpoint)
    epoch_end(validator.get_summary("val"))

    for epoch in epochs:
        model.train()
        for batch in train_loader:
            model.optimizer.zero_grad()
            batch["data"] = batch["data"].to(device)
            batch["gt"] = [y.to(device) for y in batch["gt"]]
            batch["output"] = model.forward(batch["data"])
            loss = sum([
                w * model.Loss(x, y)
                for w,x,y in zip(model.weights, batch["output"], batch["gt"]) if w!=0
            ])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 12)
            model.optimizer.step()
        model.adjust_learning_rate()
        
        model.eval()
        with torch.no_grad():
            validator.domain_evaluation("val", model, val_loader, checkpoint)
        epoch_end(validator.get_summary("val"), epoch)
        if epoch % 10 == 0: checkpoint("{:03d}".format(epoch), model, validator)
    return

def plot_history(history):
    plt.plot(history, '-x', label="loss")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.title('Losses vs. No. of epochs')
    plt.grid()
    plt.show()

###################
#####Baselines#####
###################

def CELoss(prediction, target):
    return -1 * torch.mean(torch.sum(torch.log(prediction + 1e-6) * target, dim=1))

def GDLoss(x, y):
    tp = torch.sum(x * y, dim=(0,2,3))
    fp = torch.sum(x*(1-y),dim=(0,2,3))
    fn = torch.sum((1-x)*y,dim=(0,2,3))
    nominator = 2*tp + 1e-05
    denominator = 2*tp + fp + fn + 1e-05
    dice_score = -(nominator / (denominator+1e-8))[1:].mean()
    return dice_score

def CELoss_RV(prediction, target):
    loss = torch.log(prediction + 1e-6) * target
    loss = torch.sum(torch.stack([
        0*loss[:,0],#BK
        0*loss[:,1],#LV
        0*loss[:,2],#MYO
        1*loss[:,3]#RV
    ]), dim=0)
    return -1*torch.mean(loss)

def GDLoss_RV(x, y):
    tp = torch.sum(x * y, dim=(0,2,3))
    fp = torch.sum(x*(1-y),dim=(0,2,3))
    fn = torch.sum((1-x)*y,dim=(0,2,3))
    nominator = 2*tp + 1e-05
    denominator = 2*tp + fp + fn + 1e-05
    dice_score = -(nominator / (denominator+1e-8))[1:]
    dice_score = torch.sum(torch.stack([
        0/3*dice_score[0],#LV
        0/3*dice_score[1],#MYO
        3/3*dice_score[2]#RV
    ]))
    return dice_score

def DC(prediction, target):
    try: return binary.dc(prediction, target)
    except Exception: return 0

def HD(prediction, target):
    try: return binary.hd(prediction, target)
    except Exception: return np.inf

MSELoss = nn.MSELoss()

class Baseline(nn.Module):
    def __init__(self, args=None):
        super().__init__()
        self.args = args

    def Loss(self, prediction, target, validation=False, onlyRV=False):
        contributes = {f.__name__: f(prediction, target) for f in (self.args.functions if not onlyRV else self.args.functions_RV)}
        contributes["Total"] = sum(contributes.values())
        if validation:
            return {k: v.item() for k,v in contributes.items()}
        return contributes["Total"]
      
    def Metrics(self,prediction,target):    
        metrics={}
        for c,key in enumerate(["LV_", "MYO_", "RV_"], start=1):
            ref = np.copy(target)
            pred = np.copy(prediction)
            ref = np.where(ref != c, 0, 1)
            pred = np.where(pred != c, 0, 1)   
            metrics[key+"dc"] = DC(pred, ref)
            metrics[key+"hd"] = HD(pred, ref)  
        return metrics

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

###############
####Testing####
###############

def get_required_models(solutions, models=[]):
    for v in solutions.values():
        if isinstance(v, dict):
            models = get_required_models(v, models)
        elif isinstance(v, list):
            for v in v:
                if v not in models: 
                    models.append(v)
    return models

def load_models(solutions):
    models = {}
    for axis in ["SA", "LA"]:
        models[axis] = {}
        for model_id in get_required_models(solutions[axis]):
            model = Model()
            ckpt = os.path.join("checkpoints", axis, "best_{:03d}.pth".format(model_id))
            model.load_state_dict(torch.load(ckpt)["M"])
            model.to(device)
            model.eval()
            models[axis][model_id] = model
    return models

def infer_predictions(inference_folder, test_loader, model=None, validator=None):
    if not os.path.isdir(inference_folder):
        os.makedirs(inference_folder)
    for patient in test_loader:
        patient_id = patient.dataset.id
        gt, prediction = [], []
        for iter, batch in enumerate(patient):
            batch = {
                "data": batch["data"].to(device),
                "gt": batch["gt"].to(device)
            }
            gt = torch.cat([gt, batch["gt"]], dim=0) if len(gt)>0 else batch["gt"]
            if model is not None:
                with torch.no_grad():
                    batch["prediction"] = model.forward(batch["data"])[0]
                prediction = torch.cat([prediction, batch["prediction"]], dim=0) if len(prediction)>0 else batch["prediction"]
        gt = {"ED": gt[:len(gt)//2].cpu().numpy(), "ES": gt[len(gt)//2:].cpu().numpy()}
        if len(prediction) != 0:
            prediction = {"ED": prediction[:len(prediction)//2].cpu().numpy(),"ES": prediction[len(prediction)//2:].cpu().numpy()}
        else:
            prediction = validator.get_best_prediction("test", patient_id)
        for phase in ["ED", "ES"]:
            np.save(
                os.path.join(inference_folder, f"{patient_id}_{phase}.npy"),
                {"gt": gt[phase], "prediction": prediction[phase]}
            )
    return

def postprocess_image(image,info,phase,current_spacing):
    postprocessed = np.zeros(info["shape_{}".format(phase)])
    crop = info["crop_{}".format(phase)]
    original_shape = postprocessed[crop].shape
    original_spacing = info["spacing"]
    tmp_shape = tuple(np.round(original_spacing[1:] / current_spacing[1:] * original_shape[:2]).astype(int)[::-1])
    image = np.argmax(image, axis=1)
    image = np.array([torchvision.transforms.Compose([
        AddPadding(tmp_shape), CenterCrop(tmp_shape), OneHot()
    ])({"gt":slice})["gt"] for slice in image])
    image = resize_segmentation(image.transpose(1,3,2,0), image.shape[1:2] + original_shape, order=1)
    image = np.argmax(image, axis=0)
    postprocessed[crop] = image
    return postprocessed

def evaluate_metrics(prediction, reference):
    results = {}
    for c,key in enumerate(["_RV"],start=1):#4->2
        ref = np.copy(reference)
        pred = np.copy(prediction)

        ref = ref if c==0 else np.where(ref != c, 0, ref)
        pred = pred if c==0 else np.where(np.rint(pred) != c, 0, pred)

        try:
            results["DSC" + key] = binary.dc(np.where(ref != 0, 1, 0), np.where(np.rint(pred) != 0, 1, 0))
        except:
            results["DSC" + key] = 0
        try:
            results["HD" + key] = binary.hd(np.where(ref != 0, 1, 0), np.where(np.rint(pred) != 0, 1, 0))
        except:
            results["HD" + key] = np.inf
    return results

def postprocess_predictions(inference_folder, patient_info, current_spacing, output_folder):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    results = {"ED":{}, "ES":{}}
    for prediction_file in os.listdir(inference_folder):
        patient_id = "_".join(prediction_file.split("_")[:-1])
        phase = prediction_file.split("_")[-1].replace(".npy", "")
        prediction_file = np.load(os.path.join(inference_folder, prediction_file), allow_pickle=True).item()
        gt = prediction_file["gt"]
        prediction = prediction_file["prediction"]
        gt = gt.transpose(1,2,0)
        prediction = postprocess_image(prediction, patient_info[patient_id], phase, current_spacing)
        results[phase][patient_id] = evaluate_metrics(prediction, gt)
        nib.save(
            nib.Nifti1Image(prediction, patient_info[patient_id]["affine"], patient_info[patient_id]["header"]),
            os.path.join(output_folder, "{}_{}.nii.gz".format(patient_id, phase))
        )
    return results

def display_results(results):
    table = pd.DataFrame()
    for axis in results:
        df_ED = pd.DataFrame.from_dict(
            results[axis]["ED"],
            orient = 'index',
            columns = ["DSC_LV", "HD_LV", "DSC_RV", "HD_RV", "DSC_MYO", "HD_MYO"]
        )
        df_ES = pd.DataFrame.from_dict(
            results[axis]["ES"],
            orient = 'index',
            columns = ["DSC_LV", "HD_LV", "DSC_RV", "HD_RV", "DSC_MYO", "HD_MYO"]
        )
        stats = {
            "axis": [axis],
            "RV_ED_DC": [df_ED["DSC_RV"].mean()],
            "RV_ED_HD": [df_ED["HD_RV"].mean()],
            "RV_ES_DC": [df_ES["DSC_RV"].mean()],
            "RV_ES_HD": [df_ES["HD_RV"].mean()],
            "RV_DC": [pd.concat([df_ED["DSC_RV"], df_ES["DSC_RV"]]).mean()],
            "RV_HD": [pd.concat([df_ED["HD_RV"], df_ES["HD_RV"]]).mean()],
        }
        table = pd.concat([table, pd.DataFrame.from_dict(stats)]).reset_index(drop=True)
    display(table.set_index("axis"))

############################
#Semi-Supervised Refinement#
############################

def optimize_segmentation(loader, model, ae=None, epoch=None, best_prediction=None):
    for batch in loader:
        model.optimizer.zero_grad()
        batch["data"] = batch["data"].to(device)
        batch["output"] = model.forward(batch["data"])
        if ae is None:
            batch["gt"] = [y.to(device) for y in batch["gt"]]
            loss = sum([
                w * model.Loss(x, y)
                for w,x,y in zip(model.weights, batch["output"], batch["gt"]) if w!=0
            ])
        else:
            batch["output"] = batch["output"][0]
            batch["pseudo_gt"] = ae.forward(batch["output"])
            batch["pseudo_gt"] = torch.mean(torch.stack([
                epoch/EPOCHS * torch.tensor(best_prediction, requires_grad=False).to(device),
                (1 - epoch/EPOCHS) * batch["pseudo_gt"]
            ]),dim=0)
            batch["pseudo_gt"] = torch.nn.functional.one_hot(torch.argmax(batch["pseudo_gt"], dim=1), num_classes=4).permute(0,3,1,2).float()
            loss = model.Loss(batch["output"], batch["pseudo_gt"], onlyRV=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 12)
        model.optimizer.step()
    model.adjust_learning_rate()
    return

def semisupervised_refinement(model, ae, epochs, train_loader, val_loader, ulab_loader, validator, checkpoint):
    model.eval()
    with torch.no_grad():
        validator.domain_evaluation("val", model, val_loader, checkpoint)
    epoch_end(validator.get_summary("val"))

    for epoch in epochs:
        model.train()
        optimize_segmentation(train_loader, model)

        model.eval()
        with torch.no_grad():
            validator.domain_evaluation("val", model, val_loader, checkpoint)
            validator.domain_evaluation("ulab", model, ulab_loader, reconstructor=ae)
        anomalies = validator.get_anomalies("ulab")
        num_anomalies = len(anomalies["ED"]) + len(anomalies["ES"])
        epoch_end({**validator.get_summary("val"), "#anom": num_anomalies}, f"{epoch} - supervised training")
        if num_anomalies == 0: break

        model.train()
        for phase in ["ED", "ES"]:
            for patient_id in anomalies[phase]:
                unlabelled_patient = torch.utils.data.DataLoader(
                    ACDCPatientPhase(ulab_loader.root_dir, patient_id, phase, transform=transform_downsample),
                    batch_size=BATCH_SIZE, shuffle=False, num_workers=0
                )
                optimize_segmentation(unlabelled_patient, model, ae, epoch, best_prediction=validator.get_best_prediction("ulab", patient_id)[phase])

        model.eval()
        with torch.no_grad():
            validator.domain_evaluation("val", model, val_loader, checkpoint)
            validator.domain_evaluation("ulab", model, ulab_loader, reconstructor=ae)
        anomalies = validator.get_anomalies("ulab")
        num_anomalies = len(anomalies["ED"]) + len(anomalies["ES"])
        epoch_end({**validator.get_summary("val"), "#anom": num_anomalies}, f"{epoch} - semisupervised refinement")
        if epoch % 10 == 0: save_checkpoint("{:03d}".format(epoch), model, validator)
        if num_anomalies == 0: break
    return
