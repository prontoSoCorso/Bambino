# Import packages
import torch
import numpy as np
import cv2

from TrainUtils.NetworkTrainer import NetworkTrainer
from _utils_.ToyOpenFaceDataset import ToyOpenFaceDataset
from _utils_.OpenFaceInstance import OpenFaceInstance
from Types.TaskType import TaskType
from Types.NetType import NetType
from _utils_.OpenFaceDataset import OpenFaceDataset


# Class
class ToyNetworkTrainer(NetworkTrainer):
    # Define class attributes
    results_fold = ToyOpenFaceDataset.results_fold
    models_fold = ToyOpenFaceDataset.models_fold
    convergence_patience = 5

    def __init__(self, model_name, working_dir, net_type, epochs, val_epochs, params=None, use_cuda=False,
                 separated_inputs=True, train_data=None, val_data=None, test_data=None, s3=None, is_boa=False):
        super().__init__(model_name, working_dir, TaskType.STIM, net_type, epochs, val_epochs, params, use_cuda,
                         separated_inputs, is_boa=is_boa, is_toy=True, train_data=train_data, val_data=val_data,
                         test_data=test_data, s3=s3)

    @staticmethod
    def custom_collate_fn(batch):
        max_len = np.max([item[0]["g"].size(0) for item in batch])
        for item in batch:
            item[0]["g"] = torch.tensor(cv2.resize(item[0]["g"].detach().numpy(), (OpenFaceInstance.dim_dict["g"],
                                                                                   max_len)))
            item[0]["h"] = torch.tensor(cv2.resize(item[0]["h"].detach().numpy(), (OpenFaceInstance.dim_dict["h"],
                                                                                   max_len)))
            item[0]["f"] = torch.tensor(cv2.resize(item[0]["f"].detach().numpy(), (OpenFaceInstance.dim_dict["f"],
                                                                                   max_len)))
        batch_inputs, batch_labels, extra = NetworkTrainer.custom_collate_fn(batch)
        batch_age, batch_trial, batch_trial_no_categorical = extra
        batch_sex = torch.stack([extra[3] for _, _, extra in batch])
        batch_sex = batch_sex.squeeze(1)
        return batch_inputs, batch_labels, [batch_age, batch_trial, batch_trial_no_categorical, batch_sex]


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    working_dir1 = "./../../"
    model_name1 = "stimulus_conv1d"
    net_type1 = NetType.CONV1D
    epochs1 = 2
    trial_n1 = None
    val_epochs1 = 10
    use_cuda1 = False
    separated_inputs1 = True
    assess_calibration1 = True
    perform_extra_analysis1 = False
    desired_class1 = 1
    show_test1 = False

    # Load data
    train_data1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="training_set", is_toy=True)
    val_data1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="validation_set", is_toy=True)
    test_data1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="test_set", is_toy=True)

    # Define trainer
    params1 = {"n_conv_neurons": 256, "n_conv_layers": 1, "kernel_size": 3, "hidden_dim": 32, "p_drop": 0.2,
               "n_extra_fc_after_conv": 0, "n_extra_fc_final": 1, "optimizer": "RMSprop", "lr": 0.01, "batch_size": 64}
    trainer1 = ToyNetworkTrainer(model_name=model_name1, working_dir=working_dir1, net_type=net_type1, epochs=epochs1,
                                 val_epochs=val_epochs1, params=params1, use_cuda=use_cuda1,
                                 separated_inputs=separated_inputs1, train_data=train_data1, val_data=val_data1, 
                                 test_data=test_data1)

    # Train model
    trainer1.train(show_epochs=True)
    
    # Evaluate model
    trainer1 = ToyNetworkTrainer.load_model(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1,
                                            use_cuda=use_cuda1, is_toy=True)
    trainer1.summarize_performance(show_test=show_test1, show_process=True, desired_class=desired_class1, show_cm=True,
                                   assess_calibration=assess_calibration1,
                                   perform_extra_analysis=perform_extra_analysis1)
