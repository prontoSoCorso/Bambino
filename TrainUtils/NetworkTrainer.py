# Import packages
import os
import torch
import torch.nn as nn
import random
import numpy as np
import time
import pickle
import matplotlib.pyplot as plt
import optuna
from torch.utils.data import DataLoader
from torcheval.metrics.functional import multiclass_confusion_matrix
from sklearn.metrics import roc_auc_score
from pandas import DataFrame
from calfram.calibrationframework import select_probability, reliabilityplot, calibrationdiagnosis, classwise_calibration
from collections import Counter

from _utils_.OpenFaceDataset import OpenFaceDataset
from _utils_.ToyOpenFaceDataset import ToyOpenFaceDataset
from _utils_.BoaOpenFaceDataset import BoaOpenFaceDataset
from _utils_.OpenFaceInstance import OpenFaceInstance
from Types.TaskType import TaskType
from Types.SetType import SetType
from TrainUtils.StatsHolder import StatsHolder
from Types.NetType import NetType
from Networks.StimulusConv1d import StimulusConv1d
from Networks.StimulusConv2d import StimulusConv2d


# Class
class NetworkTrainer:
    # Define class attributes
    results_fold = OpenFaceDataset.results_fold
    models_fold = OpenFaceDataset.models_fold
    convergence_patience = 3
    convergence_thresh = 1e-3

    def __init__(self, model_name, working_dir, task_type, net_type, epochs, val_epochs, params=None, use_cuda=True,
                 separated_inputs=True, is_boa=False, is_toy=False, train_data=None, val_data=None, test_data=None,
                 s3=None):
        # Initialize attributes
        self.is_boa = is_boa
        self.is_toy = is_toy
        self.model_name = model_name
        self.working_dir = working_dir
        self.results_dir = working_dir + self.results_fold + self.models_fold
        if s3 is None:
            if model_name not in os.listdir(self.results_dir):
                os.mkdir(self.results_dir + model_name)
        else:
            if not s3.exists(self.results_dir + model_name):
                s3.touch(self.results_dir + model_name + "/empty.txt")
        self.results_dir += model_name + "/"

        self.s3 = s3

        self.task_type = task_type
        self.net_type = net_type
        self.start_time = None
        self.end_time = None

        self.descr_train = None
        self.descr_test = None
        if task_type == TaskType.AGE:
            self.classes = OpenFaceDataset.age_groups
        elif task_type == TaskType.TRIAL:
            self.classes = OpenFaceDataset.trial_id_groups
        else:
            # TaskType.STIM
            self.classes = OpenFaceDataset.trial_types

        if net_type == NetType.CONV1D:
            self.net = StimulusConv1d(params=params, separated_inputs=separated_inputs, n_classes=len(self.classes))
        else:
            # NetType.CONV2D
            self.net = StimulusConv2d(params=params, separated_inputs=separated_inputs, n_classes=len(self.classes))

        # Define training parameters
        self.epochs = epochs
        self.batch_size = self.net.batch_size
        self.val_epochs = val_epochs

        self.criterion = nn.CrossEntropyLoss()
        if params is None:
            self.optimizer = torch.optim.Adam(params=self.net.parameters(), lr=self.net.lr)
        else:
            self.optimizer = getattr(torch.optim, params["optimizer"])(self.net.parameters(), lr=self.net.lr)
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.val_eval_epochs = []
        self.optuna_study = None

        self.use_cuda = torch.cuda.is_available() and use_cuda
        # Otherwise CUDA runs out of memory
        if params is not None and params["n_conv_neurons"] > 1000:
            self.use_cuda = False
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        # Load datasets
        if train_data is None:
            self.train_data = OpenFaceDataset.load_dataset(working_dir=self.working_dir, dataset_name="training_set",
                                                           task_type=self.task_type, is_boa=is_boa, is_toy=is_toy, s3=s3)
        else:
            self.train_data = train_data
        self.train_loader, self.train_dim = self.load_data(self.train_data, shuffle=True)

        if val_data is None:
            self.val_data = OpenFaceDataset.load_dataset(working_dir=self.working_dir, dataset_name="validation_set",
                                                         task_type=self.task_type,
                                                         train_trial_id_stats=self.train_data.trial_id_stats,
                                                         is_boa=is_boa, is_toy=is_toy, s3=s3)
        else:
            self.val_data = val_data
        self.val_loader, self.val_dim = self.load_data(self.val_data)

        if test_data is None:
            self.test_data = OpenFaceDataset.load_dataset(working_dir=self.working_dir, dataset_name="test_set",
                                                          task_type=self.task_type,
                                                          train_trial_id_stats=self.train_data.trial_id_stats,
                                                          is_boa=is_boa, is_toy=is_toy, s3=s3)
        else:
            self.test_data = test_data
        self.test_loader, self.test_dim = self.load_data(self.test_data)

        # Data statistics
        self.train_mean, self.train_std = self.get_normalization_params(self.train_data)

        # Trial ID statistics
        trials = [instance.trial_id for instance in self.train_data.instances]
        m_trial = np.mean(trials)
        s_trial = np.std(trials)
        self.train_data.__dict__["trial_id_stats"] = (m_trial, s_trial)
        self.val_data.__dict__["trial_id_stats"] = (m_trial, s_trial)
        self.test_data.__dict__["trial_id_stats"] = (m_trial, s_trial)

    def load_data(self, data, shuffle=False):
        dataloader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=shuffle, num_workers=2,
                                collate_fn=self.custom_collate_fn)
        dim = len(data)

        return dataloader, dim

    def train(self, show_epochs=False, trial_n=None, trial=None, output_metric="f1", double_output=False):
        if show_epochs:
            self.start_time = time.time()

        self.net.set_cuda(cuda=self.use_cuda)
        if self.use_cuda:
            self.criterion = self.criterion.cuda()
        net = self.net

        if len(self.train_losses) == 0 and trial is None:
            self.summarize_performance(show_test=False, show_process=False, show_cm=False)

        for epoch in range(self.epochs):
            net.set_training(True)
            train_loss = 0
            train_acc = 0
            for x, y, _ in self.train_loader:
                loss, _, acc = self.apply_network(net, x, y)
                train_loss += loss.item()
                train_acc += acc.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss = train_loss / len(self.train_data)
            train_acc = train_acc / len(self.train_loader)
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)

            if epoch % self.val_epochs == 0:
                val_stats = self.test(set_type=SetType.VAL)
                self.val_losses.append(val_stats.loss)
                self.val_accuracies.append(val_stats.acc)
                self.val_eval_epochs.append(epoch)

                # Update and store training curves
                if epoch != 0:
                    self.draw_training_curves()
                    if trial_n is None:
                        filepath = self.results_dir + "training_curves.jpg"
                    else:
                        filepath = self.results_dir + "trial_" + str(trial_n) + "_curves.jpg"
                    if self.s3 is not None:
                        filepath = self.s3.open(filepath, "wb")
                    plt.savefig(filepath)
                    plt.close()

                if trial is not None and not double_output:
                    trial.report(getattr(val_stats, output_metric), epoch)
                    if trial.should_prune():
                        raise optuna.exceptions.TrialPruned()

            if show_epochs and epoch % 10 == 0:
                print("Epoch " + str(epoch + 1) + "/" + str(self.epochs) + " completed... train loss = " +
                      str(np.round(train_loss, 5)))

            if epoch % self.val_epochs == 0 and len(self.val_losses) > NetworkTrainer.convergence_patience:
                # Check for convergence
                val_mean = np.mean(self.val_losses[-NetworkTrainer.convergence_patience:])
                val_std = np.std(self.val_losses[-NetworkTrainer.convergence_patience:])
                val_cv = val_std / val_mean
                if val_cv < NetworkTrainer.convergence_thresh:
                    print("Validation convergence has been reached sooner...")
                    break

        self.net = net
        if show_epochs:
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            print("Execution time:", round(duration / 60, 4), "min")

        self.save_model(trial_n)
        if trial_n is not None:
            train_stats, val_stats = self.summarize_performance(show_test=False, show_process=False, show_cm=False,
                                                                trial_n=trial_n)
            val_output = getattr(val_stats, output_metric)
            if double_output:
                train_output = getattr(train_stats, output_metric)
                return val_output, train_output
            else:
                return val_output

    def apply_network(self, net, x, y):
        x = {key: (x[key] - self.train_mean[key]) / self.train_std[key] for key in x.keys()}
        x = {key: x[key].to(self.device) for key in x.keys()}
        y = y.to(self.device)

        output = net(x)

        # Train loss evaluation
        loss = self.criterion(output, y)

        # Accuracy evaluation
        pred = torch.argmax(output, dim=1)
        acc = torch.sum(pred == y) / y.shape[0]

        return loss, output, acc

    def select_dataset(self, set_type):
        if set_type == SetType.TRAIN:
            data = self.train_data
            data_loader = self.train_loader
            dim = self.train_dim
        elif set_type == SetType.VAL:
            data = self.val_data
            data_loader = self.val_loader
            dim = self.val_dim
        else:
            # SetType.TEST
            data = self.test_data
            data_loader = self.test_loader
            dim = self.test_dim

        return data, data_loader, dim

    def compute_stats(self, y_true, y_pred, loss, acc, desired_class=None, y_prob=None):
        n_vals = len(y_true)
        if loss is None:
            if y_prob is not None:
                loss = self.criterion(y_prob, y_true)
                loss = loss.item()
            else:
                loss = None

        if acc is None:
            acc = torch.sum(y_true == y_pred) / n_vals
            acc = acc.item()

        # Get binary confusion matrix
        cm = NetworkTrainer.compute_binary_confusion_matrix(y_true, y_pred, range(len(self.classes)))
        tp = cm[0]
        tn = cm[1]
        fp = cm[2]
        fn = cm[3]
        auc = NetworkTrainer.compute_binary_auc(y_true, y_pred, range(len(self.classes)))

        stats = StatsHolder(loss, acc, tp, tn, fp, fn, auc, desired_class=desired_class)
        return stats

    def test(self, set_type=SetType.TRAIN, show_cm=False, desired_class=None, assess_calibration=False,
             perform_extra_analysis=False):
        self.net.set_cuda(cuda=self.use_cuda)
        if self.use_cuda:
            self.criterion = self.criterion.cuda()

        net = self.net
        data, data_loader, dim = self.select_dataset(set_type)

        # Store class labels
        y_prob = []
        y_true = []
        y_pred = []
        ages = []
        trial_ids = []
        trial_ids_no_categorical = []
        if self.is_boa:
            sexes = []
            audios = []
            speakers = []
        loss = 0
        net.set_training(False)
        with torch.no_grad():
            for x, y, extra_info in data_loader:
                temp_loss, output, _ = self.apply_network(net, x, y)
                loss += temp_loss.item()

                # Accuracy evaluation
                prediction = torch.argmax(output, dim=1)

                # Store values for Confusion Matrix calculation
                y_prob.append(output)
                y_true.append(y.to(self.device))
                y_pred.append(prediction)

                # Store extra information for detailed analysis
                if perform_extra_analysis:
                    ages.append(extra_info[0])
                    trial_ids.append(extra_info[1])
                    trial_ids_no_categorical.append(extra_info[2])
                    if self.is_boa:
                        sexes.append(extra_info[3])
                        audios.append(extra_info[4])
                        speakers.append(extra_info[5])

            y_prob = torch.concat(y_prob)
            y_true = torch.concat(y_true)
            y_pred = torch.concat(y_pred)
            if perform_extra_analysis:
                ages = torch.concat(ages)
                trial_ids = torch.concat(trial_ids)
                trial_ids_no_categorical = torch.concat(trial_ids_no_categorical)

            loss /= dim
            acc = torch.sum(y_true == y_pred) / dim
            acc = acc.item()
        stats_holder = self.compute_stats(y_true, y_pred, loss, acc, desired_class=desired_class)

        # Compute multiclass confusion matrix
        cm_name = set_type.value + "_cm"
        if show_cm:
            img_path = self.results_dir + cm_name + ".jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
        else:
            img_path = None
        self.__dict__[cm_name] = NetworkTrainer.compute_multiclass_confusion_matrix(y_true, y_pred, self.classes,
                                                                                    img_path)

        if assess_calibration:
            stats_holder.calibration_results = self.assess_calibration(y_true, y_prob, y_pred, set_type)

        if perform_extra_analysis:
            self.perform_extra_analysis(y_true, y_prob, y_pred, ages, set_type, desired_class, "age",
                                        checking_patterns=[(1, 0), (1, 2), (0, 2)])
            print("---------------------------------------------------------------------------------------------------")
            self.perform_extra_analysis(y_true, y_prob, y_pred, trial_ids, set_type, desired_class,
                                        "trial", checking_patterns=[(0, 2), (0, 1), (1, 2)])
            print("---------------------------------------------------------------------------------------------------")
            for age in np.unique(ages):
                ind = ages == age
                y_true_i = y_true[ind]
                y_prob_i = y_prob[ind, :]
                y_pred_i = y_pred[ind]
                trial_ids_i = trial_ids[ind]
                self.perform_extra_analysis(y_true_i, y_prob_i, y_pred_i, trial_ids_i, set_type, desired_class,
                                            "age = " + str(age) + " - trial", checking_patterns=[(0, 2), (0, 1), (1, 2)])
            print("---------------------------------------------------------------------------------------------------")
            for trial in np.unique(trial_ids):
                ind = trial_ids == trial
                y_true_i = y_true[ind]
                y_prob_i = y_prob[ind, :]
                y_pred_i = y_pred[ind]
                ages_i = ages[ind]
                self.perform_extra_analysis(y_true_i, y_prob_i, y_pred_i, ages_i, set_type, desired_class,
                                            "trial = " + str(trial) + " - age", checking_patterns=[(1, 0), (1, 2), (0, 2)])
            print("---------------------------------------------------------------------------------------------------")
            for desired_stat in ["acc", "f1"]:
                self.perform_extra_analysis(y_true, y_prob, y_pred, trial_ids_no_categorical, set_type, desired_class,
                                            "trial no categorical", checking_patterns=None,
                                            desired_stats=[desired_stat])
            print("---------------------------------------------------------------------------------------------------")
            if self.is_boa:
                # ADD HERE EXTRA ANALYSIS FOR BOAprint
                print("---------------------------------------------------------------------------------------------------")

        return stats_holder

    def perform_extra_analysis(self, y_true, y_prob, y_pred, analyzed_var, set_type, desired_class, analysis_criterion,
                               checking_patterns, desired_stats=None):
        analysis_criterion_short = analysis_criterion.replace("=", "")
        analysis_criterion_short = analysis_criterion_short.replace(" ", "")
        path = self.results_dir + analysis_criterion_short + "/"
        if analysis_criterion_short not in os.listdir(self.results_dir):
            os.mkdir(path)

        stat_list = []
        for val in np.unique(analyzed_var):
            # Extract data
            ind = analyzed_var == val
            true = y_true[ind]
            if y_prob is not None:
                prob = y_prob[ind, :]
            else:
                prob = None
            pred = y_pred[ind]

            # Get bootstrap distributions
            boot_stats = self.get_bootstrapped_metrics(true, pred, prob, desired_class=desired_class)
            stat_list.append(boot_stats)

            # Compute statistics
            stats = self.compute_stats(true, pred, None, None, desired_class=desired_class,
                                       y_prob=prob)
            if desired_class is not None and checking_patterns is not None:
                NetworkTrainer.show_performance_table(stats, analysis_criterion + " = " + str(val) + " in " +
                                                      set_type.value, boot_stats=boot_stats)

            # Compute multiclass confusion matrix
            img_path = path + analysis_criterion_short + str(val) + "_" + set_type.value + "_cm.jpg"
            NetworkTrainer.compute_multiclass_confusion_matrix(true, pred, self.classes, img_path)

        # Compare results
        StatsHolder.draw_compare_plots(stat_list=stat_list, extra_feature_name=analysis_criterion, set_type=set_type,
                                       path=path, desired_stats=desired_stats)
        if checking_patterns is not None:
            StatsHolder.statistically_compare_stats(stat_list=stat_list, extra_feature_name=analysis_criterion,
                                                    set_type=set_type, path=path, checking_patterns=checking_patterns)

    def assess_calibration(self, y_true, y_prob, y_pred, set_type):
        y_true = y_true.cpu().numpy()
        y_prob = y_prob.cpu().numpy()
        y_pred = y_pred.cpu().numpy()
        class_scores = select_probability(y_true, y_prob, y_pred)

        # Store results file
        data = np.concatenate((y_true[:, np.newaxis], y_pred[:, np.newaxis], y_prob), axis=1)
        titles = ["y_true", "y_pred"] + ["y_prob" + str(i) for i in range(y_prob.shape[1])]
        df = DataFrame(data, columns=titles)
        filepath = self.results_dir + set_type.value + "_classification_results.csv"
        if self.s3 is not None:
            filepath = self.s3.open(filepath, "wb")
        df.to_csv(filepath, index=False)

        # Draw reliability plot
        reliabilityplot(class_scores, strategy=10, split=False)
        plt.xlabel("Predicted probability")
        plt.ylabel("True probability")
        filepath = self.results_dir + set_type.value + "_calibration.png"
        if self.s3 is not None:
            filepath = self.s3.open(filepath, "wb")
        plt.savefig(filepath)
        plt.close()

        # Compute local metrics
        results, _ = calibrationdiagnosis(class_scores, strategy=10)

        # Compute global metrics
        results_cw = classwise_calibration(results)
        return results_cw

    def save_model(self, trial_n=None):
        if trial_n is None:
            addon = self.model_name
        else:
            addon = "trial_" + str(trial_n)
        filepath = self.results_dir + addon + ".pt"
        file = open(filepath, "wb") if self.s3 is None else self.s3.open(filepath, "wb")
        pickle.dump(self, file)
        print("'" + self.model_name + "' has been successfully saved!... train loss: " +
              str(np.round(self.train_losses[0], 4)) + " -> " + str(np.round(self.train_losses[-1],
                                                                             4)))

    def summarize_performance(self, show_test=False, show_process=False, show_cm=False, desired_class=None,
                              trial_n=None, assess_calibration=False, perform_extra_analysis=False):
        # Show final losses
        train_stats = self.test(set_type=SetType.TRAIN, show_cm=show_cm, desired_class=desired_class,
                                assess_calibration=assess_calibration, perform_extra_analysis=perform_extra_analysis)
        print("Training loss = " + str(np.round(train_stats.loss, 5)) + " - Training accuracy = " +
              str(np.round(train_stats.acc * 100, 7)) + "% - Training F1-score = " +
              str(np.round(train_stats.f1 * 100, 7)))
        if desired_class is not None:
            NetworkTrainer.show_performance_table(train_stats, "training")
        if assess_calibration:
            NetworkTrainer.show_calibration_table(train_stats, "training")

        if perform_extra_analysis:
            print("\n=======================================================================================================\n")
        val_stats = self.test(set_type=SetType.VAL, show_cm=show_cm, desired_class=desired_class,
                              assess_calibration=assess_calibration, perform_extra_analysis=perform_extra_analysis)
        print("Validation loss = " + str(np.round(val_stats.loss, 5)) + " - Validation accuracy = " +
              str(np.round(val_stats.acc * 100, 7)) + "% - Validation F1-score = " +
              str(np.round(val_stats.f1 * 100, 7)))
        if desired_class is not None:
            NetworkTrainer.show_performance_table(val_stats, "validation")
        if assess_calibration:
            NetworkTrainer.show_calibration_table(val_stats, "validation")

        if show_test:
            if perform_extra_analysis:
                print("\n=======================================================================================================\n")
            test_stats = self.test(set_type=SetType.TEST, show_cm=show_cm, desired_class=desired_class,
                                   assess_calibration=assess_calibration, perform_extra_analysis=perform_extra_analysis)
            print("Test loss = " + str(np.round(test_stats.loss, 5)) + " - Test accuracy = " +
                  str(np.round(test_stats.acc * 100, 7)) + "% - Test F1-score = " +
                  str(np.round(test_stats.f1 * 100, 7)))
            if desired_class is not None:
                NetworkTrainer.show_performance_table(test_stats, "test")
            if assess_calibration:
                NetworkTrainer.show_calibration_table(test_stats, "test")

        if show_process or trial_n is not None and len(self.train_losses) != 0:
            self.draw_training_curves()
            if show_process:
                filepath = self.results_dir + "training_curves.jpg"
                if self.s3 is not None:
                    filepath = self.s3.open(filepath, "wb")
                plt.savefig(filepath)
                plt.close()
            if trial_n is not None:
                filepath = self.results_dir + "trial_" + str(trial_n) + "_curves.jpg"
                if self.s3 is not None:
                    filepath = self.s3.open(filepath, "wb")
                plt.savefig(filepath)
                plt.close()

        return train_stats, val_stats

    def draw_training_curves(self):
        plt.close()
        plt.figure(figsize=(5, 7))
        plt.suptitle("Training curves")

        # Losses
        plt.subplot(2, 1, 1)
        plt.plot(self.train_losses, "b", label="Training set")
        plt.plot(self.val_eval_epochs, self.val_losses, "g", label="Validation set")
        plt.legend()
        plt.ylabel("Loss")
        plt.xlabel("Epoch")

        # Accuracies
        plt.subplot(2, 1, 2)
        plt.plot(self.train_accuracies, "b", label="Training set")
        plt.plot(self.val_eval_epochs, self.val_accuracies, "g", label="Validation set")
        plt.legend()
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")

    def show_clinician_stim_performance(self, set_type=SetType.TRAIN, desired_class=None, perform_extra_analysis=False):
        data, _, dim = self.select_dataset(set_type)

        # Get true labels and predicted values
        y_true = []
        y_pred = []
        ages = []
        trial_ids = []
        trial_ids_no_categorical = []
        for instance in data.instances:
            yt = torch.Tensor([instance.trial_type])
            y_true.append(yt)

            yp = torch.Tensor([instance.clinician_pred])
            y_pred.append(yp)

            if perform_extra_analysis:
                age = OpenFaceInstance.categorize_age(instance.age)
                ages.append(torch.Tensor([age]).to(torch.long))
                trial_ids_no_categorical.append(torch.Tensor([instance.trial_id]).to(torch.long))
                trial_id = OpenFaceInstance.categorize_trial_id(instance.trial_id, data.trial_id_stats)
                trial_ids.append(torch.Tensor([trial_id]).to(torch.long))
        y_true = torch.concat(y_true)
        y_pred = torch.concat(y_pred)
        if perform_extra_analysis:
            ages = torch.concat(ages)
            trial_ids = torch.concat(trial_ids)
            trial_ids_no_categorical = torch.concat(trial_ids_no_categorical)

        # Get evaluation metrics
        acc = torch.sum(y_true == y_pred) / dim
        acc = acc.item()
        data.clinician_stats = self.compute_stats(y_true, y_pred, None, acc, desired_class=desired_class)

        # Compute multiclass confusion matrix
        y_true = y_true.to(torch.int64)
        y_pred = y_pred.to(torch.int64)
        img_path = (self.working_dir + OpenFaceDataset.results_fold + OpenFaceDataset.models_fold +
                    "clinician_performance/" + set_type.value
                    + "_cm.jpg")
        data.clinician_cm = NetworkTrainer.compute_multiclass_confusion_matrix(y_true, y_pred,
                                                                               OpenFaceDataset.trial_types,
                                                                               img_path)

        if perform_extra_analysis:
            self.perform_extra_analysis(y_true, None, y_pred, ages, set_type, desired_class,
                                        "age", checking_patterns=[(1, 0), (1, 2), (0, 2)])
            print(
                "---------------------------------------------------------------------------------------------------")
            self.perform_extra_analysis(y_true, None, y_pred, trial_ids, set_type, desired_class,
                                        "trial", checking_patterns=[(0, 2), (0, 1), (1, 2)])
            print(
                "---------------------------------------------------------------------------------------------------")
            for age in np.unique(ages):
                ind = ages == age
                y_true_i = y_true[ind]
                y_pred_i = y_pred[ind]
                trial_ids_i = trial_ids[ind]
                self.perform_extra_analysis(y_true_i, None, y_pred_i, trial_ids_i, set_type, desired_class,
                                            "age = " + str(age) + " - trial",
                                            checking_patterns=[(0, 2), (0, 1), (1, 2)])
            print("---------------------------------------------------------------------------------------------------")
            for trial in np.unique(trial_ids):
                ind = trial_ids == trial
                y_true_i = y_true[ind]
                y_pred_i = y_pred[ind]
                ages_i = ages[ind]
                self.perform_extra_analysis(y_true_i, None, y_pred_i, ages_i, set_type, desired_class,
                                            "trial = " + str(trial) + " - age", checking_patterns=[(1, 0), (1, 2), (0, 2)])
            print("---------------------------------------------------------------------------------------------------")
            for desired_stat in ["acc", "f1"]:
                self.perform_extra_analysis(y_true, None, y_pred, trial_ids_no_categorical, set_type,
                                            desired_class, "trial no categorical", checking_patterns=None,
                                            desired_stats=[desired_stat])
            print("---------------------------------------------------------------------------------------------------")

        # Show performance
        print(set_type.value.upper() + ": Clinician accuracy = " + str(np.round(data.clinician_stats.acc * 100, 7)) +
              "% - Clinician F1-score = " + str(np.round(data.clinician_stats.f1 * 100, 7)) + "%")
        if desired_class is not None:
            NetworkTrainer.show_performance_table(data.clinician_stats, set_type.value)

    def show_model(self):
        print("MODEL:")

        attributes = self.net.__dict__
        for attr in attributes.keys():
            val = attributes[attr]
            if issubclass(type(val), nn.Module):
                print(" > " + attr, "-" * (20 - len(attr)), val)

    def get_bootstrapped_metrics(self, y_true, y_pred, y_prob, desired_class=None, n_rep=100, boot_dim=70):
        boot_stats = []
        n_elem = len(y_true)
        n_samples = int(n_elem * boot_dim / 100)
        for _ in range(n_rep):
            boot_ind = np.random.choice(range(n_elem), size=n_samples, replace=True)
            y_true_boot = y_true[boot_ind]
            y_pred_boot = y_pred[boot_ind]
            if y_prob is not None:
                y_prob_boot = y_prob[boot_ind, :]
            else:
                y_prob_boot = None
            boot_stats.append(self.compute_stats(y_true_boot, y_pred_boot, None, None,
                                                 desired_class=desired_class, y_prob=y_prob_boot))

        return StatsHolder.average_bootstrap_values(stat_list=boot_stats)

    def get_normalization_params(self, data):
        loader = DataLoader(dataset=data, batch_size=data.len, shuffle=False, num_workers=0,
                            collate_fn=self.custom_collate_fn)
        for x, _, _ in loader:
            mean = {key: torch.mean(x[key], dim=(0, 1)) for key in x.keys()}
            std = {key: torch.std(x[key], dim=(0, 1)) for key in x.keys()}
        return mean, std

    @staticmethod
    def compute_binary_confusion_matrix(y_true, y_predicted, classes=None):
        if classes is None:
            # Classical binary computation (class 0 as negative and class 1 as positive)
            tp = torch.sum((y_predicted == 1) & (y_true == 1))
            tn = torch.sum((y_predicted == 0) & (y_true == 0))
            fp = torch.sum((y_predicted == 1) & (y_true == 0))
            fn = torch.sum((y_predicted == 0) & (y_true == 1))

            out = [tp.item(), tn.item(), fp.item(), fn.item()]
            return out
        else:
            # One VS Rest computation for Macro-Averaged F1-score and other metrics
            out = []
            for c in classes:
                y_true_i = (y_true == c).to(int)
                y_predicted_i = (y_predicted == c).to(int)
                out_i = NetworkTrainer.compute_binary_confusion_matrix(y_true_i, y_predicted_i, classes=None)
                out.append(out_i)

            out = np.asarray(out)
            out = [out[:, i] for i in range(out.shape[1])]
            return out

    @staticmethod
    def compute_binary_auc(y_true, y_predicted, classes):
        y_true = y_true.cpu()
        y_predicted = y_predicted.cpu()

        # One VS Rest computation for Macro-Averaged AUC
        out = []
        for c in classes:
            y_true_i = (y_true == c).to(int)
            y_predicted_i = (y_predicted == c).to(int)
            try:
                out_i = roc_auc_score(y_true_i, y_predicted_i)
            except ValueError:
                out_i = 0.5
            out.append(out_i)

        out = np.asarray(out)
        return out

    @staticmethod
    def compute_multiclass_confusion_matrix(y_true, y_pred, classes, img_path=None):
        # Compute confusion matrix
        cm = multiclass_confusion_matrix(y_pred, y_true, len(classes))

        # Draw heatmap
        if img_path is not None:
            NetworkTrainer.draw_multiclass_confusion_matrix(cm, ["no stimulus", "stimulus"], img_path)

        return cm

    @staticmethod
    def draw_multiclass_confusion_matrix(cm, labels, img_path):
        plt.figure(figsize=(2, 2))
        cm = cm.cpu()
        plt.imshow(cm, cmap="Reds")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                val = cm[i, j]
                plt.text(j, i, f"{val.item()}", ha="center", va="center", color="black", fontsize="xx-large")
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.xlabel("Predicted class")
        plt.yticks(range(len(labels)), labels, rotation=45)
        plt.ylabel("True class")
        plt.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close()

    @staticmethod
    def custom_collate_fn(batch):
        batch_inputs = {key: torch.stack([d[key] for d, _, _ in batch]) for key in batch[0][0]}
        batch_labels = torch.stack([label for _, label, _ in batch])
        batch_labels = batch_labels.squeeze(1)
        batch_age = torch.stack([extra[0] for _, _, extra in batch])
        batch_age = batch_age.squeeze(1)
        batch_trial = torch.stack([extra[1] for _, _, extra in batch])
        batch_trial = batch_trial.squeeze(1)
        batch_trial_no_categorical = torch.stack([extra[2] for _, _, extra in batch])
        batch_trial_no_categorical = batch_trial_no_categorical.squeeze(1)
        return batch_inputs, batch_labels, [batch_age, batch_trial, batch_trial_no_categorical]

    @staticmethod
    def load_model(working_dir, model_name, trial_n=None, use_cuda=True, is_toy=False, is_boa=False, s3=None):
        if not is_boa:
            if is_toy:
                results_fold = ToyOpenFaceDataset.results_fold
                models_fold = ToyOpenFaceDataset.models_fold
            else:
                results_fold = OpenFaceDataset.results_fold
                models_fold = OpenFaceDataset.models_fold
        else:
            results_fold = BoaOpenFaceDataset.results_fold
            models_fold = BoaOpenFaceDataset.models_fold
        if trial_n is None:
            file_name = model_name
        else:
            file_name = "trial_" + str(trial_n)
        filepath = (working_dir + results_fold + models_fold + model_name + "/" +
                    file_name + ".pt")
        file = open(filepath, "rb") if s3 is None else s3.open(filepath, "rb")
        network_trainer = pickle.load(file)

        network_trainer.use_cuda = torch.cuda.is_available() and use_cuda
        network_trainer.device = torch.device("cuda" if network_trainer.use_cuda else "cpu")

        # Handle previous versions of the network attributes
        if "separated_inputs" not in network_trainer.net.__dict__.keys():
            network_trainer.net.__dict__["separated_inputs"] = True
            network_trainer.net.__dict__["blocks"] = list(OpenFaceInstance.dim_dict.keys())

        # Handle previous versions of the dataset attributes
        if "train_id_stats" not in network_trainer.train_data.__dict__.keys():
            trials = [instance.trial_id for instance in network_trainer.train_data.instances]
            m_trial = np.mean(trials)
            s_trial = np.std(trials)
            network_trainer.train_data.__dict__["trial_id_stats"] = (m_trial, s_trial)
            network_trainer.val_data.__dict__["trial_id_stats"] = (m_trial, s_trial)
            network_trainer.test_data.__dict__["trial_id_stats"] = (m_trial, s_trial)

        # Handle models created with Optuna
        if trial_n is None and network_trainer.model_name.endswith("_optuna"):
            old_model_name = network_trainer.model_name
            network_trainer.model_name = old_model_name[:-7]
            addon = models_fold
            if addon in network_trainer.results_dir:
                addon = ""
            network_trainer.results_dir = (network_trainer.results_dir[:-(len(old_model_name) + 1)] +
                                           addon + network_trainer.model_name + "/")
        return network_trainer

    @staticmethod
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.deterministic = True
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    @staticmethod
    def show_performance_table(stats, set_name, boot_stats=None):
        print("Performance for", set_name.upper() + " set:")

        for stat in ["acc", "loss"]:
            name = StatsHolder.comparable_stats[stat] if stat in StatsHolder.comparable_stats else stat.upper()
            try:
                if boot_stats is None:
                    addon = ""
                else:
                    addon = " (std: " + str(np.round(boot_stats.__dict__[stat + "_s"] * 100, 2)) + "%)"
                print(" - " + name + ": " + str(np.round(stats.__dict__[stat] * 100, 2)) + "%" + addon)
            except:
                print(" - " + stat + ": missing information")

        for stat in StatsHolder.table_stats:
            if boot_stats is None:
                addon = ""
            else:
                if stat != "mcc":
                    s = str(np.round(boot_stats.__dict__[stat + "_s"] * 100, 2)) + "%)"
                else:
                    s = str(np.round(boot_stats.__dict__[stat + "_s"], 2)) + ")"
                addon = " (std: " + s

            if stat != "mcc":
                s = str(np.round(stats.__dict__["target_" + stat] * 100, 2)) + "%"
            else:
                s = str(np.round(stats.__dict__["target_" + stat], 2))
            print(" - " + StatsHolder.comparable_stats[stat] + ": " + s + addon)

    @staticmethod
    def show_calibration_table(stats, set_name):
        print("Calibration information for", set_name.upper() + " set:")
        for stat in stats.calibration_results.keys():
            print(" - " + stat + ": " + str(stats.calibration_results[stat]))


# Main
if __name__ == "__main__":
    # Define seed
    NetworkTrainer.set_seed(111099)

    # Define variables
    working_dir1 = "./../../"
    model_name1 = "clinician_performance"
    # model_name1 = "stimulus_conv1d"
    net_type1 = NetType.CONV1D
    task_type1 = TaskType.STIM
    epochs1 = 200
    trial_n1 = None
    val_epochs1 = 10
    use_cuda1 = False
    separated_inputs1 = True
    assess_calibration1 = True
    perform_extra_analysis1 = True
    desired_class1 = 1
    show_test1 = True

    # Define trainer
    # params1 = {"n_conv_neurons": 1536, "n_conv_layers": 1, "kernel_size": 7, "hidden_dim": 64, "p_drop": 0.5,
    #            "n_extra_fc_after_conv": 1, "n_extra_fc_final": 1, "optimizer": "RMSprop", "lr": 0.008, "batch_size": 64}  # stimulus_conv1
    params1 = {"n_conv_neurons": 256, "n_conv_layers": 1, "kernel_size": 3, "hidden_dim": 32, "p_drop": 0.2,
               "n_extra_fc_after_conv": 0, "n_extra_fc_final": 1, "optimizer": "RMSprop", "lr": 0.01, "batch_size": 64}  # stimulus_conv2
    trainer1 = NetworkTrainer(model_name=model_name1, working_dir=working_dir1, task_type=task_type1,
                              net_type=net_type1, epochs=epochs1, val_epochs=val_epochs1, params=params1,
                              use_cuda=use_cuda1, separated_inputs=separated_inputs1)

    # Show clinician performance
    trainer1.show_clinician_stim_performance(set_type=SetType.TRAIN, desired_class=desired_class1,
                                             perform_extra_analysis=perform_extra_analysis1)
    print("\n=======================================================================================================\n")
    trainer1.show_clinician_stim_performance(set_type=SetType.VAL, desired_class=desired_class1,
                                             perform_extra_analysis=perform_extra_analysis1)
    print("\n=======================================================================================================\n")
    trainer1.show_clinician_stim_performance(set_type=SetType.TEST, desired_class=desired_class1,
                                             perform_extra_analysis=perform_extra_analysis1)
    # Train model
    print()
    print()
    # trainer1.train(show_epochs=True)
    
    # Evaluate model
    '''trainer1 = NetworkTrainer.load_model(working_dir=working_dir1, model_name=model_name1, trial_n=trial_n1,
                                         use_cuda=use_cuda1)
    trainer1.summarize_performance(show_test=show_test1, show_process=True, desired_class=desired_class1, show_cm=True,
                                   assess_calibration=assess_calibration1,
                                   perform_extra_analysis=perform_extra_analysis1)'''

    # Retrain model
    # trainer1.train(show_epochs=True)
