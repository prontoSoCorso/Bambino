# Import packages
import os
import math
import pandas as pd
import optuna
import numpy as np
import torch
import json
from datetime import datetime
from optuna.trial import FrozenTrial, TrialState
from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution
from optuna.exceptions import TrialPruned

from TrainUtils.NetworkTrainer import NetworkTrainer
from TrainUtils.ToyNetworkTrainer import ToyNetworkTrainer
from TrainUtils.BoaNetworkTrainer import BoaNetworkTrainer
from Types.TaskType import TaskType
from Types.NetType import NetType
from _utils_.OpenFaceDataset import OpenFaceDataset
from _utils_.ToyOpenFaceDataset import ToyOpenFaceDataset
from _utils_.BoaOpenFaceDataset import BoaOpenFaceDataset


# Class
class OptunaParamFinder:

    def __init__(self, model_name, working_dir, task_type, net_type, epochs, batch_size, val_epochs, n_trials,
                 separated_inputs=True, use_cuda=False, is_toy=False, is_boa=False, output_metric="f1",
                 double_output=False, train_data=None, val_data=None, test_data=None, s3=None):
        self.model_name = model_name
        self.working_dir = working_dir
        self.task_type = task_type
        self.net_type = net_type
        self.epochs = epochs
        self.batch_size = batch_size
        self.val_epochs = val_epochs
        self.separated_inputs = separated_inputs
        self.use_cuda = use_cuda
        self.is_toy = is_toy
        self.is_boa = is_boa

        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

        self.output_metric = output_metric
        self.double_output = double_output

        self.s3 = s3

        # Create folder
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
        self.results_dir = working_dir + results_fold + models_fold
        if s3 is None:
            if model_name not in os.listdir(self.results_dir):
                os.mkdir(self.results_dir + model_name)
        else:
            if not s3.exists(self.results_dir + model_name):
                s3.touch(self.results_dir + model_name + "/empty.txt")

        self.counter = 0
        if not self.double_output:
            self.study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(),
                                             pruner=optuna.pruners.MedianPruner())
        else:
            self.study = optuna.create_study(directions=["maximize", "maximize"], sampler=optuna.samplers.TPESampler(),
                                             pruner=optuna.pruners.MedianPruner())
        self.results_dir += model_name + "/"
        self.n_trials = n_trials

        # Retrieve previous results
        file_list = os.listdir(self.results_dir) if s3 is None else [x.split("/")[-1] for x in s3.ls(self.results_dir)]

        if "optuna_study_results.csv" in file_list:
            filepath = self.results_dir + "distributions.json"
            f = open(filepath, "r") if s3 is None else s3.open(filepath, "r")
            distributions = json.load(f)
            distributions = {k: eval(v["name"])(**{key: value for key, value in v.items() if key != "name"})
                             for k, v in distributions.items()}

            # CSV-stored models
            df = pd.read_csv(self.results_dir + "optuna_study_results.csv")
            for _, row in df.iterrows():
                if row["state"] == "RUNNING":
                    continue
                params = row.filter(like="params_").to_dict()
                params = {k.replace("params_", ""): v for k, v in params.items()}
                self.counter += 1
                for k, v in params.items():
                    if k != "optimizer" and k != "use_batch_norm":
                        params.update({k: int(v)})
                self.insert_trial(row, params, distributions)
            print("CSV-stored models inserted!")

            # Untracked models
            models_present = [name.strip(".pt") for name in os.listdir(self.results_dir) if ".pt" in name]
            if len(models_present) > 0:
                max_model_id = np.max([int(name[5:]) for name in models_present])
                n_untracked_models = max_model_id - self.counter
                if n_untracked_models > -1:
                    for i in range(n_untracked_models + 1):
                        print()
                        trainer = NetworkTrainer.load_model(self.working_dir, self.model_name, trial_n=self.counter,
                                                            use_cuda=self.use_cuda, is_toy=self.is_toy, is_boa=self.is_boa,
                                                            s3=s3)
                        train_stats, val_stats = trainer.summarize_performance(show_test=False, show_process=False,
                                                                               show_cm=False, trial_n=self.counter)
                        val_output = getattr(val_stats, output_metric)
                        if double_output:
                            train_output = getattr(train_stats, output_metric)

                        row = {"number": self.counter,
                               "datetime_start": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
                               "datetime_complete": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")}
                        if not self.double_output:
                            row.update({"value": val_output})
                        else:
                            row.update({"values_0": val_output, "values_1": train_output})
                        params = trainer.net_params
                        params.update({"batch_size": int(math.log2(params["batch_size"])),
                                       "lr_last": int(-math.log10(params["lr_last"])),
                                       "n_conv_view_neurons": int(math.log2(params["n_conv_view_neurons"])),
                                       "n_conv_segment_neurons": int(math.log2(params["n_conv_segment_neurons"])),
                                       "p_drop": int(params["p_dropout"] * 10)})
                        del params["p_dropout"]
                        self.insert_trial(row, params, distributions)
                        print("Untracked model", self.counter, "inserted!")
                        self.counter += 1

    def initialize_study(self):
        self.study.optimize(lambda trial: self.objective(trial), self.n_trials)

    def objective(self, trial):
        # Store previous results
        if self.counter >= 10 and self.counter % 10 == 0:
            self.analyze_study(show_best=False)

        # Sample parameters
        params = {
            "n_conv_neurons": int(2 ** (trial.suggest_int("n_conv_neurons", 8, 10, step=1))),
            "n_conv_layers": int(trial.suggest_int("n_conv_layers", 1, 3, step=1)),
            "kernel_size": trial.suggest_int("kernel_size", 3, 5, step=2),
            "hidden_dim": int(2 ** (trial.suggest_int("hidden_dim", 7, 10, step=1))),
            "p_drop": np.round(trial.suggest_float("p_drop", 0, 0.6, step=0.2), 1),
            "n_extra_fc_after_conv": int(trial.suggest_int("n_extra_fc_after_conv", 1, 3, step=1)),
            "n_extra_fc_final": int(trial.suggest_int("n_extra_fc_final", 1, 3, step=1)),
            "optimizer": trial.suggest_categorical("optimizer", ["RMSprop", "Adam"]),
            "lr": np.round(10 ** (-1 * trial.suggest_int("lr", 1, 5, step=1)), 4),
            "batch_size": int(2 ** (trial.suggest_int("batch_size", 5, 6, step=1))),
        }

        # Define seeds
        NetworkTrainer.set_seed(111099)

        print("-------------------------------------------------------------------------------------------------------")
        print("Trial ID:", self.counter)
        print("Parameters:", params)
        self.counter += 1
        try:
            if not self.is_boa:
                if not self.is_toy:
                    trainer = NetworkTrainer(model_name=self.model_name, working_dir=self.working_dir,
                                             task_type=self.task_type, net_type=self.net_type, epochs=self.epochs,
                                             val_epochs=self.val_epochs, params=params,
                                             separated_inputs=self.separated_inputs, use_cuda=self.use_cuda,
                                             train_data=self.train_data, val_data=self.val_data,
                                             test_data=self.test_data, s3=self.s3)
                else:
                    trainer = ToyNetworkTrainer(model_name=self.model_name, working_dir=self.working_dir,
                                                net_type=self.net_type, epochs=self.epochs, val_epochs=self.val_epochs,
                                                params=params, separated_inputs=self.separated_inputs,
                                                use_cuda=self.use_cuda, train_data=self.train_data,
                                                val_data=self.val_data, test_data=self.test_data, s3=self.s3)
            else:
                trainer = BoaNetworkTrainer(model_name=self.model_name, working_dir=self.working_dir,
                                            net_type=self.net_type, epochs=self.epochs, val_epochs=self.val_epochs,
                                            params=params, separated_inputs=self.separated_inputs,
                                            use_cuda=self.use_cuda, train_data=self.train_data, val_data=self.val_data,
                                            test_data=self.test_data, s3=self.s3)
            val_metric = trainer.train(show_epochs=False, trial_n=self.counter-1, trial=trial,
                                       output_metric=self.output_metric, double_output=self.double_output)
            print("Value:", val_metric)
            if self.double_output:
                val_metric, train_metric = val_metric
        except TrialPruned:
            raise
        '''except Exception as e:
            print(f"An error occurred: {e}")
            val_metric = 0
            train_metric = 0'''

        if not self.double_output:
            return val_metric
        else:
            return val_metric, train_metric

    def analyze_study(self, show_best=True):
        # Store study results
        print("-------------------------------------------------------------------------------------------------------")
        print("Storing study...")
        df = self.study.trials_dataframe()
        df.to_csv(self.results_dir + "optuna_study_results.csv", index=False)

        # Store parameters distributions
        distr_file = "distributions.json"
        file_list = os.listdir(self.results_dir) if self.s3 is None else [x.split("/")[-1] for x in self.s3.ls(self.results_dir)]
        if distr_file not in file_list:
            distributions = {k: {"name": type(v).__name__, **v._asdict()} for k, v in
                             self.study.trials[-1].distributions.items()}
            filepath = self.results_dir + distr_file
            f = open(filepath, "w") if self.s3 is None else self.s3.open(filepath, "w")
            json_file = json.dump(distributions, f, indent=4)
            print("Study stored!")

        if not self.double_output:
            if show_best:
                print("Best study:")
                best_trial = self.study.best_trial
                for key, value in best_trial.params.items():
                    print("{}: {}".format(key, value))

            fig = optuna.visualization.plot_intermediate_values(self.study)
            imgpath = self.results_dir + "plot_intermediate_values.jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")

            targets = [None]

        else:
            if show_best:
                print("Best study:")
                best_trials = self.study.best_trials
                for trial in best_trials:
                    print("Trial ID", trial.number - 1, "- outputs:", trial.values)

            fig = optuna.visualization.plot_pareto_front(self.study,
                                                         target_names=["Validation metric", "Training metric"])
            imgpath = self.results_dir + "plot_pareto_front.jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")

            targets = [lambda t: t.values[0], lambda t: t.values[1]]

        for i in range(len(targets)):
            target = targets[i]
            addon = str(i) if target is not None else ""
            target_name = "Objective" + addon + " Value"

            fig = optuna.visualization.plot_optimization_history(self.study, target=target, target_name=target_name)
            imgpath = self.results_dir + "plot_optimization_history" + addon + ".jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")

            fig = optuna.visualization.plot_parallel_coordinate(self.study, target=target, target_name=target_name)
            imgpath = self.results_dir + "plot_parallel_coordinate" + addon + ".jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")

        try:
            fig = optuna.visualization.plot_param_importances(self.study)
            imgpath = self.results_dir + "plot_param_importance.jpg"
            if self.s3 is not None:
                imgpath = self.s3.open(imgpath, "wb")
            fig.write_image(imgpath, format="jpg")
        except RuntimeError as e:
            print("Unable to plot parameter importance!", e)

    def insert_trial(self, row, params, distributions):
        num = int(row["number"])
        if not self.double_output:
            trial = FrozenTrial(
                number=num,
                state=TrialState.COMPLETE,
                value=row["value"],
                datetime_start=datetime.strptime(row["datetime_start"], "%Y-%m-%d %H:%M:%S.%f"),
                datetime_complete=datetime.strptime(row["datetime_complete"], "%Y-%m-%d %H:%M:%S.%f"),
                params=params,
                distributions=distributions,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=num,
            )
        else:
            trial = FrozenTrial(
                number=num,
                state=TrialState.COMPLETE,
                value=None,
                values=[row["values_0"], row["values_1"]],
                datetime_start=datetime.strptime(row["datetime_start"], "%Y-%m-%d %H:%M:%S.%f"),
                datetime_complete=datetime.strptime(row["datetime_complete"], "%Y-%m-%d %H:%M:%S.%f"),
                params=params,
                distributions=distributions,
                user_attrs={},
                system_attrs={},
                intermediate_values={},
                trial_id=num,
            )
        self.study.add_trial(trial)


if __name__ == "__main__":
    # Define variables
    working_dir1 = "./../../"
    model_name1 = "stimulus_conv1d_optuna"
    net_type1 = NetType.CONV1D
    task_type1 = TaskType.TRIAL
    epochs1 = 200
    batch_size1 = None
    val_epochs1 = 10
    separated_inputs1 = True
    is_boa1 = False
    is_toy1 = True

    # Load data
    train_data1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="training_set", is_toy=is_toy1,
                                               is_boa=is_boa1)
    val_data1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="validation_set", is_toy=is_toy1,
                                             is_boa=is_boa1)
    test_data1 = OpenFaceDataset.load_dataset(working_dir=working_dir1, dataset_name="test_set", is_toy=is_toy1,
                                              is_boa=is_boa1)

    # Define Optuna model
    n_trials1 = 10
    output_metric1 = "mcc"
    double_output1 = False
    optuna1 = OptunaParamFinder(model_name=model_name1, working_dir=working_dir1, task_type=task_type1,
                                net_type=net_type1, epochs=epochs1, batch_size=batch_size1, val_epochs=val_epochs1,
                                n_trials=n_trials1, separated_inputs=separated_inputs1, output_metric=output_metric1,
                                double_output=double_output1, is_boa=is_boa1, is_toy=is_toy1, train_data=train_data1,
                                val_data=val_data1, test_data=test_data1)

    # Run search
    optuna1.initialize_study()

    # Evaluate study
    print()
    optuna1.analyze_study()
