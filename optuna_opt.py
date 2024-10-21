"""
Optuna example that optimizes multi-layer perceptrons using PyTorch.

In this example, we optimize the validation accuracy of fashion product recognition using
PyTorch and FashionMNIST. We optimize the neural network architecture as well as the optimizer
configuration. As it is too time consuming to use the whole FashionMNIST dataset,
we here use a small subset of it.

"""

import os

import optuna
from optuna.trial import TrialState
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
import ray
from torchvision import datasets
from torchvision import transforms
from tqdm import tqdm
import time
import logging
import joblib
import pickle
import wandb
import matplotlib.pyplot as plt
from optuna.integration.wandb import WeightsAndBiasesCallback

# wandb might cause an error without this.
os.environ["WANDB_START_METHOD"] = "thread"

from Fashion_NN import Net


DEVICE = torch.device("cuda")
BATCHSIZE = 128
DIR = os.getcwd()
EPOCHS = 10
N_TRAIN_EXAMPLES = BATCHSIZE * 30
N_VALID_EXAMPLES = BATCHSIZE * 10

CPU_per_Actor = 8
GPU_per_Actor = 1

@ray.remote(num_gpus=GPU_per_Actor, num_cpus=CPU_per_Actor)
class Trainable_NN:

    def __init__(self, train_dataloader, val_dataloader, device):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

    def set_NN(self, parameters):
        self.net = Net(n_layers=parameters["n_layers"],
                    out_features=parameters["n_hidden"],
                    dropout_p=parameters["dropout_p"])

        self.net.to(self.device)

    def set_optimizer(self, parameters):
        # Generate the optimizers.
        optimizer_name = parameters["optimizer"]
        lr = parameters["lr"]
        self.optimizer = getattr(optim, optimizer_name)(self.net.parameters(), lr=lr)

    def train_pass(self):
        self.net.train()
        for batch_idx, (data, target) in enumerate(self.train_dataloader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.view(data.size(0), -1).to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.net(data)
            loss = F.nll_loss(output, target)
            loss.backward()
            self.optimizer.step()
        
    
    def val_pass(self):
        # Validation of the model.
        self.net.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(self.val_dataloader):
                # Limiting validation data.
                if batch_idx * BATCHSIZE >= N_VALID_EXAMPLES:
                    break
                data, target = data.view(data.size(0), -1).to(self.device), target.to(self.device)
                output = self.net(data)
                # Get the index of the max log-probability.
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / min(len(self.val_dataloader.dataset), N_VALID_EXAMPLES)
        return accuracy

    def train_epoch(self):
        self.train_pass()
        val_acc = self.val_pass()
        return val_acc


class Hyper_Optimizer:

    def __init__(self, study_name, pruner, sampler, device, NN, group,
                    results_folder, load_study=False, sql=False):
        self.study_name = study_name
        self.study = self.create_study(pruner, sampler, sql, load_study=load_study)
        self.results_folder = results_folder
        self.device = device
        self.num_NN = NN
        self.group = group
        self.wandbc = self.init_wandb(study_name)
        self.parameters = None

    def create_study(self, pruner, sampler, sql, load_study=False):
        if sql:
            optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))
            storage_name = "sqlite:///{}.db".format(self.study_name)
            if load_study:
                study = optuna.create_study(direction="maximize", study_name=self.study_name, pruner=pruner,
                                                storage=storage_name, load_if_exists=True)
            else:
                study = optuna.create_study(direction="maximize", study_name=self.study_name, pruner=pruner,
                                                storage=storage_name)
        else:
            if load_study:
                study = joblib.load(load_study)
            else:
                study = optuna.create_study(direction="maximize", study_name=self.study_name, pruner=pruner)
        return study




    def init_wandb(self, name):
        wandb_kwargs = {"project": name, "group":self.group, "reinit": True}
        wandbc = WeightsAndBiasesCallback(
                        metric_name="final validation accuracy",
                        wandb_kwargs=wandb_kwargs, as_multirun=True)
        return wandbc

    def get_mnist(self):
        # Load FashionMNIST dataset.
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(DIR, train=True, download=True, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE,
            shuffle=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST(DIR, train=False, transform=transforms.ToTensor()),
            batch_size=BATCHSIZE,
            shuffle=True,
        )

        return train_loader, valid_loader


    def define_parameters(self, trial):
        parameters = {}
        parameters["optimizer"] = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
        parameters["lr"] = trial.suggest_float("lr", 1e-4, 1e-1, log=True)
        parameters["n_layers"] =  trial.suggest_int("n_layers", 1, 3)
        parameters["dropout_p"] = trial.suggest_float("dropout_p", 0.2, 0.5)
        parameters["n_hidden"] =  trial.suggest_int("n_hidden", 4, 128)

        return parameters

    def run_optimization(self, n_trials, timeout):
        
        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout, callbacks=[self.wandbc],
                                gc_after_trial=True)

    def report_optimization(self):
        pruned_trials = self.study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = self.study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(self.study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = self.study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))


    def create_NNSet(self, parameters):
        train_NNs = []
        for n in range(self.num_NN):
            trainloader1, valloader1 = self.get_mnist()
            train_NN1 = self.create_actor(trainloader1, valloader1, parameters)

            train_NNs.append(train_NN1)
        return train_NNs

    def create_actor(self, trainloader, valloader, parameters):
        train_NN = Trainable_NN.remote(trainloader, valloader, self.device)
        train_NN.set_NN.remote(parameters)
        train_NN.set_optimizer.remote(parameters)
        return train_NN

    def create_name(self, parameters):
        name = ""
        for k,v in parameters.items():
            name += "{}_{}".format(k,v)
        return name

    def add_advice(self, advice):
        self.study.enqueue_trial(advice)

    def objective(self, trial):
        # Define Parameters
        parameters = self.define_parameters(trial)
        self.parameters = parameters.keys()
        config = dict(trial.params)
        config["trial.number"] = trial.number
        wandb_run = wandb.init(project=self.study_name, name=self.create_name(parameters),# NOTE: this entity depends on your wandb account.
                                group=self.group, config=config,reinit=True)
        # Get the FashionMNIST dataset.
        train_NNs = self.create_NNSet(parameters)
        # Training of the model.
        for epoch in tqdm(range(EPOCHS)):
            acc_loss = ray.get([trains.train_epoch.remote() for trains in train_NNs])
            acc = 0
            for n in acc_loss:
                acc+=n
            acc/=self.num_NN
            wandb_run.log(data={"validation accuracy": acc}, step=epoch)
            trial.report(acc, epoch)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                wandb_run.summary["state"] = "pruned"
                wandb_run.finish(quiet=True)
                for actor in train_NNs:
                    ray.kill(actor)
                raise optuna.exceptions.TrialPruned()
        for actor in train_NNs:
            ray.kill(actor)
        wandb_run.summary["final accuracy"] = acc
        wandb_run.summary["state"] = "complated"
        wandb_run.finish(quiet=True)
        return acc

    def run_save(self, name=None):
        if name is None:
            name = "{}/{}_{}".format(self.results_folder, self.study_name, self.group)
        joblib.dump(self.study, "{}.pkl".format(name))
        with open("{}_sampler.pkl".format(name), "wb") as fout:
            pickle.dump(self.study.sampler, fout)


    def visualization(self):
        wandb_run = wandb.init(project=self.study_name, name="{}_resume".format(self.group),
                                group=self.group, # NOTE: this entity depends on your wandb account.
                                reinit=True)
        fig = optuna.visualization.plot_param_importances(self.study)
        plt.title("Lable Importance")
        wandb.log({"chart": fig})
        fig = optuna.visualization.plot_optimization_history(self.study)
        wandb.log({"chart": fig})
        fig = optuna.visualization.plot_intermediate_values(self.study)
        wandb.log({"chart":fig})
        fig = optuna.visualization.plot_parallel_coordinate(self.study, params=self.parameters)
        wandb.log({"chart":fig})

if __name__ == "__main__":
    if ray.is_initialized:
        ray.shutdown()
    ray.init(logging_level=logging.ERROR)
    optimizer = Hyper_Optimizer(study_name="Optuna_test", results_folder="/zhome/94/2/142348/CIFAR_HOpt/results_try/",
                                    load_study=False, sql=False, device="cuda", NN=2, group="try4",
                                    sampler=optuna.samplers.RandomSampler(),
                                    pruner=optuna.pruners.SuccessiveHalvingPruner())
    advice = {"optimizer": "Adam", "lr":1e-3, "n_layers":2,"dropout_p":0.2, "n_hidden":10}
    optimizer.add_advice(advice=advice)
    optimizer.run_optimization(n_trials=50, timeout=600)
    optimizer.report_optimization()
    optimizer.visualization()
    optimizer.run_save()
