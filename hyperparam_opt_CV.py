import numpy as np
import os
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from filelock import FileLock
from tqdm import tqdm
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from typing import Dict
import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler

from datetime import datetime

#from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

from utils import load_data, load_test_data
from CIFAR_NN import Net

CPU_per_Actor = 4
GPU_per_Actor = 1

@ray.remote(num_gpus=GPU_per_Actor, num_cpus=CPU_per_Actor)
class Trainable_NN:

    def __init__(self, train_dataloader, val_dataloader, device):
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device

    def set_NN(self, config):
        self.net = Net(config["l1"], config["l2"])

        self.net.to(self.device)

    def set_optimizer(self, config):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.net.parameters(), lr=config["lr"], momentum=0.9)

    def train_pass(self):
        running_loss = 0
        epoch_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(self.train_dataloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
        train_loss = running_loss/epoch_steps
        train_acc = correct/total
        return train_loss, train_acc

    def val_pass(self):
        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(self.val_dataloader, 0):
            with torch.no_grad():
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.net(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = self.criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1
        loss_step = val_loss/val_steps
        acc = correct/total
        return loss_step, acc

    def train_epoch(self):
        train_loss, train_acc = self.train_pass()
        val_loss, val_acc = self.val_pass()
        return val_acc, val_loss



class HyperParam_Opt:

    def __init__(self, location_dataset, checkpoint_folder, epochs, load_checkpoint,
                    device, results_path, log_path, num_NN, wandb_project="CIFAR_example2",
                    checkpoint_freq=3):

        self.location_dataset = location_dataset
        self.checkpoint_folder = checkpoint_folder
        self.epochs = epochs
        self.load_checkpoint = load_checkpoint
        self.results_path = results_path
        self.device = device
        self.wandb_project = wandb_project
        self.checkpoint_freq = checkpoint_freq
        self.cpus_req, self.gpus_req = self.get_CPUSGPUS(num_NN)
        self.stdout_log, self.stderr_log = self.make_logfiles(log_path)
        self.num_NN = num_NN

    def get_CPUSGPUS(self, num_NN):

        return CPU_per_Actor*num_NN, GPU_per_Actor*num_NN


    def make_logfiles(self, log_path):
        date_time = datetime.today().strftime('%Y-%m-%d_%H:%M:%S')
        stderr = "{}/{}_{}.stderr".format(log_path, self.wandb_project, date_time)
        stdout = "{}/{}_{}.stdout".format(log_path, self.wandb_project, date_time)
        return stdout, stderr

    def make_checkpoint(self, val_loss, val_acc, net, optimizer):
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save((net.state_dict(), optimizer.state_dict()), path)
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report({"loss": val_loss, "accuracy": val_acc})

    def load_checkpoint(self):
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
                    )
            net.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
        return net, optimizer

    def load_dataloader(self, smoke_test, batch_size):
        if smoke_test:
            trainset, _ = load_test_data()
        else:
            trainset, _ = load_data()

        test_abs = int(len(trainset) * 0.8)
        train_subset, val_subset = random_split(
            trainset, [test_abs, len(trainset) - test_abs])

        trainloader = torch.utils.data.DataLoader(
            train_subset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0 if smoke_test else 8,
        )
        valloader = torch.utils.data.DataLoader(
            val_subset,
            batch_size=int(batch_size),
            shuffle=True,
            num_workers=0 if smoke_test else 8,
        )
        return trainloader, valloader

    def create_actor(self, trainloader, valloader, config):
        train_NN = Trainable_NN.remote(trainloader, valloader, self.device)
        train_NN.set_NN.remote(config)
        train_NN.set_optimizer.remote(config)
        return train_NN

    def create_NNset(self, config):
        train_NNs = []
        for n in range(self.num_NN):
            trainloader1, valloader1 = self.load_dataloader(config["smoke_test"], config["batch_size"])
            train_NN1 = self.create_actor(trainloader1, valloader1, config)

            train_NNs.append(train_NN1)
        return train_NNs


    def train_cifar(self, config):

       # wandb = setup_wandb(config, project=self.wandb_project)

        # Load existing checkpoint through `get_checkpoint()` API.      Does it work for several?
 #       if train.get_checkpoint() and self.load_checkpoint:
  #          net, optimizer = self.load_checkpoint()
        train_NNs = self.create_NNset(config)

        for epoch in range(self.epochs):
            acc_loss = ray.get([trains.train_epoch.remote() for trains in train_NNs])
            acc = 0
            loss = 0
            for n in acc_loss:
                acc+=n[0]
                loss+=n[1]
            acc/=2
            loss/=2
            train.report({"accuracy":acc, "loss": loss})


    def main(self, config, num_samples=10, grace_period=7):
        max_num_epochs = self.epochs
        resources_schedule = [{'CPU': 1.0, 'GPU': 0.0}] + [{'CPU': CPU_per_Actor, 'GPU': GPU_per_Actor}] * self.num_NN
        scheduler = ASHAScheduler(
            time_attr='training_iteration',
            metric='accuracy',  #on loss or on mcc?
            mode='max',
            max_t=max_num_epochs,
            grace_period=grace_period,
            reduction_factor=3,
            brackets=1,
            )
        tuner = tune.Tuner(
            tune.with_resources(
                tune.with_parameters(self.train_cifar),
                resources=tune.PlacementGroupFactory(resources_schedule)  # add memory requirement?  
            ),
            run_config=train.RunConfig(#storage_path=self.results_path,
                    log_to_file=(self.stdout_log, self.stderr_log),
                    name="CIFAR_HOpt"
#                    callbacks=[WandbLoggerCallback(project=self.wandb_project)],
                    storage_path=self.results_path, # MAYBE ERROR
                    stop={"training_iteration": self.epochs},
                    ),
            tune_config=tune.TuneConfig( #Tune.report? num_samples? max_concurrent_trials? time_budget_s? reuse_actors? 
                scheduler=scheduler,
                num_samples=num_samples,
                ),
            param_space=config,
        )

        results = tuner.fit()

        best_result = results.get_best_result("accuracy", "max")

        print("Best trial config: {}".format(best_result.config))
        print("Best trial final validation loss: {}".format(
            best_result.metrics["loss"]))
        print("Best trial final validation accuracy: {}".format(
            best_result.metrics["accuracy"]))

if __name__ == "__main__":
    N=2
    folder_name = "/work3/alff/CIFAR2/"
    smoke_test = True
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16]),
        "smoke_test": smoke_test,
        }
    instance = HyperParam_Opt(
                    location_dataset="{}/data/".format(folder_name),
                    checkpoint_folder="{}/checkpoint/".format(folder_name),
                    results_path="{}/results/".format(folder_name),
                    log_path="{}/logs/".format(folder_name),
                    epochs=15,
                    load_checkpoint=False,
                    device="cuda",
                    num_NN=N
                    )
  #  instance.train_cifar(config_try)
    instance.main(config=config, num_samples=5)



