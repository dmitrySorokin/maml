import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import trange
import higher


X_MIN = -5
X_MAX = 5
AMPL_MIN = 0.1
AMPL_MAX = 5
PHASE_MIN = 0
PHASE_MAX = np.pi

INNER_LR = 0.01
META_LR = 0.001
NUM_TASKS = 1000
BATCH_SIZE = 10
INNER_STEPS = 1
OUTER_STEPS = 10000

AMPLS = []
PHASES = []
for i in range(NUM_TASKS):
    AMPLS.append(np.random.uniform(AMPL_MIN, AMPL_MAX))
    PHASES.append(np.random.uniform(PHASE_MIN, PHASE_MAX))


def generate_sin(x, ampl, phase):
    return ampl * np.sin(x + phase)


def generate_batch(ampl, phase, size):
    x = np.random.uniform(X_MIN, X_MAX, size)
    #ampl = np.random.uniform(ampl_limits[0], ampl_limits[1], size)
    #phase = np.random.uniform(phase_limits[0], phase_limits[1])
    y = generate_sin(x, ampl, phase)
    return \
        torch.from_numpy(x).float().reshape([-1, 1]), \
        torch.from_numpy(y).float().reshape([-1, 1])


def get_task(task_id, size):
    return generate_batch(AMPLS[task_id], PHASES[task_id], size)


class MAMLModel(nn.Module):
    def __init__(self):
        super(MAMLModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 40),
            nn.ReLU(),
            nn.Linear(40, 40),
            nn.ReLU(),
            nn.Linear(40, 1)
        )

    def forward(self, x):
        return self.model(x)

    def parameterised(self, x, weights):
        # like forward, but uses ``weights`` instead of ``model.parameters()``
        # it'd be nice if this could be generated automatically for any nn.Module...
        x = F.linear(x, weights[0], weights[1])
        x = F.relu(x)
        x = F.linear(x, weights[2], weights[3])
        x = F.relu(x)
        x = F.linear(x, weights[4], weights[5])
        return x

    def save(self, name):
        torch.save(self.model.state_dict(), name)

    def load(self, name):
        self.model.load_state_dict(torch.load(name))


class MAML:
    def __init__(
            self, model, task_provider, inner_lr,
            meta_lr, batch_size, inner_steps,
            num_tasks):

        # important objects
        self.task_provider = task_provider
        self.model = model
        self.criterion = nn.MSELoss()
        self.meta_opt = torch.optim.Adam(self.model.parameters(), meta_lr)
        self.inner_opt = torch.optim.SGD(self.model.parameters(), inner_lr)

        # hyperparameters
        self.batch_size = batch_size
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.num_tasks = num_tasks

        # metrics
        self.print_every = 500
        self.writer = SummaryWriter('regr_logs')

    def inner_loop(self, xs, ys, xt, yt):
        with higher.innerloop_ctx(
                model,
                self.inner_opt,
                copy_initial_weights=False,
                track_higher_grads=False
        ) as (fmodel, diffopt):
            # perform training on data sampled from task
            for step in range(self.inner_steps):
                loss = self.criterion(fmodel(xs), ys) / self.batch_size
                diffopt.step(loss)

        # sample new data for meta-update and compute loss
        loss = self.criterion(self.model(xt), yt) / self.batch_size

        return loss

    def meta_train(self, num_iterations):
        for iteration in trange(num_iterations):

            # compute meta loss
            self.meta_opt.zero_grad()
            meta_loss = 0
            for i in range(self.num_tasks):
                xs, ys = self.task_provider(i, self.batch_size)
                xt, yt = self.task_provider(i, self.batch_size)
                meta_loss += self.inner_loop(xs, ys, xt, yt)

            meta_loss.backward()
            self.meta_opt.step()

            # log metrics
            epoch_loss = meta_loss.item() / self.num_tasks

            self.writer.add_scalar('epoch_loss', epoch_loss, iteration)

            if iteration % self.print_every == 0:
                print("{}/{}. loss: {}".format(iteration, num_iterations, epoch_loss))


model = MAMLModel()

maml = MAML(
    model,
    get_task,
    inner_lr=INNER_LR,
    meta_lr=META_LR,
    batch_size=BATCH_SIZE,
    inner_steps=INNER_STEPS,
    num_tasks=NUM_TASKS
)

maml.meta_train(num_iterations=OUTER_STEPS)

model.save('sin_model')








exit(0)





def loss_on_random_task(initial_model, num_steps, optim=torch.optim.SGD):
    """
    trains the model on a random sine task and measures the loss curve.

    for each n in num_steps_measured, records the model function after n gradient updates.
    """

    # copy MAML model into a new object to preserve MAML weights during training
    model = MAMLModel().model
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), 0.01)

    # train model on a random task
    ampl = np.random.uniform(AMPL_MIN, AMPL_MAX)
    phase = np.random.uniform(PHASE_MIN, PHASE_MAX)
    x, y = generate_batch(ampl=ampl, phase=phase, size=BATCH_SIZE)

    losses = []
    for step in range(1, num_steps + 1):
        loss = criterion(model(x), y) / K
        losses.append(loss.item())

        # compute grad and update inner loop weights
        model.zero_grad()
        loss.backward()
        optimiser.step()

    return losses


def average_losses(initial_model, n_samples, K=10, n_steps=10, optim=torch.optim.SGD):
    """
    returns the average learning trajectory of the model trained for ``n_iterations`` over ``n_samples`` tasks
    """

    avg_losses = [0] * K
    for i in range(n_samples):
        losses = loss_on_random_task(initial_model, K, n_steps, optim)
        avg_losses = [l + l_new for l, l_new in zip(avg_losses, losses)]
    avg_losses = [l / n_samples for l in avg_losses]

    return avg_losses


def mixed_pretrained(iterations=500):
    """
    returns a model pretrained on a selection of ``iterations`` random tasks.
    """

    # set up model
    model = MAMLModel().model
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    # fit the model
    for i in range(iterations):
        model.zero_grad()

        x, y = tasks.sample_task().sample_data(10)
        loss = criterion(model(x), y)
        loss.backward()
        optimiser.step()

    return model



pretrained = mixed_pretrained(10000)




plt.plot(average_losses(maml.model.model, n_samples=5000, K=10), label='maml')
plt.plot(average_losses(pretrained,       n_samples=5000, K=10), label='pretrained')
plt.legend()
plt.title("Average learning trajectory for K=10, starting from initial weights")
plt.xlabel("gradient steps taken with SGD")
plt.show()


plt.plot(average_losses(maml.model.model, n_samples=5000, K=10, optim=torch.optim.Adam), label='maml')
plt.plot(average_losses(pretrained,       n_samples=5000, K=10, optim=torch.optim.Adam), label='pretrained')
plt.legend()
plt.title("Average learning trajectory for K=10, starting from initial weights")
plt.xlabel("gradient steps taken with Adam")
plt.show()


def model_functions_at_training(initial_model, X, y, sampled_steps, x_axis, optim=torch.optim.SGD, lr=0.01):
    """
    trains the model on X, y and measures the loss curve.

    for each n in sampled_steps, records model(x_axis) after n gradient updates.
    """

    # copy MAML model into a new object to preserve MAML weights during training
    model = nn.Sequential(OrderedDict([
        ('l1', nn.Linear(1, 40)),
        ('relu1', nn.ReLU()),
        ('l2', nn.Linear(40, 40)),
        ('relu2', nn.ReLU()),
        ('l3', nn.Linear(40, 1))
    ]))
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), lr)

    # train model on a random task
    num_steps = max(sampled_steps)
    K = X.shape[0]

    losses = []
    outputs = {}
    for step in range(1, num_steps + 1):
        loss = criterion(model(X), y) / K
        losses.append(loss.item())

        # compute grad and update inner loop weights
        model.zero_grad()
        loss.backward()
        optimiser.step()

        # plot the model function
        if step in sampled_steps:
            outputs[step] = model(torch.tensor(x_axis, dtype=torch.float).view(-1, 1)).detach().numpy()

    outputs['initial'] = initial_model(torch.tensor(x_axis, dtype=torch.float).view(-1, 1)).detach().numpy()

    return outputs, losses


def plot_sampled_performance(initial_model, model_name, task, X, y, optim=torch.optim.SGD, lr=0.01):
    x_axis = np.linspace(-5, 5, 1000)
    sampled_steps = [1, 10]
    outputs, losses = model_functions_at_training(initial_model,
                                                  X, y,
                                                  sampled_steps=sampled_steps,
                                                  x_axis=x_axis,
                                                  optim=optim, lr=lr)

    plt.figure(figsize=(15, 5))

    # plot the model functions
    plt.subplot(1, 2, 1)

    plt.plot(x_axis, task.true_function(x_axis), '-', color=(0, 0, 1, 0.5), label='true function')
    plt.scatter(X, y, label='data')
    plt.plot(x_axis, outputs['initial'], ':', color=(0.7, 0, 0, 1), label='initial weights')

    for step in sampled_steps:
        plt.plot(x_axis, outputs[step],
                 '-.' if step == 1 else '-', color=(0.5, 0, 0, 1),
                 label='model after {} steps'.format(step))

    plt.legend(loc='lower right')
    plt.title("Model fit: {}".format(model_name))

    # plot losses
    plt.subplot(1, 2, 2)
    plt.plot(losses)
    plt.title("Loss over time")
    plt.xlabel("gradient steps taken")
    plt.show()



K = 10
task = tasks.sample_task()
X, y = task.sample_data(K)

plot_sampled_performance(maml.model.model, 'MAML', task, X, y)



K = 5
task = tasks.sample_task()
X, y = task.sample_data(K)

plot_sampled_performance(maml.model.model, 'MAML', task, X, y)

