import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import trange


X_MIN = -5
X_MAX = 5
AMPL_MIN = 0.1
AMPL_MAX = 5
PHASE_MIN = 0
PHASE_MAX = np.pi

AMPL_EVAL = 4
PHASE_EVAL = 0

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
        self.weights = list(model.parameters())  # the maml weights we will be meta-optimising
        self.criterion = nn.MSELoss()
        self.meta_optimiser = torch.optim.Adam(self.weights, meta_lr)

        # hyperparameters
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.batch_size = batch_size
        self.inner_steps = inner_steps  # with the current design of MAML, >1 is unlikely to work well
        self.num_tasks = num_tasks

        # metrics
        self.print_every = 500
        self.writer = SummaryWriter('regr_logs')

    def meta_train(self, num_iterations):
        for iteration in trange(num_iterations):

            # compute meta loss
            meta_loss = 0
            for i in range(self.num_tasks):
                # reset inner model to current maml weights
                temp_weights = [w.clone() for w in self.weights]

                # perform training on data sampled from task
                for step in range(self.inner_steps):
                    xs, ys = self.task_provider(i, self.batch_size)
                    inner_loss = self.criterion(self.model.parameterised(xs, temp_weights), ys) / self.batch_size

                    # compute grad and update inner loop weights
                    grad = torch.autograd.grad(inner_loss, temp_weights)
                    temp_weights = [w - self.inner_lr * g for w, g in zip(temp_weights, grad)]

                # sample new data for meta-update and compute loss
                xt, yt = self.task_provider(i, self.batch_size)
                meta_loss += self.criterion(self.model.parameterised(xt, temp_weights), yt) / self.batch_size

            # compute meta gradient of loss with respect to maml weights
            meta_grads = torch.autograd.grad(meta_loss, self.weights)

            # assign meta gradient to weights and take optimisation step
            for w, g in zip(self.weights, meta_grads):
                w.grad = g
            self.meta_optimiser.step()

            # log metrics
            epoch_loss = meta_loss.item() / self.num_tasks

            self.writer.add_scalar('epoch_loss', epoch_loss, iteration)

            if iteration % self.print_every == 0:
                print('{}/{}. epoch loss: {}'.format(
                    iteration, num_iterations, epoch_loss))
                self.model.save('sin_model')


maml = MAML(
    MAMLModel(),
    get_task,
    inner_lr=INNER_LR,
    meta_lr=META_LR,
    batch_size=BATCH_SIZE,
    inner_steps=INNER_STEPS,
    num_tasks=NUM_TASKS
)

maml.meta_train(num_iterations=OUTER_STEPS)

#maml.model.load('sin_model')


def eval(initial_model, optim=torch.optim.SGD):
    """
    trains the model on a random sine task and measures the loss curve.

    for each n in num_steps_measured, records the model function after n gradient updates.
    """

    # copy MAML model into a new object to preserve MAML weights during training
    model = MAMLModel().model
    model.load_state_dict(initial_model.state_dict())
    criterion = nn.MSELoss()
    optimiser = optim(model.parameters(), INNER_LR)

    losses = []
    for step in range(INNER_STEPS):
        x, y = generate_batch(ampl=AMPL_EVAL, phase=PHASE_EVAL, size=BATCH_SIZE)
        loss = criterion(model(x), y) / BATCH_SIZE
        losses.append(loss.item())

        # compute grad and update inner loop weights
        model.zero_grad()
        loss.backward()
        optimiser.step()

    return losses





plt.plot(eval(maml.model.model), label='maml')
#plt.plot(average_losses(pretrained,       n_samples=5000, K=10), label='pretrained')
plt.legend()
plt.title("Average learning trajectory for K=10, starting from initial weights")
plt.xlabel("gradient steps taken with SGD")
plt.show()


plt.plot(eval(maml.model.model, optim=torch.optim.Adam), label='maml')
#plt.plot(average_losses(pretrained,       n_samples=5000, K=10, optim=torch.optim.Adam), label='pretrained')
plt.legend()
plt.title("Average learning trajectory for K=10, starting from initial weights")
plt.xlabel("gradient steps taken with Adam")
plt.show()

