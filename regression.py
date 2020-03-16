import torch
import higher
import numpy as np
from torch import nn
from torch import functional as F
from torch import optim
from matplotlib import pyplot as plt
from copy import deepcopy
from tqdm import trange
from tensorboardX import SummaryWriter
from torchviz import make_dot
from collections import defaultdict



HIDDEN_SIZE = 40

AMPL_MIN = 0.1
AMPL_MAX = 5.0
PHASE_MIN = 0
PHASE_MAX = np.pi
X_MIN = -5.0
X_MAX = 5.0

BATCH_SIZE = 10
NUM_TASKS = 1000
NUM_UPDATES = 1

INNER_LR = 0.01
META_LR = 0.001

AMPL_EVAL = 4
PHASE_EVAL = 3

DEVICE = torch.device('cpu')
print(DEVICE)

writer = SummaryWriter('logs/sin')


def generate_sin(x, ampl, phase):
    return ampl * np.sin(x + phase)


def generate_batch(ampl, phase, size):
    x = np.random.uniform(X_MIN, X_MAX, size)
    #ampl = np.random.uniform(ampl_limits[0], ampl_limits[1], size)
    #phase = np.random.uniform(phase_limits[0], phase_limits[1])
    y = generate_sin(x, ampl, phase)
    return \
        torch.from_numpy(x).float().reshape([-1, 1]).to(DEVICE), \
        torch.from_numpy(y).float().reshape([-1, 1]).to(DEVICE)


MODEL = nn.Sequential(
    nn.Linear(1, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
    nn.ReLU(),
    nn.Linear(HIDDEN_SIZE, 1)
).to(DEVICE)



LOSS = nn.MSELoss()
OPT = optim.Adam(MODEL.parameters(), lr=META_LR)


def train_maml(model, opt, loss):
    model.train()
    n_train_iter = 10000
    losses = []
    for step_id in trange(n_train_iter):
        opt.zero_grad()
        loss_before_update = defaultdict(list)

        for task_id in range(NUM_TASKS):
            ampl = np.random.uniform(AMPL_MIN, AMPL_MAX)
            phase = np.random.uniform(PHASE_MIN, PHASE_MAX)

            inner_opt = optim.Adam(model.parameters(), lr=INNER_LR)
            inner_opt.zero_grad()

            with higher.innerloop_ctx(model, inner_opt, copy_initial_weights=False) as (fmodel, diffopt):
                for i in range(NUM_UPDATES):
                    xs, ys = generate_batch(ampl, phase, BATCH_SIZE)
                    logits = fmodel(xs)  # modified `params` can also be passed as a kwarg
                    l = loss(logits, ys)  # no need to call loss.backwards()

                    grad = torch.autograd.grad(l, fmodel.fast_params)
                    fmodel.fast_params = [w - INNER_LR * g for w, g in zip(fmodel.fast_params, grad)]


                    #diffopt.step(l)  # note that `step` must take `loss` as an argument!
                    loss_before_update[i].append(l.item())
                    #print(list(fmodel.parameters())[0].grad)
                    # The line above gets P[t+1] from P[t] and loss[t]. `step` also returns
                    # these new parameters, as an alternative to getting them from
                    # `fmodel.fast_params` or `fmodel.parameters()` after calling
                    # `diffopt.step`.

                    # At this point, or at any point in the iteration, you can take the
                    # gradient of `fmodel.parameters()` (or equivalently
                    # `fmodel.fast_params`) w.r.t. `fmodel.parameters(time=0)` (equivalently
                    # `fmodel.init_fast_params`). i.e. `fast_params` will always have
                    # `grad_fn` as an attribute, and be part of the gradient tape.

                xt, yt = generate_batch(ampl, phase, BATCH_SIZE)
                logits = fmodel(xt)
                l = loss(logits, yt)
                l.backward()
                #make_dot(l).render("attached", format="png")
        #print(list(model.children())[0].weight.grad)

        opt.step()

        writer.add_scalar('loss before update', np.mean(loss_before_update[0]), step_id)
        for i in range(1, NUM_UPDATES):
            writer.add_scalar('loss_after_{}_update'.format(i), np.mean(loss_before_update[i]), step_id)


        if step_id % 500 == 0:
            torch.save(model, 'sin_model')
            print('before: ', np.mean(loss_before_update[0]), end=';')
            for i in range(1, NUM_UPDATES):
                print('after {}: {}'.format(i ,np.mean(loss_before_update[i])), end=', ')
            print()

        losses.append(np.mean(loss_before_update[1]))

    return losses


def train(model, opt, loss):
    losses = []
    for _ in trange(7000):
        opt.zero_grad()
        ampl = np.random.uniform(AMPL_MIN, AMPL_MIN, BATCH_SIZE)
        phase = np.random.uniform(PHASE_MIN, PHASE_MAX, BATCH_SIZE)
        x, y = generate_batch(ampl, phase, BATCH_SIZE)
        pred = model(x)
        l = loss(y, pred)
        l.backward()
        opt.step()

        losses.append(float(l.detach()))
    return losses


def adapt(model, opt, loss, ampl, phase):
    losses = []
    x, y = generate_batch(ampl, phase, BATCH_SIZE)
    for _ in trange(8):
        opt.zero_grad()
        pred = model(x)
        l = loss(y, pred)
        l.backward()
        opt.step()

        losses.append(float(l.detach()))
    return losses


fig, axis = plt.subplots(2, 2)
x = np.linspace(X_MIN, X_MAX, 100).reshape(-1, 1)
xt = torch.from_numpy(x).float().to(DEVICE)

losses_train = train_maml(MODEL, OPT, LOSS)
axis[0][0].plot(losses_train)
axis[0][0].set_title('loss_train')
#MODEL.load_state_dict(torch.load('sin_model').state_dict())

y = MODEL(xt).detach().cpu().numpy()
axis[1][0].plot(x, y)
axis[1][0].set_title('fit')
axis[1][0].plot(x, generate_sin(x, AMPL_EVAL, PHASE_EVAL))


losses_eval = adapt(MODEL, OPT, LOSS, AMPL_EVAL, PHASE_EVAL)
axis[0][1].plot(losses_eval)
axis[0][1].set_title('loss_eval')

y = MODEL(xt).detach().cpu().numpy()
axis[1][1].plot(x, y)
axis[1][1].set_title('fit')
axis[1][1].plot(x, generate_sin(x, AMPL_EVAL, PHASE_EVAL))


plt.show()
