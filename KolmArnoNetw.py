# conda activate VIP

# pip install pykan ## pykan-0.0.5
# pip install mpmath sympy tqdm

from kan import *
# import torch
from matplotlib import pyplot as plt


# формируем KAN: 2D входы, 1D выходы, 5 скрытых нейронов, 
# кубические сплайны и сетка на 5 точках.
model = KAN(width=[2,5,1], grid=5, k=3, seed=0)

# сгенерируем датасет
f = lambda x: torch.exp(torch.sin(torch.pi*x[:,[0]]) + x[:,[1]]**2)
dataset = create_dataset(f, n_var=2)
dataset['train_input'].shape, dataset['train_label'].shape
#(torch.Size([1000, 2]), torch.Size([1000, 1]))

# plot KAN at initialization
model(dataset['train_input']);
model.plot(beta=100)
plt.show()


# train the model
model.train(dataset, opt="LBFGS", steps=20, lamb=0.01, lamb_entropy=10.)

model.plot()
plt.show()

model.prune()
model.plot(mask=True)
plt.show()

# model = model.prune()
# model(dataset['train_input'])
# model.plot()
# plt.show()


model.train(dataset, opt="LBFGS", steps=50);

model.plot()
plt.show()


mode = "auto" # "manual"

if mode == "manual":
    # manual mode
    model.fix_symbolic(0,0,0,'sin');
    model.fix_symbolic(0,1,0,'x^2');
    model.fix_symbolic(1,0,0,'exp');
elif mode == "auto":
    # automatic mode
    lib = ['x','x^2','x^3','x^4','exp','log','sqrt','tanh','sin','abs']
    model.auto_symbolic(lib=lib)


model.train(dataset, opt="LBFGS", steps=50)

print(model.symbolic_formula()[0][0])