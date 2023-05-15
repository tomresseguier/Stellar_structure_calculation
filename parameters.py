exec(open("./constants.py").read());

X = 0.69;  #X = 0.7438
Y = 0.28;  #Y = 0.2423
Z = 0.03;  #Z = 1 - X - Y
M = 1.1 * Ms;
mu = 4 / (5*X + 3);
mr = M * 1e-10;
shooting_fraction = 0.2;
N_steps = 1000;


