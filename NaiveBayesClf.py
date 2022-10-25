import argparse
import sys
import matplotlib.pyplot as plt
import numpy as np
import re
import fnmatch
from scipy.stats import norm
import matplotlib as m

def main():

    print("*****NAIVE BAYES CLASSIFIER*****\n\n")

    # check for args
    parser = argparse.ArgumentParser()
    # program allows two classes
    parser.add_argument("sample_file", help="insert a file containing samples. File should contain two dimensional sample points: (x1,y1) 'line break' (x1,y2), ...",
                    type=str)
    parser.add_argument("sample_file2", help="insert a file containing samples. File should contain two dimensional sample points: (x1,y1) 'line break' (x1,y2), ...",
                    type=str)
    # precision parameter: program computes N by N grid in order to find decision boundary
    parser.add_argument("N", help="sample size: program computes N by N grid in order to find decision boundary.",
                    type=str)
    args = parser.parse_args()

    samples_per_dim = int(args.N)
    assert samples_per_dim >= 2, "N must be at least 2."

    coords = []
    fname = args.sample_file
    with open(fname, 'r') as file:
        lines = file.readlines()

        for st in lines:
            spl = st.split(",")
            spl = [el.replace("(", "") for el in spl]
            spl = [el.replace(")", "") for el in spl]
            spl = [float(el.replace("\n", "")) for el in spl]
            coords.append(np.array(spl))
    coords = np.array(coords)

    xs = list(coords[:,0])
    ys = list(coords[:, 1])

    coords2 = []
    fname2 = args.sample_file2
    with open(fname2, 'r') as file2:
        lines2 = file2.readlines()

        for st2 in lines2:
            spl2 = st2.split(",")
            spl2 = [el.replace("(", "") for el in spl2]
            spl2 = [el.replace(")", "") for el in spl2]
            spl2 = [float(el.replace("\n", "")) for el in spl2]
            coords2.append(np.array(spl2))
    coords2 = np.array(coords2)


    xs2 = list(coords2[:,0])
    ys2 = list(coords2[:,1])

    coords_concat = np.concatenate((coords, coords2), axis=0)
    xs_concat = list(coords_concat[:,0])
    ys_concat = list(coords_concat[:, 1])

    min_x = min(xs_concat)
    min_y = min(ys_concat)
    max_x = max(xs_concat)
    max_y = max(ys_concat)

    x_space = np.linspace(min_x, max_x, samples_per_dim)
    y_space = np.linspace(min_y, max_y, samples_per_dim)

    # fit data
    # assume posterior probability to be Gaussian distributed
    mux, stdx = norm.fit(xs)
    pdf_class1x = norm.pdf(x_space, mux, stdx)

    muy, stdy = norm.fit(ys)
    pdf_class1y = norm.pdf(y_space, muy, stdy)

    mux2, stdx2 = norm.fit(xs2)
    pdf_class2x = norm.pdf(x_space, mux2, stdx2)

    muy2, stdy2 = norm.fit(ys2)
    pdf_class2y = norm.pdf(y_space, muy2, stdy2)

    plt = plot_func(coords, coords2)

    p_class1 = len(coords) / (len(coords) + len(coords2))
    p_class2 = len(coords2) / (len(coords) + len(coords2))

    field_class1 = np.zeros((x_space.size, y_space.size), dtype = float)
    field_class2 = np.zeros((x_space.size, y_space.size), dtype = float)

    pdf_class1x_stack = np.tile(pdf_class1x, (samples_per_dim,1))
    pdf_class1y_stack = np.repeat(pdf_class1y, samples_per_dim, axis = 0).reshape(samples_per_dim, samples_per_dim)
    # assume: features are independent of each other; therefore multiply all features of class 1
    field_class1 = np.log(pdf_class1x_stack * pdf_class1y_stack) + np.log(p_class1)

    pdf_class2x_stack = np.tile(pdf_class2x, (samples_per_dim,1))
    pdf_class2y_stack = np.repeat(pdf_class2y, samples_per_dim, axis = 0).reshape(samples_per_dim, samples_per_dim)
    # assume: features are independent of each other; therefore multiply all features of class 2
    field_class2 = np.log(pdf_class2x_stack * pdf_class2y_stack) + np.log(p_class2)

    # at the cedision boundary diff is ideally zero
    diff = abs(field_class2 - field_class1)
    
    precision = int(np.sqrt(samples_per_dim))
    if precision > samples_per_dim ** 2:
        precision = samples_per_dim
    if precision < 2:
        precision = 2
    indices = []
    diff_aux = diff
    
    # find zero points of diff; then fit line using polyfit
    for i in range(precision):
        idx = np.argmin(diff_aux.flatten())
        idx_x = idx // samples_per_dim
        idx_y = idx % samples_per_dim
        indices.append((idx_x, idx_y))
        diff_aux[idx_x, idx_y] = float("inf")

    X = x_space[[i[0] for i in indices]] 
    y = y_space[[i[1] for i in indices]]

    theta = np.polyfit(X, y, 1)
    y_line = theta[1] + theta[0] * x_space
    x_space_hat = np.linspace(-1 + min_x, 1 + max_x, samples_per_dim)
    plt.plot(x_space_hat, y_line, 'r')

    plt.show()

    
def plot_func(coords, coords2):

    xs = list(coords[:,0])
    ys = list(coords[:, 1])

    xs2 = list(coords2[:,0])
    ys2 = list(coords2[:, 1])

    coords_concat = np.concatenate((coords, coords2), axis=0)
    xs_concat = list(coords_concat[:,0])
    ys_concat = list(coords_concat[:, 1])    

    xmin, xmax, ymin, ymax = min(xs_concat), max(xs_concat), min(ys_concat), max(ys_concat)
    ticks_frequency = 1
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.title("NAIVE BAYES CLASSIFICATION\n")
    ax.scatter(xs, ys, c="blue", s=30, marker = "+")
    ax.scatter(xs2, ys2, c="orange",s=30, marker = "+")
    
    ax.set(xlim=(xmin-1.0, xmax+1.0), ylim=(ymin-1.0, ymax+1), aspect='equal')
    
    ax.spines['bottom'].set_position('zero')
    ax.spines['left'].set_position('zero')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_xlabel('x', size=14, labelpad=-24, x=1.03)
    ax.set_ylabel('y', size=14, labelpad=-21, y=1.02, rotation=0)

    x_ticks = np.arange(xmin, xmax+1, ticks_frequency)
    y_ticks = np.arange(ymin, ymax+1, ticks_frequency)
    ax.set_xticks(x_ticks[x_ticks != 0])
    ax.set_yticks(y_ticks[y_ticks != 0])
    ax.set_xticks(np.arange(xmin, xmax+1), minor=True)
    ax.set_yticks(np.arange(ymin, ymax+1), minor=True)
    ax.grid(which='both', color='grey', linewidth=1, linestyle='-', alpha=0.2)
    arrow_fmt = dict(markersize=4, color='black', clip_on=False)
    ax.legend(['class 1', 'class 2'])
    ax.plot((1), (0), marker='>', transform=ax.get_yaxis_transform(), ** arrow_fmt)
    ax.plot((0), (1), marker='^', transform=ax.get_xaxis_transform(), ** arrow_fmt)

    return plt

if __name__ == "__main__":
    main()