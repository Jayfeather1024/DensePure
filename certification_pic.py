import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import *
import pandas as pd
import seaborn as sns
import math

sns.set()


class Accuracy(object):
    def at_radii(self, radii: np.ndarray):
        raise NotImplementedError()


class ApproximateAccuracy(Accuracy):
    def __init__(self, data_file_path: str):
        self.data_file_path = data_file_path

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\s+")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        return (df["correct"] & (df["radius"] >= radius)).mean()


class HighProbAccuracy(Accuracy):
    def __init__(self, data_file_path: str, alpha: float, rho: float):
        self.data_file_path = data_file_path
        self.alpha = alpha
        self.rho = rho

    def at_radii(self, radii: np.ndarray) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return np.array([self.at_radius(df, radius) for radius in radii])

    def at_radius(self, df: pd.DataFrame, radius: float):
        mean = (df["correct"] & (df["radius"] >= radius)).mean()
        num_examples = len(df)
        return (mean - self.alpha - math.sqrt(self.alpha * (1 - self.alpha) * math.log(1 / self.rho) / num_examples)
                - math.log(1 / self.rho) / (3 * num_examples))


class Line(object):
    def __init__(self, quantity: Accuracy, legend: str, plot_fmt: str = "", scale_x: float = 1, sec=False):
        self.quantity = quantity
        self.legend = legend
        self.plot_fmt = plot_fmt
        self.scale_x = scale_x
        self.sec = sec


def plot_certified_accuracy(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for line in lines:
        if line.sec:
            plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, dashes=[2, 2])
        else:
            plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0.0, 1.0))
    plt.xlim((0, max_radius))
    plt.tick_params(labelsize=14)
    plt.xlabel("Radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend([method.legend for method in lines], loc='upper right', fontsize=12, framealpha=0.5)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_figure5_accuracy_cifar10(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()

    # ,markerfacecolor='black', markeredgecolor='black'

    plt.plot([1,3,5,7,10,20,30,40,50], [0.81,0.88,0.9,0.9,0.92,0.91,0.92,0.92,0.92], 'b',marker='o',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40,50], [0.73,0.74,0.74,0.75,0.78,0.79,0.81,0.82,0.82], 'orange',marker='^',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40,50], [0.54,0.59,0.61,0.63,0.65,0.65,0.66,0.69,0.69], 'g',marker='s',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40,50], [0.38,0.45,0.48,0.49,0.5,0.51,0.53,0.56,0.56], 'r',marker='p',markersize=4)
    # plt.plot([1,3,5,7,10,20,30,40], [], 'm')

    # for line in lines:
    #     if line.sec:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, dashes=[2, 2])
    #     else:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0.3, 1))
    plt.xlim((0, 54))
    plt.tick_params(labelsize=14)
    plt.xlabel("Majority Vote Numbers", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(['radius = 0.0', 'radius = 0.25', 'radius = 0.5', 'radius = 0.75'], loc='lower right', fontsize=12)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()



def plot_figure5_accuracy_cifar101(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()

    # ,markerfacecolor='black', markeredgecolor='black'

    plt.plot([1,3,5,7,10,20,30,40], [0.66,0.74,0.77,0.77,0.78,0.77,0.76,0.77], 'b',marker='o',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.57,0.65,0.67,0.67,0.67,0.68,0.68,0.68], 'orange',marker='^',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.45,0.52,0.55,0.56,0.57,0.57,0.57,0.57], 'g',marker='s',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.40,0.44,0.46,0.46,0.47,0.48,0.48,0.49], 'r',marker='p',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.36,0.37,0.37,0.40,0.40,0.40,0.40,0.40], 'm',marker='P',markersize=4)

    # for line in lines:
    #     if line.sec:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, dashes=[2, 2])
    #     else:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0.3, 0.8))
    plt.xlim((0, 44))
    plt.tick_params(labelsize=14)
    plt.xlabel("Majority Vote Numbers", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(['radius = 0.0', 'radius = 0.25', 'radius = 0.5', 'radius = 0.75', 'radius=1.0'], loc='lower right', fontsize=12, framealpha=0.5)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()


def plot_figure5_accuracy_cifar102(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()

    # ,markerfacecolor='black', markeredgecolor='black'

    plt.plot([1,3,5,7,10,20,30,40], [0.37,0.48,0.51,0.52,0.53,0.55,0.55,0.55], 'b',marker='o',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.36,0.43,0.44,0.45,0.45,0.46,0.46,0.46], 'orange',marker='^',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.30,0.37,0.39,0.42,0.43,0.44,0.44,0.44], 'g',marker='s',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.24,0.29,0.31,0.33,0.35,0.35,0.35,0.35], 'r',marker='p',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.18,0.20,0.23,0.26,0.27,0.27,0.28,0.29], 'm', marker='P',markersize=4)

    # for line in lines:
    #     if line.sec:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, dashes=[2, 2])
    #     else:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0.1, 0.6))
    plt.xlim((0, 44))
    plt.tick_params(labelsize=14)
    plt.xlabel("Majority Vote Numbers", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(['radius = 0.0', 'radius = 0.25', 'radius = 0.5', 'radius = 0.75', 'radius = 1.0'], loc='lower right', fontsize=12, framealpha=0.5)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()





def plot_figure5_accuracy_imagenet(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()

    # ,markerfacecolor='black', markeredgecolor='black'

    plt.plot([1,3,5,7,10,20,30,40,50], [0.79,0.79,0.80,0.80,0.80,0.80,0.80,0.80,0.80], 'b',marker='o',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40,50], [0.76,0.76,0.77,0.77,0.77,0.78,0.78,0.78,0.78], 'orange',marker='^',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40,50], [0.69,0.72,0.73,0.74,0.75,0.76,0.76,0.76,0.76], 'g',marker='s',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40,50], [0.58,0.62,0.65,0.67,0.68,0.71,0.71,0.71,0.72], 'r',marker='p',markersize=4)
    # plt.plot([1,3,5,7,10,20,30,40], [], 'm')

    # for line in lines:
    #     if line.sec:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, dashes=[2, 2])
    #     else:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0.55, 0.85))
    plt.xlim((0, 54))
    plt.tick_params(labelsize=14)
    plt.xlabel("Majority Vote Numbers", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(['radius = 0.0', 'radius = 0.25', 'radius = 0.5', 'radius = 0.75'], loc='lower right', fontsize=12)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()




def plot_figure5_accuracy_imagenet1(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()

    # ,markerfacecolor='black', markeredgecolor='black'

    plt.plot([1,3,5,7,10,20,30,40], [0.74,0.75,0.75,0.75,0.75,0.75,0.75,0.75], 'b',marker='o',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.63,0.67,0.69,0.70,0.72,0.72,0.72,0.72], 'orange',marker='^',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.49,0.54,0.57,0.57,0.59,0.62,0.62,0.62], 'g',marker='s',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.38,0.46,0.48,0.48,0.49,0.50,0.49,0.49], 'r',marker='p',markersize=4)
    # plt.plot([1,3,5,7,10,20,30,40], [], 'm')

    # for line in lines:
    #     if line.sec:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, dashes=[2, 2])
    #     else:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0.35, 0.80))
    plt.xlim((0, 44))
    plt.tick_params(labelsize=14)
    plt.xlabel("Majority Vote Numbers", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(['radius = 0.0', 'radius = 0.5', 'radius = 1.0', 'radius = 1.5'], loc='lower right', fontsize=12, framealpha=0.5)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()






def plot_figure5_accuracy_imagenet2(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()

    # ,markerfacecolor='black', markeredgecolor='black'

    plt.plot([1,3,5,7,10,20,30,40], [0.57,0.58,0.59,0.59,0.59,0.59,0.61,0.61], 'b',marker='o',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.50,0.55,0.55,0.55,0.55,0.57,0.57,0.57], 'orange',marker='^',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.37,0.46,0.48,0.50,0.50,0.53,0.53,0.53], 'g',marker='s',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.29,0.36,0.36,0.39,0.44,0.49,0.49,0.49], 'r',marker='p',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.25,0.29,0.31,0.33,0.33,0.36,0.37,0.37], 'm',marker='P',markersize=4)
    plt.plot([1,3,5,7,10,20,30,40], [0.12,0.20,0.22,0.22,0.22,0.25,0.26,0.26], 'y',marker='*',markersize=4)
    # plt.plot([1,3,5,7,10,20,30,40], [], 'm')

    # for line in lines:
    #     if line.sec:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, dashes=[2, 2])
    #     else:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0.1, 0.65))
    plt.xlim((0, 44))
    plt.tick_params(labelsize=14)
    plt.xlabel("Majority Vote Numbers", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(['radius = 0.0', 'radius = 0.5', 'radius = 1.0', 'radius = 1.5','radius = 2.0','radius = 3.0'], loc='lower right', fontsize=12, framealpha=0.5)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()










def plot_figure5_hist(outfile: str, title: str, max_radius: float,
                            lines: List[Line], radius_step: float = 0.01) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()

    bar_width = 0.1

    x = np.arange(4)
    y1=[0.838,0.7,0.518,0.36]
    y2=[0.856,0.73,0.574,0.44]
    y3=[0.87,0.74,0.592,0.468]
    y4=[0.87,0.742,0.606,0.478]
    y5=[0.876,0.75,0.616,0.484]
    y6=[0.874,0.758,0.628,0.492]
    y7=[0.876,0.762,0.636,0.498]
    y8=[0.876,0.766,0.646,0.504]

    plt.bar(x, y1, bar_width, align='center', color=plt.get_cmap('Blues')(4), label='1 step')
    plt.bar(x+bar_width*1, y2, bar_width, align='center', color=plt.get_cmap('Blues')(300), label='3 steps')
    plt.bar(x+bar_width*2, y3, bar_width, align='center', color=plt.get_cmap('Blues')(350), label='5 steps')
    plt.bar(x+bar_width*3, y4, bar_width, align='center', color=plt.get_cmap('Blues')(400), label='7 steps')
    plt.bar(x+bar_width*4, y5, bar_width, align='center', color=plt.get_cmap('Blues')(450), label='10 steps')
    plt.bar(x+bar_width*5, y6, bar_width, align='center', color=plt.get_cmap('Blues')(500), label='20 steps')
    plt.bar(x+bar_width*6, y7, bar_width, align='center', color=plt.get_cmap('Blues')(550), label='30 steps')
    plt.bar(x+bar_width*7, y8, bar_width, align='center', color=plt.get_cmap('Blues')(600), label='40 steps')

    # plt.plot([1,3,5,7,10,20,30,40], [], 'm')

    # for line in lines:
    #     if line.sec:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt, dashes=[2, 2])
    #     else:
    #         plt.plot(radii * line.scale_x, line.quantity.at_radii(radii), line.plot_fmt)

    plt.ylim((0.3, 1))
    tick_label=['0.0', '0.25', '0.5', '0.75']
    plt.xticks(x+bar_width*4, tick_label)
    plt.tick_params(labelsize=14)
    plt.xlabel("Radius", fontsize=16)
    plt.ylabel("Certified Accuracy", fontsize=16)
    plt.legend(loc='upper right', fontsize=8)
    plt.savefig(outfile + ".pdf")
    plt.tight_layout()
    plt.title(title, fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".png", dpi=300)
    plt.close()



def smallplot_certified_accuracy(outfile: str, title: str, max_radius: float,
                                 methods: List[Line], radius_step: float = 0.01, xticks=0.5) -> None:
    radii = np.arange(0, max_radius + radius_step, radius_step)
    plt.figure()
    for method in methods:
        plt.plot(radii, method.quantity.at_radii(radii), method.plot_fmt)

    plt.ylim((0, 1))
    plt.xlim((0, max_radius))
    plt.xlabel("radius", fontsize=22)
    plt.ylabel("certified accuracy", fontsize=22)
    plt.tick_params(labelsize=20)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(xticks))
    plt.legend([method.legend for method in methods], loc='upper right', fontsize=20)
    plt.tight_layout()
    plt.savefig(outfile + ".pdf")
    plt.close()


def latex_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                   methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')

    for radius in radii:
        f.write("& $r = {:.3}$".format(radius))
    f.write("\\\\\n")

    f.write("\midrule\n")

    for i, method in enumerate(methods):
        f.write(method.legend)
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = r" & \textbf{" + "{:.2f}".format(accuracies[i, j]) + "}"
            else:
                txt = " & {:.2f}".format(accuracies[i, j])
            f.write(txt)
        f.write("\\\\\n")
    f.close()


def markdown_table_certified_accuracy(outfile: str, radius_start: float, radius_stop: float, radius_step: float,
                                      methods: List[Line]):
    radii = np.arange(radius_start, radius_stop + radius_step, radius_step)
    accuracies = np.zeros((len(methods), len(radii)))
    for i, method in enumerate(methods):
        accuracies[i, :] = method.quantity.at_radii(radii)

    f = open(outfile, 'w')
    f.write("|  | ")
    for radius in radii:
        f.write("r = {:.3} |".format(radius))
    f.write("\n")

    f.write("| --- | ")
    for i in range(len(radii)):
        f.write(" --- |")
    f.write("\n")

    for i, method in enumerate(methods):
        f.write("<b> {} </b>| ".format(method.legend))
        for j, radius in enumerate(radii):
            if i == accuracies[:, j].argmax():
                txt = "{:.2f}<b>*</b> |".format(accuracies[i, j])
            else:
                txt = "{:.2f} |".format(accuracies[i, j])
            f.write(txt)
        f.write("\n")
    f.close()


if __name__ == "__main__":
    plot_certified_accuracy(
        "cifar10_500_comparation", "", 4.0, [
            Line(ApproximateAccuracy("cifar10_500_10steps_40majority/0.25_10steps_40_500.txt"), "$Ours\ \sigma = 0.25$", plot_fmt='b'),
            Line(ApproximateAccuracy("cifar10_500_10steps_40majority/0.5_10steps_40_500.txt"), "$Ours\ \sigma = 0.50$", plot_fmt='orange'),
            Line(ApproximateAccuracy("cifar10_500_10steps_40majority/1.0_10steps_40_500.txt"), "$Ours\ \sigma = 1.00$", plot_fmt='g'),
            Line(ApproximateAccuracy("cifar10_500_1step/0.25_cifar10_500_1step.txt"), "$Carlini\ et\ al.\ \sigma = 0.25$", plot_fmt='b',sec=True),
            Line(ApproximateAccuracy("cifar10_500_1step/0.5_cifar10_500_1step"), "$Carlini\ et\ al.\ \sigma = 0.50$", plot_fmt='orange',sec=True),
            Line(ApproximateAccuracy("cifar10_500_1step/1.0_cifar10_500_1step"), "$Carlini\ et\ al.\ \sigma = 1.00$", plot_fmt='g',sec=True),
        ])

    # plot_certified_accuracy(
    # "imagenet_10steps_majority_vote", "", 1.0, [
    #     Line(ApproximateAccuracy("imagenet_right_100_10steps/imagenet_10steps_0.25_1.txt"), "$no\ Majority\ Vote$", plot_fmt='b'),
    #     Line(ApproximateAccuracy("imagenet_right_100_10steps/imagenet_10steps_0.25_10.txt"), "$10\ Majority\ Vote$", plot_fmt='orange'),
    #     Line(ApproximateAccuracy("imagenet_right_100_10steps/imagenet_10steps_0.25_20.txt"), "$20\ Majority\ Vote$", plot_fmt='g'),
    #     Line(ApproximateAccuracy("imagenet_right_100_10steps/imagenet_10steps_0.25_30.txt"), "$30\ Majority\ Vote$", plot_fmt='r'),
    #     Line(ApproximateAccuracy("imagenet_right_100_10steps/imagenet_10steps_0.25_41.txt"), "$40\ Majority\ Vote$", plot_fmt='m'),
    # ])

    # plot_certified_accuracy(
    #     "imagenet_100_comparation", "", 4.0, [
    #         Line(ApproximateAccuracy("stopseed_imagenet_10steps_0.25_40.txt"), "$Ours\ \sigma = 0.25$", plot_fmt='b'),
    #         Line(ApproximateAccuracy("stopseed_imagenet_10steps_0.50_40.txt"), "$Ours\ \sigma = 0.50$", plot_fmt='orange'),
    #         Line(ApproximateAccuracy("stopseed_imagenet_10steps_1.00_40.txt"), "$Ours\ \sigma = 1.00$", plot_fmt='g'),
    #         Line(ApproximateAccuracy("imagenet_right/merge_1step_0.25.txt"), "$Carlini\ et\ al.\ \sigma = 0.25$", plot_fmt='b',sec=True),
    #         Line(ApproximateAccuracy("imagenet_right/merge_1step_0.50.txt"), "$Carlini\ et\ al.\ \sigma = 0.50$", plot_fmt='orange',sec=True),
    #         Line(ApproximateAccuracy("imagenet_right/merge_1step_1.00.txt"), "$Carlini\ et\ al.\ \sigma = 1.00$", plot_fmt='g',sec=True),
    #     ])

    # plot_certified_accuracy(
    # "cifar10_steps", "", 1.0, [
    #     Line(ApproximateAccuracy("cifar10_steps_pic/0.25_cifar10_500_1step.txt"), "$Carlini\ et\ al.$", plot_fmt='r'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/2steps_40.txt"), "$2\ steps\ with\ MV$", plot_fmt='b'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/5steps_40.txt"), "$5\ steps\ with\ MV$", plot_fmt='orange'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/10steps_40.txt"), "$10\ steps\ with\ MV$", plot_fmt='g'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/2steps_1.txt"), "$2\ steps\ without\ MV$", plot_fmt='b',sec=True),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/5steps_1.txt"), "$5\ steps\ without\ MV$", plot_fmt='orange',sec=True),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/0.25_10steps_1.txt"), "$10\ steps\ without\ MV$", plot_fmt='g',sec=True),
    # ])

    # plot_figure5_accuracy_imagenet("imagenet_mv", "", 0.5, [])

    # plot_figure5_accuracy_imagenet1("imagenet_mv_0.5", "", 0.5, [])

    # plot_figure5_accuracy_imagenet2("imagenet_mv_1.0", "", 0.5, [])


    # plot_figure5_accuracy_cifar10("cifar10_mv", "", 0.5, [])

    # plot_figure5_accuracy_cifar101("cifar10_mv_0.5", "", 0.5, [])

    # plot_figure5_accuracy_cifar102("cifar10_mv_1.0", "", 0.5, [])


    # plot_certified_accuracy(
    # "imagenet_steps", "", 0.9, [
    #     Line(ApproximateAccuracy("imagenet_right_steps/merge_1step_0.25.txt"), "$Carlini\ et\ al.$", plot_fmt='r'),
    #     Line(ApproximateAccuracy("imagenet_right_steps/imagenet_2steps_0.25_40.txt"), "$2\ steps\ with\ MV$", plot_fmt='b'),
    #     Line(ApproximateAccuracy("imagenet_right_steps/imagenet_5steps_0.25_40.txt"), "$5\ steps\ with\ MV$", plot_fmt='orange'),
    #     Line(ApproximateAccuracy("3462286/stopseed_imagenet_10steps_0.25_50.txt"), "$10\ steps\ with\ MV$", plot_fmt='g'),
    #     Line(ApproximateAccuracy("imagenet_right_steps/imagenet_2steps_0.25_1.txt"), "$2\ steps\ without\ MV$", plot_fmt='b',sec=True),
    #     Line(ApproximateAccuracy("imagenet_right_steps/imagenet_5steps_0.25_1.txt"), "$5\ steps\ without\ MV$", plot_fmt='orange',sec=True),
    #     Line(ApproximateAccuracy("imagenet_right_steps/stopseed_imagenet_10steps_0.25_1.txt"), "$10\ steps\ without\ MV$", plot_fmt='g',sec=True),
    # ])


    # plot_certified_accuracy(
    # "imagenet_steps_0.50", "", 1.75, [
    #     Line(ApproximateAccuracy("imagenet_right/merge_1step_0.50.txt"), "$Carlini\ et\ al.$", plot_fmt='r'),
    #     Line(ApproximateAccuracy("imagenet_right/imagenet_2steps_0.50_30.txt"), "$2\ steps\ with\ MV$", plot_fmt='b'),
    #     Line(ApproximateAccuracy("imagenet_right/stopseed_imagenet_10steps_0.50_40.txt"), "$10\ steps\ with\ MV$", plot_fmt='g'),
    #     Line(ApproximateAccuracy("imagenet_right/imagenet_2steps_0.50_1.txt"), "$2\ steps\ without\ MV$", plot_fmt='b',sec=True),
    #     Line(ApproximateAccuracy("imagenet_right/stopseed_imagenet_10steps_0.50_1.txt"), "$10\ steps\ without\ MV$", plot_fmt='g',sec=True),
    # ])


    # plot_certified_accuracy(
    # "imagenet_steps_1.00", "", 3.5, [
    #     Line(ApproximateAccuracy("imagenet_right/merge_1step_1.00.txt"), "$Carlini\ et\ al.$", plot_fmt='r'),
    #     Line(ApproximateAccuracy("imagenet_right/imagenet_2steps_1.00_30.txt"), "$2\ steps\ with\ MV$", plot_fmt='b'),
    #     Line(ApproximateAccuracy("imagenet_right/stopseed_imagenet_10steps_1.00_40.txt"), "$10\ steps\ with\ MV$", plot_fmt='g'),
    #     Line(ApproximateAccuracy("imagenet_right/imagenet_2steps_1.00_1.txt"), "$2\ steps\ without\ MV$", plot_fmt='b',sec=True),
    #     Line(ApproximateAccuracy("imagenet_right/stopseed_imagenet_10steps_1.00_1.txt"), "$10\ steps\ without\ MV$", plot_fmt='g',sec=True),
    # ])


    # plot_certified_accuracy(
    # "cifar10_steps_0.50", "", 2, [
    #     Line(ApproximateAccuracy("cifar10_steps_pic/0.5_1step"), "$Carlini\ et\ al.$", plot_fmt='r'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/0.50_2steps_10.txt"), "$2\ steps\ with\ MV$", plot_fmt='b'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/0.5_10steps_40.txt"), "$10\ steps\ with\ MV$", plot_fmt='g'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/0.50_2steps_1.txt"), "$2\ steps\ without\ MV$", plot_fmt='b',sec=True),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/0.5_10steps_1.txt"), "$10\ steps\ without\ MV$", plot_fmt='g',sec=True),
    # ])


    # plot_certified_accuracy(
    # "cifar10_steps_1.00", "", 4, [
    #     Line(ApproximateAccuracy("cifar10_steps_pic/1.0_1step"), "$Carlini\ et\ al.$", plot_fmt='r'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/1.00_2steps_10.txt"), "$2\ steps\ with\ MV$", plot_fmt='b'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/1.0_10steps_40.txt"), "$10\ steps\ with\ MV$", plot_fmt='g'),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/1.00_2steps_1.txt"), "$2\ steps\ without\ MV$", plot_fmt='b',sec=True),
    #     Line(ApproximateAccuracy("cifar10_steps_pic/1.0_10steps_1.txt"), "$10\ steps\ without\ MV$", plot_fmt='g',sec=True),
    # ])


    # plot_certified_accuracy(
    #     "imagenet_2steps_5", "", 4.0, [
    #         Line(ApproximateAccuracy("imagenet_0.25_5.txt"), "$2\ steps\ 5\ MV\ \sigma = 0.25$", plot_fmt='b'),
    #         Line(ApproximateAccuracy("imagenet_0.50_5.txt"), "$2\ Steps\ 5\ MV\ \sigma = 0.50$", plot_fmt='orange'),
    #         Line(ApproximateAccuracy("imagenet_1.00_5.txt"), "$2\ Steps\ 5\ MV\ \sigma = 1.00$", plot_fmt='g'),
    #         Line(ApproximateAccuracy("imagenet_right_steps/merge_1step_0.25.txt"), "$Carlini\ et\ al.\ \sigma = 0.25$", plot_fmt='b',sec=True),
    #         Line(ApproximateAccuracy("imagenet_right/merge_1step_0.50.txt"), "$Carlini\ et\ al.\ \sigma = 0.50$", plot_fmt='orange',sec=True),
    #         Line(ApproximateAccuracy("imagenet_right/merge_1step_1.00.txt"), "$Carlini\ et\ al.\ \sigma = 1.00$", plot_fmt='g',sec=True),
    #     ])





    # plot_certified_accuracy(
    #     "imagenet_wrn", "", 4.0, [
    #         Line(ApproximateAccuracy("0.25_2steps_11.txt"), "$Ours\ \sigma = 0.25$", plot_fmt='b'),
    #         Line(ApproximateAccuracy("0.50_2steps_11.txt"), "$Ours\ \sigma = 0.50$", plot_fmt='orange'),
    #         Line(ApproximateAccuracy("1.00_2steps_11.txt"), "$Ours\ \sigma = 1.00$", plot_fmt='g'),
    #         Line(ApproximateAccuracy("merge_wrn_1step_0.25.txt"), "$Carlini\ et\ al.\ \sigma = 0.25$", plot_fmt='b',sec=True),
    #         Line(ApproximateAccuracy("merge_wrn_1step_0.50.txt"), "$Carlini\ et\ al.\ \sigma = 0.50$", plot_fmt='orange',sec=True),
    #         Line(ApproximateAccuracy("merge_wrn_1step_1.00.txt"), "$Carlini\ et\ al.\ \sigma = 1.00$", plot_fmt='g',sec=True),
    #     ])



    # plot_certified_accuracy(
    # "cifar10_wrn_0.25", "", 1.0, [
    #         Line(ApproximateAccuracy("cifar10_wrn/10steps_40.txt"), "$Ours\ ViT$", plot_fmt='b'),
    #         Line(ApproximateAccuracy("cifar10_wrn/wrn_10steps_40.txt"), "$Ours\ Wide-ResNet$", plot_fmt='orange'),
    #         Line(ApproximateAccuracy("cifar10_wrn/0.25_cifar10_500_1step.txt"), "$Carlini\ et\ al.\ ViT$", plot_fmt='b',sec=True),
    #         Line(ApproximateAccuracy("cifar10_wrn/1step.txt"), "$Carlini\ et\ al.\ Wide-ResNet$", plot_fmt='orange',sec=True),
    # ])


    # plot_certified_accuracy(
    #     "imagenet_resnet", "", 4.0, [
    #         Line(ApproximateAccuracy("resnet_imagenet_2steps_0.25_10.txt"), "$Ours\ \sigma = 0.25$", plot_fmt='b'),
    #         Line(ApproximateAccuracy("resnet_imagenet_2steps_0.50_10.txt"), "$Ours\ \sigma = 0.50$", plot_fmt='orange'),
    #         Line(ApproximateAccuracy("resnet_imagenet_2steps_1.00_10.txt"), "$Ours\ \sigma = 1.00$", plot_fmt='g'),
    #         Line(ApproximateAccuracy("merge_resnet_1step_0.25.txt"), "$Carlini\ et\ al.\ \sigma = 0.25$", plot_fmt='b',sec=True),
    #         Line(ApproximateAccuracy("merge_resnet_1step_0.50.txt"), "$Carlini\ et\ al.\ \sigma = 0.50$", plot_fmt='orange',sec=True),
    #         Line(ApproximateAccuracy("merge_resnet_1step_1.00.txt"), "$Carlini\ et\ al.\ \sigma = 1.00$", plot_fmt='g',sec=True),
    #     ])

    # plot_certified_accuracy(
    #     "imagenet_MLP", "", 4.0, [
    #         Line(ApproximateAccuracy("0.25_2steps_11.txt"), "$Ours\ \sigma = 0.25$", plot_fmt='b'),
    #         Line(ApproximateAccuracy("0.50_2steps_11.txt"), "$Ours\ \sigma = 0.50$", plot_fmt='orange'),
    #         Line(ApproximateAccuracy("1.00_2steps_11.txt"), "$Ours\ \sigma = 1.00$", plot_fmt='g'),
    #         Line(ApproximateAccuracy("merge_mlp_1step_0.25.txt"), "$Carlini\ et\ al.\ \sigma = 0.25$", plot_fmt='b',sec=True),
    #         Line(ApproximateAccuracy("merge_mlp_1step_0.50.txt"), "$Carlini\ et\ al.\ \sigma = 0.50$", plot_fmt='orange',sec=True),
    #         Line(ApproximateAccuracy("merge_mlp_1step_1.00.txt"), "$Carlini\ et\ al.\ \sigma = 1.00$", plot_fmt='g',sec=True),
    #     ])


    # plot_certified_accuracy(
    # "imagenet_wrn_0.25", "", 1.0, [
    #         Line(ApproximateAccuracy("stopseed_imagenet_10steps_0.25_40.txt"), "$Ours\ BEiT$", plot_fmt='b'),
    #         Line(ApproximateAccuracy("wrn_imagenet_10steps_0.25_20.txt"), "$Ours\ Wide-ResNet$", plot_fmt='orange'),
    #         Line(ApproximateAccuracy("merge_1step_0.25.txt"), "$Carlini\ et\ al.\ BEiT$", plot_fmt='b',sec=True),
    #         Line(ApproximateAccuracy("merge_wrn_1step_0.25.txt"), "$Carlini\ et\ al.\ Wide-ResNet$", plot_fmt='orange',sec=True),
    # ])