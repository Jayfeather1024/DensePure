import numpy as np
import pandas as pd

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

    def get_abstention_rate(self) -> np.ndarray:
        df = pd.read_csv(self.data_file_path, delimiter="\t")
        return 1.*(df["predict"]==-1).sum()/len(df["predict"])*100

acc = ApproximateAccuracy("results_file_path")

def latex_table_certified_accuracy(radius):
    radii = [radius]
    accuracy = acc.at_radii(radii)
    print('certified_acc:'+str(accuracy))

if __name__ == "__main__":
    #certified accuracy for imagenet
    # latex_table_certified_accuracy(0.00)
    # latex_table_certified_accuracy(0.50)
    # latex_table_certified_accuracy(1.00)
    # latex_table_certified_accuracy(1.50)
    # latex_table_certified_accuracy(2.00)
    # latex_table_certified_accuracy(3.00)

    # certified accuracy for cifar10
    latex_table_certified_accuracy(0.00)
    latex_table_certified_accuracy(0.25)
    latex_table_certified_accuracy(0.50)
    latex_table_certified_accuracy(0.75)
    latex_table_certified_accuracy(1.00)
