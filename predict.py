
"""
Contains methods to load the models and make predictions on the Test-data from arXiv:2202.13947
"""
import warnings

warnings.filterwarnings("ignore")
import numpy as np
from data import CIFData, collate_pool, DataLoader
import csv
import utils


def get_loaders(test_dir="./test_data"):
    """Initializes the data loaders for Test-relaxed and Test-unrelaxed
    Parameters
    ----------
    test_dir: String
      directory containing test data
    Returns
    -------
    relaxed_loader: data.DataLoader
      data loader for Test-relaxed
    unrelaxed_loader: data.DataLoader
      data loader for Test-unrelaxed
    """
    dataset_relaxed = CIFData(f"{test_dir}/relaxed")
    dataset_unrelaxed = CIFData(f"{test_dir}/unrelaxed")
    relaxed_loader = DataLoader(
        dataset_relaxed,
        batch_size=len(dataset_relaxed),
        collate_fn=collate_pool,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )
    unrelaxed_loader = DataLoader(
        dataset_unrelaxed,
        batch_size=len(dataset_unrelaxed),
        collate_fn=collate_pool,
        num_workers=0,
        shuffle=True,
        pin_memory=True,
    )
    return relaxed_loader, unrelaxed_loader


def write_csv(file_name, target, predicted):
    """Write prediction results to csv file. The first row is the DFT value the second row is the predicted value
    Parameters
    ----------
    file_name: String
      Name of csv file
    target: List or np.ndarray
      List of target values
    predicted: List or np.ndarray
      List of predicted values
    """
    with open(f"./{file_name}", "w") as f:
        csv_writer = csv.writer(f)
        for tar, pred in zip(target, predicted):
            csv_writer.writerow((tar, pred))


def run_test(name, org,relaxed_loader,unrelaxed_loader, write_relaxed=False, write_unrelaxed=False):
    """Convient method to load pretrained models and make predictions for Test-relaxed and Test-unrelaxed
    Parameters
    ----------
    name: String
      Name of pre-trained model
    org: Bool
       if True the CGCNN model is loaded. if False the CGCNN-HD model is loaded
    relaxed_loader: data.DataLoader
      data loader for Test-relaxed
    unrelaxed_loader: data.DataLoader
      data loader for Test-Unrelaxed
    write_relaxed: Bool
      if True the results for Test-relaxed are writen to csv
    write_unrelaxed: Bool 
      if True the results for Test-unrelaxed are written to csv
    """
    model_path = "pre_trained/" + name + ".pth.tar"
    normalizer, model = utils.get_model(
        model_path, n_h=6, h_fea_len=64, n_convs=3, org=org
    )
    target_relaxed, pred_relaxed = utils.predict(model, normalizer, relaxed_loader)
    target_unrelaxed, pred_unrelaxed = utils.predict(
        model, normalizer, unrelaxed_loader
    )

    mae_r, rmse_r, r2_r = utils.eval_predictions(target_relaxed, pred_relaxed)
    mae, rmse, r2 = utils.eval_predictions(target_unrelaxed, pred_unrelaxed)
    print(
        f"{name}: RMSE[unrelaxed/relaxed] = {rmse:.2f}/{rmse_r:.2f},\t MAE = {mae:.2f}/{mae_r:.2f},\t r2 = {r2:.4f}/{r2_r:.4f}"
    )
    ### writes prediction to csv file ###
    if write_relaxed:
        file_name = f"./{name}_relaxed.csv"
        write_csv(file_name, target_relaxed, pred_relaxed)

    if write_unrelaxed:
        file_name = f"./{name}_unrelaxed.csv"
        write_csv(file_name, target_unrelaxed, pred_unrelaxed)



if __name__ == "__main__":
    relaxed_loader, unrelaxed_loader = get_loaders()
    model_names = ["CGCNN", "PCGCNN", "PCGCNN_HD", "CGCNN_HD"]
    orgs = [True, True, False, False]
    for org, name in zip(orgs, model_names):
        run_test(name,org,relaxed_loader, unrelaxed_loader, False,True)
