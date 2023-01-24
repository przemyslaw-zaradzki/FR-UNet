import os
import argparse
import torch
from bunch import Bunch
from ruamel.yaml import safe_load
from torch.utils.data import DataLoader
import models
from dataset import vessel_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance


def main(data_path, weight_path, dataset, experiment_date, CFG, show):
    if torch.cuda.is_available():
        checkpoint = torch.load(weight_path)
    else:
        checkpoint = torch.load(weight_path, map_location=torch.device('cpu'))
    if 'config' in checkpoint:
        CFG_ck = checkpoint['config']
    else:
        with open('config.yaml', encoding='utf-8') as file:
            CFG_ck = Bunch(safe_load(file))
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1,
                             shuffle=False,  num_workers=16, pin_memory=True)
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG_ck)
    test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, dataset, experiment_date, show)
    test.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "--dataset", type=str, required=True, help="dataset name")
    parser.add_argument("--experiment_date", type=str, required=False)
    parser.add_argument("-dp", "--dataset_path", required=False, type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--wetght_path", required=False, type=str,
                        help='the path of wetght.pt')
    parser.add_argument("--show", help="save predict image",
                        required=False, default=False, action="store_true")
    args = parser.parse_args()
    with open("config.yaml", encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))

    if args.dataset_path is None:
        args.dataset_path = os.path.join('data', args.dataset_name)
    if args.experiment_date is None:
        models_dir_path = os.path.join('saved/FR_UNet', args.dataset_name)
        experiments_list = sorted(os.listdir(models_dir_path))
        args.experiment_date = experiments_list[-1]
    if args.wetght_path is None:
        args.wetght_path = os.path.join('saved/FR_UNet', args.dataset_name, args.experiment_date, 'best_model.pth')

    main(args.dataset_path, args.wetght_path, args.dataset_name, args.experiment_date, CFG, args.show)
