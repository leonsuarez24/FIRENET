from network import ConvLSTM
from utils import (
    AverageMeter,
    TempDataset,
    save_metrics,
    set_seed,
    save_npy_metric,
)
import argparse
import os
from tqdm import tqdm
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
import numpy as np
import logging
import torch
import wandb


def main(args):
    set_seed(args.seed)
    PATH_NAME = f"lr_{args.lr}_b_{args.batch_size}_e{args.epochs}_sd_{args.seed}_hd_{args.hidden_dim}_ks_{args.kernel_size}_nl_{args.num_layers}"
    args.save_path = args.save_path + PATH_NAME
    if os.path.exists(f"{args.save_path}/metrics/metrics.npy"):
        raise FileExistsError("Model already trained")

    _, model_path, metrics_path = save_metrics(args.save_path)
    current_pnsr = 0

    logging.basicConfig(
        filename=f"{metrics_path}/train.log", level=logging.INFO, format="%(asctime)s %(message)s"
    )

    logging.info(f"Training model with parameters: {args}")

    loss_train_rec = np.zeros(args.epochs)
    ssim_train_rec = np.zeros(args.epochs)
    pnsr_train_rec = np.zeros(args.epochs)
    loss_val_rec = np.zeros(args.epochs)
    ssim_val_rec = np.zeros(args.epochs)
    pnsr_val_rec = np.zeros(args.epochs)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    SSIM = StructuralSimilarityIndexMeasure().to(device)
    PSNR = PeakSignalNoiseRatio().to(device)

    dataset = TempDataset(
        batch_size=args.batch_size,
        seed=args.seed,
        input_seq_len=args.input_seq_len,
        output_seq_len=args.output_seq_len,
    )
    train_loader, val_loader, test_loader = dataset.get_loaders()

    model = ConvLSTM(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        num_layers=args.num_layers,
        batch_first=True,
        bias=True,
        return_all_layers=False,
    ).to(device)

    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=0.9)
    criterion = torch.nn.MSELoss()

    wandb.init(project=args.project_name, config=args)

    for epoch in range(args.epochs):
        model.train()

        train_loss = AverageMeter()
        train_ssim = AverageMeter()
        train_psnr = AverageMeter()

        data_loop_train = tqdm(enumerate(train_loader), total=len(train_loader), colour="red")
        for _, (input_tensor, target) in data_loop_train:

            input_tensor = input_tensor.to(device)
            target = target.to(device)

            outputs, _ = model(input_tensor)
            loss = criterion(outputs[-1], target)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.update(loss.item())
            train_ssim.update(SSIM(outputs[-1].squeeze(2), target.squeeze(2)).item())
            train_psnr.update(PSNR(outputs[-1].squeeze(2), target.squeeze(2)).item())

            data_loop_train.set_description(f"Epoch: {epoch + 1}/{args.epochs}")
            data_loop_train.set_postfix(
                loss=train_loss.avg, ssim=train_ssim.avg, pnsr=train_psnr.avg
            )

        logging.info(
            f"Epoch: {epoch + 1}/{args.epochs} Train Loss: {train_loss.avg} Train SSIM: {train_ssim.avg} Train PSNR: {train_psnr.avg}"
        )

        val_loss = AverageMeter()
        val_ssim = AverageMeter()
        val_psnr = AverageMeter()
        data_loop_val = tqdm(enumerate(val_loader), total=len(val_loader), colour="green")
        with torch.no_grad():
            model.eval()
            for _, (input_tensor, target) in data_loop_val:

                input_tensor = input_tensor.to(device)
                target = target.to(device)

                outputs, _ = model(input_tensor)
                loss = criterion(outputs[-1], target)

                val_loss.update(loss.item())
                val_ssim.update(SSIM(outputs[-1].squeeze(2), target.squeeze(2)).item())
                val_psnr.update(PSNR(outputs[-1].squeeze(2), target.squeeze(2)).item())

                data_loop_val.set_description(f"Epoch: {epoch + 1}/{args.epochs}")
                data_loop_val.set_postfix(loss=val_loss.avg, ssim=val_ssim.avg, pnsr=val_psnr.avg)

        logging.info(
            f"Epoch: {epoch + 1}/{args.epochs} Val Loss: {val_loss.avg} Val SSIM: {val_ssim.avg} Val PSNR: {val_psnr.avg}"
        )

        if val_psnr.avg > current_pnsr:
            current_pnsr = val_psnr.avg
            torch.save(model.state_dict(), f"{model_path}/model.pth")

        wandb.log(
            {
                "Train Loss": train_loss.avg,
                "Train SSIM": train_ssim.avg,
                "Train PSNR": train_psnr.avg,
                "Val Loss": val_loss.avg,
                "Val SSIM": val_ssim.avg,
                "Val PSNR": val_psnr.avg,
            }
        )

        loss_train_rec[epoch] = train_loss.avg
        ssim_train_rec[epoch] = train_ssim.avg
        pnsr_train_rec[epoch] = train_psnr.avg
        loss_val_rec[epoch] = val_loss.avg
        ssim_val_rec[epoch] = val_ssim.avg
        pnsr_val_rec[epoch] = val_psnr.avg

    test_loss = AverageMeter()
    test_ssim = AverageMeter()
    test_psnr = AverageMeter()

    del model

    model = ConvLSTM(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        kernel_size=args.kernel_size,
        num_layers=args.num_layers,
        batch_first=True,
    ).to(device)

    model.load_state_dict(torch.load(f"{model_path}/model.pth"))

    data_loop_test = tqdm(enumerate(test_loader), total=len(test_loader), colour="blue")
    with torch.no_grad():
        model.eval()
        for _, (input_tensor, target) in data_loop_test:

            input_tensor = input_tensor.to(device)
            target = target.to(device)

            outputs, _ = model(input_tensor)
            loss = criterion(outputs[-1], target)

            test_loss.update(loss.item())
            test_ssim.update(SSIM(outputs[-1].squeeze(2), target.squeeze(2)).item())
            test_psnr.update(PSNR(outputs[-1].squeeze(2), target.squeeze(2)).item())

            data_loop_test.set_description(f"Epoch: {epoch + 1}/{args.epochs}")
            data_loop_test.set_postfix(loss=test_loss.avg, ssim=test_ssim.avg, pnsr=test_psnr.avg)

    logging.info(
        f"Test Loss: {test_loss.avg} Test SSIM: {test_ssim.avg} Test PSNR: {test_psnr.avg}"
    )

    save_npy_metric(
        dict(
            loss_train=loss_train_rec,
            ssim_train=ssim_train_rec,
            pnsr_train=pnsr_train_rec,
            loss_val=loss_val_rec,
            ssim_val=ssim_val_rec,
            pnsr_val=pnsr_val_rec,
        ),
        f"{metrics_path}/metrics",
    )

    wandb.log(
        {
            "Test Loss": test_loss.avg,
            "Test SSIM": test_ssim.avg,
            "Test PSNR": test_psnr.avg,
        }
    )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FireNet")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--lr", type=float, default=1e-1, help="learning rate")
    parser.add_argument("--seed", type=int, default=1, help="seed")
    parser.add_argument("--hidden_dim", type=list, default=[16, 32, 32, 1], help="hidden dimension")
    parser.add_argument(
        "--kernel_size", type=list, default=[(3, 3), (3, 3), (3, 3), (3, 3)], help="kernel size"
    )
    parser.add_argument("--num_layers", type=int, default=4, help="number of layers")
    parser.add_argument(
        "--save_path", type=str, default="model/weights/", help="path to save model"
    )
    parser.add_argument("--project_name", type=str, default="FireNet", help="project name")
    parser.add_argument("--input_seq_len", type=int, default=6, help="input sequence length")
    parser.add_argument("--output_seq_len", type=int, default=6, help="output sequence")

    args = parser.parse_args()
    main(args)
