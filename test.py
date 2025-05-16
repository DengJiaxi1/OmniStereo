import argparse
import os
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
import numpy as np
import time
import cv2
from tqdm import tqdm
import json

from utils.load_ocam import load_ocam
from utils.evaluation import evaluate, eval_names
from utils.write_depth import write_depth_png
from dataloader.fisheye_loader import fisheyeDataset
from models.omnistereo import OmniStereo

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

parser = argparse.ArgumentParser()
parser.add_argument('--fov', type=int, default=220, help='fov of the camera')
parser.add_argument('--max-disp', type=int, default=128, help='maxium disparity')
parser.add_argument('--max-depth', type=float, default=1000.0, help='maximum depth in meters')
parser.add_argument('--data-path', type=str, default='../../dataset/Omnidirectional_Stereo_Dataset')
parser.add_argument('--ocam-path', type=str, default='../../dataset/Omnidirectional_Stereo_Dataset/cloudy')
parser.add_argument('--val-list', type=str, default='./dataloader/data_list/urban_val.txt')
parser.add_argument('--out-path', type=str, default='./output/', help='the output path for results')
parser.add_argument('--batch-size', type=int, default=1, help='batch size')
parser.add_argument('--loadmodel', type=str, default=None, help='load model path')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save', action='store_true', default=False, help='save results')
parser.add_argument('--crop', action='store_true', default=False, help='crop depth map (Urban dataset)')
parser.add_argument('--amp', action='store_true', help='use auto mix precision')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(json.dumps(vars(args), indent=1))


def test(model, imgs, gt):
    model.eval()

    if args.cuda:
        imgs = [rgb.cuda() for rgb in imgs]
        gt = gt.cuda()

    with torch.no_grad():
        with autocast(enabled=args.amp):
            tic = time.time()
            depth_refine, depth_fusion, depth_trans, conf_trans = model(imgs)
            toc = time.time()

    mask = gt <= args.max_depth
    eval_metrics = evaluate(depth_refine, gt, mask)

    return (eval_metrics, toc - tic, depth_refine.data.cpu().numpy(), depth_fusion.data.cpu().numpy(),
            depth_trans.data.cpu().numpy(), conf_trans.data.cpu().numpy())


def main():
    #  ------------- Prepare dataset and model ----------
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    cam_list = ['cam1', 'cam2', 'cam3', 'cam4']
    ocams, poses = load_ocam(args.ocam_path, cam_list, args.fov)

    model = OmniStereo(ocams, poses, args.max_disp, args.max_depth)

    if args.cuda:
        model = torch.nn.DataParallel(model)
        model.cuda()

    if args.loadmodel is not None:
        print('Load pretrained model:', args.loadmodel)
        pretrain_dict = torch.load(args.loadmodel)
        model.load_state_dict(pretrain_dict['state_dict'])

    print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

    test_dataset = fisheyeDataset(args.data_path, args.val_list, cam_list, 'omnidepth_gt_640')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8, drop_last=False)

    #  ------------- Check the output directories -------
    if args.save:
        ckpt_name = os.path.split(args.loadmodel)
        snapshot_name = os.path.split(ckpt_name[0])[1] + '_' + os.path.splitext(ckpt_name[1])[0]
        result_dir = os.path.join(str(args.out_path), snapshot_name)
        file_list = ["conf", "depth_fusion", "depth_refine", "disp2depth", "error_map", "gt_png"]
        for dataset in test_dataset.dataset_list:
            for file in file_list:
                if not os.path.exists(os.path.join(result_dir, dataset, file)):
                    os.makedirs(os.path.join(result_dir, dataset, file))

    #  ------------- TESTING -------------------------
    total_eval_metrics = np.zeros(len(eval_names))
    total_runtime = 0
    runtime_count = 0

    for batch_idx, batch_data in enumerate(tqdm(test_loader)):
        gt_batch = batch_data['depth']
        imgs = batch_data['imgs']

        eval_metrics, runtime, depth_refine_batch, depth_fusion_batch, depth_trans, conf_trans = test(model, imgs, gt_batch)
        total_eval_metrics += eval_metrics
        if batch_idx >= 50:
            total_runtime += runtime
            runtime_count += 1

        if args.save:
            gt_batch = gt_batch.numpy()
            dataset_name = batch_data['dataset']
            img_name = batch_data['name']

            for i in range(gt_batch.shape[0]):
                name = dataset_name[i] + '_' + os.path.splitext(img_name[i])[0]

                # error map
                gt = gt_batch[i]
                depth_refine = depth_refine_batch[i]
                depth_fusion = depth_fusion_batch[i]

                if args.crop:
                    gt = gt[gt.shape[0] // 4:gt.shape[0] // 4 * 3]
                    depth_refine = depth_refine[depth_refine.shape[0] // 4:depth_refine.shape[0] // 4 * 3]

                error_map = np.abs(depth_refine - gt)

                gt_min = np.min(gt)
                gt_max = np.max(gt)

                # save depth png
                write_depth_png(os.path.join(result_dir, dataset_name[i], "gt_png", name + "_gt.png"), gt, gt_min,
                                gt_max)
                write_depth_png(os.path.join(result_dir, dataset_name[i], "depth_refine", name + "_refine.png"),
                                depth_refine, gt_min, gt_max)
                write_depth_png(os.path.join(result_dir, dataset_name[i], "depth_fusion", name + "_fusion.png"),
                                depth_fusion, gt_min, gt_max)
                write_depth_png(os.path.join(result_dir, dataset_name[i], "error_map", name + "_error_map.png"),
                                error_map, gt_min, gt_max)

                for j in range(depth_trans.shape[1]):
                    depth = depth_trans[i][j]
                    write_depth_png(os.path.join(result_dir, dataset_name[i], "disp2depth", name + f"_{j + 1}.png"),
                                    depth, gt_min, gt_max)
                    conf = conf_trans[i][j]
                    cv2.imwrite(os.path.join(result_dir, dataset_name[i], "conf", name + f"_{j + 1}_conf.png"),
                                conf * 255)

    eval_metrics = total_eval_metrics / len(test_loader)

    print('Test Results:')
    print(('{:>8}  ' * len(eval_names)).format(*eval_names))
    print(('{:8.4f}  ' * len(eval_names)).format(*eval_metrics))
    print('Average Runtime: {:.2f} ms'.format(total_runtime / runtime_count * 1000))


if __name__ == '__main__':
    main()
