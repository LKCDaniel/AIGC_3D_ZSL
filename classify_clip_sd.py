import json
import os
import glob
import time
import torch
import argparse
import numpy as np
from os.path import join, isdir, isfile
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from process_projection.comfyui import Comfyui

import clip
from pytorch_lightning import seed_everything
import torch.nn.functional as F

from classify_util import chunk, get_mesh, init_render_with_ha, render_object_style, class_accuracy

class_map = {'bathtub': 0, 'bed': 1, 'chair': 2, 'desk': 3, 'dresser': 4, 'monitor': 5, 'nightstand': 6, 'sofa': 7,
             'table': 8, 'toilet': 9, 'night': 6}

test_class = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'nightstand', 'sofa', 'table', 'toilet']


def rot_img(x, degree, dtype):
    device = x.device
    theta = torch.deg2rad(degree).to(device)
    rot_mat = torch.zeros(2, 3, device=device)
    rot_mat[0, 0] = torch.cos(theta)
    rot_mat[0, 1] = -torch.sin(theta)
    rot_mat[1, 0] = torch.sin(theta)
    rot_mat[1, 1] = torch.cos(theta)
    rot_mat = rot_mat[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, x.size(), align_corners=True).type(dtype)
    x = F.grid_sample(x, grid, padding_mode='border', align_corners=True)
    return x.squeeze()  # c w h


def run(opt, comfyui=None):
    seed_everything(opt.seed)

    test_file_list = glob.glob('./datasets/ModelNet10/ModelNet10/*/test/*.off')
    test_file_list.sort()
    print(f'total {len(test_file_list)} test samples')

    pretrain_model = "ViT-B/16"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print('device: ' + device)
    model, preprocess = clip.load(pretrain_model, device=device)  # ViT-B/32
    img_norm = preprocess.transforms[4]

    print(f'prompt: {opt.prompt}, style: {opt.style}, bg_color: {opt.bg_color}, IARM: {opt.iarm}')
    if opt.comment:
        print(f'comment: {opt.comment}')
    else:
        opt.comment = opt.prompt
    if opt.aigc_dsc:
        print(f'AIGC description: {opt.aigc_dsc}')

    model = model.to(device)
    label_list = []
    pred_list = []
    probs_list = []
    time_list = []
    Tp = 0

    for f_i, f_path in enumerate(test_file_list):
        start_time = time.time()
        class_name = os.path.basename(f_path).split('_')[0]
        file_name = os.path.basename(f_path).split('.')[0]
        cur_label = class_map[class_name]
        label_list.append(cur_label)

        # ####get base information for point cloud
        xyz_raw = np.load(join(opt.data_dir, 'npy_render', f'{file_name}_xyz.npy'))
        faces = np.load(join(opt.data_dir, 'npy_render', f'{file_name}_faces.npy'))
        xyz_tensor = torch.from_numpy(xyz_raw).type(torch.float32)
        faces_tensor = torch.from_numpy(faces).type(torch.int64)
        mesh = get_mesh(xyz_tensor, faces_tensor)
        rasterizer, renderer = init_render_with_ha(opt.r_p, opt.phi_1, opt.phi_2,
                                                   img_size=opt.resolution)
        init_image, image_mask = render_object_style(rasterizer, renderer, mesh, opt.style)

        if opt.style == 'normal':
            init_image[image_mask == 0, :] = torch.tensor([0.5, 0.5, 1], dtype=torch.float32)

        projection_path = join(opt.data_dir, 'projections', f'{opt.style}_{opt.resolution}', f'{file_name}.png')
        if not isfile(projection_path):
            os.makedirs(os.path.dirname(projection_path), exist_ok=True)
            plt.imsave(projection_path, init_image)
            print(f'{opt.style} projection of {file_name} saved to {projection_path}')

        torch.no_grad()
        if comfyui:
            processed_projection_dir = join(opt.data_dir, 'processed_projections',
                                            f'{opt.style}_{opt.resolution}{f'_{opt.aigc_dsc}' if opt.aigc_dsc else ''}',
                                            file_name)
            os.makedirs(processed_projection_dir, exist_ok=True)

            probs = []
            for class_name in test_class:
                processed_image_batch = None
                for i in range(opt.num_per_input):
                    projection_processed_path = join(processed_projection_dir, f'{class_name}_{i}.png')
                    if not isfile(projection_processed_path):
                        comfyui.process_image(class_name, opt.style, projection_path, projection_processed_path)

                    processed_image = torch.from_numpy(plt.imread(projection_processed_path))

                    if opt.bg_color == 'white':
                        processed_image[image_mask == 0, ...] = 1
                    else:
                        processed_image[image_mask == 0, ...] = 0

                    masked_proj_path = join(processed_projection_dir, f'{class_name}_{i}_masked.png')
                    if not isfile(masked_proj_path):
                        os.makedirs(os.path.dirname(masked_proj_path), exist_ok=True)
                        plt_image = processed_image.cpu().numpy()
                        plt.imsave(masked_proj_path, plt_image)
                        print(f'{opt.style} projection of {file_name} saved to {masked_proj_path}')
                    # plt.imshow(processed_image)
                    # plt.show()

                    processed_image = img_norm(processed_image.to(device).permute(2, 0, 1)).unsqueeze(0)
                    if processed_image_batch is None:
                        processed_image_batch = processed_image
                    else:
                        processed_image_batch = torch.cat((processed_image_batch, processed_image), dim=0)

                if processed_image_batch.shape[2:] != (224, 224):
                    processed_image_batch = torch.nn.functional.interpolate(processed_image_batch, size=(224, 224),
                                                                            mode='bicubic')

                prompt = opt.prompt.replace('@', class_name)
                current_text = clip.tokenize(prompt).to(device)
                logits_per_image, _ = model(processed_image_batch, current_text)
                prob = torch.mean(logits_per_image.squeeze()).item()
                probs.append(prob)

            pred_cls = np.argmax(probs)

        else:
            text = []
            for name in test_class:
                text.append(opt.prompt.replace('@', name))
            text = clip.tokenize(text).to(device)

            if opt.iarm:
                rotate_list = []
                indice_list = []

                init_image_rotated = rot_img(torch.from_numpy(init_image).to(device).unsqueeze(0).permute(0, 3, 1, 2),
                                             torch.tensor([0.0]).to(device),
                                             dtype=torch.float32)

                if init_image_rotated.shape[2:] != (224, 224):
                    init_image_rotated = torch.nn.functional.interpolate(init_image_rotated, size=(224, 224),
                                                                         mode='bicubic')

                init_image_rotated = img_norm(init_image_rotated).unsqueeze(0)
                logits_per_image, logits_per_text = model(init_image_rotated, text)
                probs_0 = logits_per_image.squeeze()

                class_numbers = len(test_class)
                _, indices = torch.topk(probs_0, k=class_numbers, largest=True)
                for id_i, ind in enumerate(indices):
                    ind_k = ind % class_numbers
                    if ind_k in indice_list or len(indice_list) >= 3:
                        continue
                    else:
                        indice_list.append(ind_k.item())
                        rotate_list.append(ind // class_numbers)

                indices = torch.tensor(indice_list).type(torch.int64).to(device)
                rotate_list = torch.tensor(rotate_list).type(torch.int64).to(device)
                print('the proposed idx: {}'.format(indices))

                with torch.enable_grad():
                    init_angle_raw = torch.tensor([0.0]).to(device)
                    k_text_probs = torch.zeros(text[indices].shape[0])
                    torch.cuda.empty_cache()

                    for p_i in range(k_text_probs.shape[0]):
                        grad_step_num = opt.grad_step_num
                        step_value = opt.step_value
                        step_reduce = opt.step_reduce
                        probs_rotate = torch.zeros(grad_step_num)
                        angle_rotate = init_angle_raw * 1.0
                        angle_rotate += rotate_list[p_i] * 90
                        angle_rotate.requires_grad = True
                        print('init rotate angle {}'.format(angle_rotate))

                        for r_i in range(grad_step_num):
                            cur_step_value = step_value - step_reduce * r_i

                            if not isinstance(init_image, torch.Tensor):
                                init_image = torch.from_numpy(init_image).to(device)
                            elif init_image.device.type != 'cuda':
                                init_image = init_image.to(device)

                            init_image_rotated = rot_img(init_image.unsqueeze(0).permute(0, 3, 1, 2),
                                                         angle_rotate,
                                                         dtype=torch.float32)

                            init_image_rotated = rot_img(init_image.unsqueeze(0).permute(0, 3, 1, 2),
                                                         angle_rotate,
                                                         dtype=torch.float32)
                            init_image_rotated = img_norm(init_image_rotated).unsqueeze(0)
                            logits_per_image, logits_per_text = model(init_image_rotated,
                                                                      text[indices[p_i]].view(-1, 77))
                            probs = logits_per_image.type(torch.float32)

                            grad = torch.autograd.grad(outputs=probs, inputs=angle_rotate, only_inputs=True)[0]
                            with torch.no_grad():
                                if grad >= 0:
                                    angle_rotate = angle_rotate + cur_step_value
                                else:
                                    angle_rotate = angle_rotate - cur_step_value

                            angle_rotate.requires_grad = True

                            print('cur_prompt: {}, cur_step: {}, cur_grad: {}, cur_probs: {}, next angle: {}'
                                  .format(indices[p_i], r_i, grad, probs, angle_rotate))
                            probs_rotate[r_i] = probs
                        k_text_probs[p_i] = probs_rotate.max()
                    pred_cls = indices[torch.argmax(k_text_probs)]
                    probs = probs.detach().cpu().numpy()
                    pred_cls = pred_cls.detach().cpu().numpy()

            else:
                init_image = img_norm(torch.from_numpy(init_image).to(device).permute(2, 0, 1)).unsqueeze(0)

                if init_image.shape[2:] != (224, 224):
                    init_image = torch.nn.functional.interpolate(init_image, size=(224, 224), mode='bicubic')
                logits_per_image, logits_per_text = model(init_image, text)
                probs = logits_per_image.softmax(dim=-1).squeeze()
                pred_cls = torch.argmax(probs)
                probs = probs.detach().cpu().numpy()

                pred_cls = pred_cls.detach().cpu().numpy()

        if pred_cls == cur_label:
            Tp += 1
        acc = Tp / (f_i + 1)
        pred_list.append(pred_cls)
        probs_list.append(probs)
        time_list.append(time.time() - start_time)

        print(f'========complete: {f_i + 1}/{len(test_file_list)} samples, current: {file_name}, label/pred: '
              f'{cur_label}/{pred_cls}, all acc: {acc}============')

    per_accuracy = class_accuracy(labels=np.array(label_list), predictions=np.array(pred_list), test_class=test_class)
    print('average time for each sample is {}'.format(np.mean(time_list)))

    run_th = -1
    file_list = glob.glob(join(opt.data_dir, 'results', '*'))
    for file in file_list:
        file = os.path.basename(file)
        file_th = int(file.split('_')[0])
        if file_th > run_th:
            run_th = file_th
    run_th += 1

    output_dir = join(opt.data_dir, 'results', f'{run_th}')
    output_dir += f'_{opt.aigc_dsc}' if opt.aigc_dsc else ''
    output_dir += f'_{opt.comment}' if opt.comment else ''

    os.makedirs(output_dir, exist_ok=True)
    np.save(join(output_dir, 'labels.npy'), label_list)
    np.save(join(output_dir, 'pred.npy'), pred_list)
    np.save(join(output_dir, 'prob.npy'), probs_list)

    cm = confusion_matrix(label_list, pred_list)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix, accuracy {acc:.4f}')
    plt.colorbar()
    plt.xticks(np.arange(len(test_class)), test_class, rotation=45)
    y_labels = []
    for i, name in enumerate(test_class):
        y_labels.append(name + f'\n{per_accuracy[i]:.4f}')
    plt.yticks(np.arange(len(test_class)), y_labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    for i in range(len(test_class)):
        for j in range(len(test_class)):
            plt.text(j, i, cm[i, j], ha='center', va='center', color='white' if cm[i, j] > cm.max() / 2 else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'result.png'))
    plt.show()

    with open(os.path.join(output_dir, 'parameters.json'), 'w') as f:
        params = vars(opt)
        params['finish time'] = time.strftime('%m-%d_%H-%M-%S')
        json.dump(vars(opt), f, indent=4)

    return acc, per_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)")

    parser.add_argument("--data_dir", type=str, default=join('datasets', 'ModelNet10'), help="dataset directory")
    parser.add_argument("--style", type=str, default='depth', help="render, depth, normal or edge")
    parser.add_argument("--bg_color", type=str, default='black', help="black or white")
    parser.add_argument("--prompt", type=str, default='@', help="prompt, @ is the placeholder for class name")
    parser.add_argument("--phi_1", type=float, default=60, help="angle for rotation")
    parser.add_argument("--phi_2", type=float, default=210, help="angle for rotation")
    parser.add_argument("--r_p", type=float, default=2.0, help="distance from camera")

    parser.add_argument("--iarm", type=bool, default=False, help="whether to use Iterative Angle Refinement Mechanism")
    parser.add_argument("--grad_step_num", type=int, default=3, help="k-step")
    parser.add_argument("--step_value", type=float, default=5, help="distance from camera")
    parser.add_argument("--step_reduce", type=float, default=1, help="distance from camera")

    parser.add_argument('--num_per_input', type=int, default=1, help='number of generated images per input')
    parser.add_argument('--resolution', type=int, default=512, help='image resolution')
    parser.add_argument('--aigc_dsc', type=str, default='', help='AIGC description')

    parser.add_argument('--comment', type=str, default='', help='comments for the run')
    opt = parser.parse_args()

    prompt_list = ['one white model of @',
                   'one line-drawn @',
                   'one photo of one @',
                   'one photo of one standalone white @',
                   'one depth map of one standalone @',
                   'one edge map of one standalone @',
                   'one render image of one standalone white @',
                   'one sketch photo of one standalone white @',
                   'one model of @ in linear composition',
                   'one photo of one @ in linear composition',
                   'one normal map of one standalone @']

    styles = ['render', 'depth', 'edge', 'normal']

    comfyui = Comfyui(workflow='controlnet')

    opt.style = styles[3]
    opt.comment = 'normal map masked'
    # opt.aigc_dsc = 'canny'
    opt.prompt = prompt_list[10]
    run(opt, comfyui=comfyui)

    for st in ['canny', 'scribble']:
        opt.aigc_dsc = st
        for p in ['one photo of one @', '@', 'one standalone @']:
            opt.prompt = p
            run(opt, comfyui=comfyui)

    for s in ['depth', 'normal']:
        opt.style = s
        opt.num_per_input = 3
        opt.prompt = 'one photo of a standalone @'
        run(opt, comfyui=comfyui)


if __name__ == "__main__":
    main()
