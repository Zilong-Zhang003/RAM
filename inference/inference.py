import argparse
import cv2
import glob
import numpy as np
import os
import torch
from torchvision.transforms.functional import normalize
from ram.archs.promptir_arch import PromptIR
from ram.archs.swinir_arch import SwinIR
from ram.archs.restormerRFR_arch import RestormerRFR
from ram.utils.dino_feature_extractor import DinoFeatureModule
import os.path as osp


def define_model(args):
    if args.model == "ram_promptir":
        model = PromptIR(decoder=True)
    elif args.model == 'ram_swinir':
        model = SwinIR(
            patch_size=1,
            in_chans=3,
            embed_dim=180,
            depths=[6, 6, 6, 6, 6, 6],
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            window_size=8,
            finetune_type=None,
            upscale=1
        )
    elif args.model == "RestormerRFR":
        model = RestormerRFR(
            inp_channels=3,
            out_channels=3,
            dim=48,
            num_blocks=[4, 6, 6, 8],
            num_refinement_blocks=4,
            heads=[1, 2, 4, 8],
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='WithBias',  
            finetune_type=None,
            img_size=128
        )
    else:
        raise NotImplementedError(f"Unknown model: {args.model}")

    
    loadnet = torch.load(args.model_path, map_location='cpu')
    keyname = 'params'
    state = loadnet[keyname] if isinstance(loadnet, dict) and keyname in loadnet else loadnet
    model.load_state_dict(state, strict=False)
    return model


def process_image(img_path, model, device, args, dino_extractor=None):
    imgname = osp.splitext(osp.basename(img_path))[0]
    print('processing image: ', imgname)

    img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
    img = img.unsqueeze(0).to(device)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalize(img, mean, std, inplace=True)

    with torch.no_grad():
        if dino_extractor is not None:
            dino_features = dino_extractor(img)
            output = model(img, dino_features)
        else:
            output = model(img)

    output = normalize(output, -1 * mean / std, 1 / std)
    output = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
    output = (output * 255.0).round().astype(np.uint8)

    os.makedirs(args.output, exist_ok=True)
    cv2.imwrite(osp.join(args.output, f'{imgname}_{args.model}.png'), output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='input image or folder')
    parser.add_argument('--output', type=str, default='outputs/', help='output folder')
    parser.add_argument('--model', type=str, default='ram_promptir',
                        choices=['ram_promptir', 'ram_swinir', 'RestormerRFR'], help='model type')
    parser.add_argument('--model_path', type=str, default='pretrained_model/ram_promptir_finetune.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = define_model(args)
    model.eval()
    model = model.to(device)

    dino_extractor = None
    if args.model == 'RestormerRFR':
        dino_extractor = DinoFeatureModule().to(device).eval()

    if osp.isdir(args.input):
        for path in sorted(glob.glob(osp.join(args.input, '*'))):
            if path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                process_image(path, model, device, args, dino_extractor)
    elif osp.isfile(args.input) and args.input.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
        process_image(args.input, model, device, args, dino_extractor)
    else:
        print('invalid img format')


if __name__ == "__main__":
    main()
