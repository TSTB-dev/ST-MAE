    # parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    # parser.add_argument('--class_name', type=str, required=True, help='Class name of the dataset')
    # parser.add_argument('--img_size', type=int, default=256, help='Image size')
    # parser.add_argument('--backbone_model', type=str, default='vgg19', help='Backbone model')
    # parser.add_argument('--stmae_model', type=str, default='stmae_base', help='ST-MAE model')
    # parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    # parser.add_argument('--num_masks', type=int, default=1, help='Number of masks')
    # parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask ratio')
    # parser.add_argument('--weights_path', type=str, default=None, help='Path to weights')
    # parser.add_argument('--gaussian_filter', action='store_true', default=False, help='Apply Gaussian filter')
    # parser.add_argument('--gaussian_sigma', type=float, default=4, help='Gaussian sigma')
    # parser.add_argument('--gaussian_ksize', type=int, default=7, help='Gaussian kernel size')
    # parser.add_argument('--transform', type=str, default='default', help='Transform type')
    # parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    # parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # parser.add_argument('--device', type=str, default='cuda', help='Device')

python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name bottle \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_bottle_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda

