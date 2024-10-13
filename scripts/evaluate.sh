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

# carpet
# grid
# leather
# tile
# wood
# bottle
# cable
# capsule
# hazelnut
# metal_nut
# pill
# screw
# toothbrush
# transistor
# zipper


python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name carpet \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_carpet_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda

python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name grid \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_grid_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda

python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name leather \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_leather_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda

python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name tile \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_tile_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda

python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name wood \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_wood_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda

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
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name cable \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_cable_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name capsule \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_capsule_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name hazelnut \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_hazelnut_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name metal_nut \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_metal_nut_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_pill_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name screw \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_screw_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name toothbrush \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_toothbrush_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name transistor \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_transistor_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 
 
python3 src/evaluate.py \
    --data_root data/mvtec_ad \
    --class_name zipper \
    --img_size 256 \
    --backbone_model vgg19 \
    --stmae_model stmae_base \
    --patch_size 4 \
    --num_masks 1 \
    --mask_ratio 0.5 \
    --weights_path weights/mvtec_ad_zipper_stmae_base_vgg19.pth \
    --gaussian_filter \
    --gaussian_sigma 4 \
    --gaussian_ksize 7 \
    --transform default \
    --log_interval 10 \
    --seed 42 \
    --device cuda 



