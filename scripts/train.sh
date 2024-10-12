    # parser.add_argument('--data_root', type=str, required=True, help='Path to dataset')
    # parser.add_argument('--class_name', type=str, required=True, help='Class name of the dataset')
    # parser.add_argument('--stmae_model', type=str, default='stmae_base', help='ST-MAE model')
    # parser.add_argument('--backbone_model', type=str, default='vgg19', help='Backbone model')
    # parser.add_argument('--img_size', type=int, default=256, help='Image size')
    # parser.add_argument('--split', type=str, default='train', help='Data split')
    # parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    # parser.add_argument('--patch_size', type=int, default=4, help='Patch size')
    
    # parser.add_argument('--mask_ratio', type=float, default=0.5, help='Mask ratio')
    # parser.add_argument('--transform', type=str, default='default', help='Transform type')
    # parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs')
    # parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')    
    # parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    # parser.add_argument('--weight_decay', type=float, default=0.05, help='Weight decay')
    # parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer')
    # parser.add_argument('--scheduler', type=str, default=None, help='Scheduler')

    # parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # parser.add_argument('--num_workers', type=int, default=1, help='Number of workers')
    # parser.add_argument('--device', type=str, default='cuda', help='Device')
    
    # parser.add_argument('--log_interval', type=int, default=1, help='Log interval')
    # parser.add_argument('--save_dir', type=str, default='weights', help='Save directory')
    # parser.add_argument('--resume_path', type=str, default=None, help='Path to resume weights')

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

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name carpet \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name grid \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name leather \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name tile \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name wood \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name bottle \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name cable \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name capsule \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name hazelnut \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name metal_nut \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name pill \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name screw \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name toothbrush \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name transistor \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

python3 src/train.py \
    --data_root data/mvtec_ad \
    --class_name zipper \
    --stmae_model stmae_base \
    --backbone_model vgg19 \
    --img_size 256 \
    --split train \
    --batch_size 8 \
    --patch_size 4 \
    --mask_ratio 0.5 \
    --transform default \
    --num_epochs 400 \
    --lr 0.0001 \
    --momentum 0.9 \
    --weight_decay 0.05 \
    --optimizer adamw \
    --scheduler None \
    --seed 42 \
    --num_workers 1 \
    --device cuda \
    --log_interval 10 \
    --save_dir weights \

