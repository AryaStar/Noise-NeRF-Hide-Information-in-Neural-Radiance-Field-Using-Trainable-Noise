conda activate advnerf

# # exp1
# python run_adv_nerf.py --config experiment/configs/lego.txt
# python run_adv_nerf.py --config experiment/configs/lego.txt --target_pose_num 15 --Adv 'adv/exp/macarom.png'
# python run_adv_nerf.py --config experiment/configs/lego.txt --target_pose_num 10 --Adv 'adv/exp/monkey.png'

# python run_adv_nerf.py --config experiment/configs/chair.txt
# python run_adv_nerf.py --config experiment/configs/chair.txt --target_pose_num 15 --Adv 'adv/exp/macarom.png'
# python run_adv_nerf.py --config experiment/configs/chair.txt --target_pose_num 10 --Adv 'adv/exp/monkey.png'

# python run_adv_nerf.py --config experiment/configs/fern.txt
# python run_adv_nerf.py --config experiment/configs/fern.txt --target_pose_num 15 --Adv 'adv/exp/macarom.png'
# python run_adv_nerf.py --config experiment/configs/fern.txt --target_pose_num 10 --Adv 'adv/exp/monkey.png'

# # exp2
# python run_adv_cut.py --config experiment/configs/lego.txt --Adv 'adv/exp/HD.png' --Cut

# python run_adv_cut.py --config experiment/configs/fern.txt --Adv 'adv/exp/HD.png' --Cut

# python run_adv_cut.py --config experiment/configs/chair.txt --Adv 'adv/exp/HD.png' --Cut


# python run_adv_cut.py --config experiment/configs/lego.txt --Adv 'adv/exp/church.png' --Cut
# python run_adv_cut.py --config experiment/configs/fern.txt --Adv 'adv/exp/church.png' --Cut
# python run_adv_cut.py --config experiment/configs/chair.txt --Adv 'adv/exp/church.png' --Cut


python run_adv_cut.py --config experiment/configs/lego.txt --Adv 'adv/exp/HD.png' --Cut
python run_adv_cut.py --config experiment/configs/chair.txt --Adv 'adv/exp/HD.png' --Cut
python run_adv_cut.py --config experiment/configs/fern.txt --Adv 'adv/exp/HD.png' --Cut