import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from load_blender import load_blender_data
from run_nerf_helpers import *
from run_nerf_adv import My_args
from run_nerf_adv import *
from PIL import Image
from numpy import asarray
from torchvision.utils import save_image
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim

# default settings
torch.cuda.set_device(6)
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor')
print('device:', device)

def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True, 
                        help='config file path')
    parser.add_argument("--expname", type=str, 
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/', 
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern', 
                        help='input data directory')

    # training options
    parser.add_argument("--netdepth", type=int, default=8, 
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256, 
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8, 
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256, 
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=32*32*4, 
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4, 
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=250, 
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=1024*32, 
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64, 
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_true', 
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true', 
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None, 
                        help='specific weights npy file to reload for coarse network')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64, 
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=0,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_true', 
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0, 
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10, 
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4, 
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0., 
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')

    parser.add_argument("--render_only", action='store_true', 
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true', 
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0, 
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops') 

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='llff', 
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=8, 
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    ## deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek', 
                        help='options : armchair / cube / greek / vase')

    ## blender flags
    parser.add_argument("--white_bkgd", action='store_true', 
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true', 
                        help='load blender synthetic data at 400x400 instead of 800x800')

    ## llff flags
    parser.add_argument("--factor", type=int, default=8, 
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true', 
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true', 
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true', 
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8, 
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100, 
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500, 
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000, 
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_testset", type=int, default=50000, 
                        help='frequency of testset saving')
    parser.add_argument("--i_video",   type=int, default=50000, 
                        help='frequency of render_poses video saving')
    
    # adversarial options
    parser.add_argument("--target_pose_num",   type=int, default=5, 
                        help='num of adversarial pose')
    parser.add_argument("--N_thresh",   type=int, default=6144, 
                        help='N_thresh')
    parser.add_argument("--Loop",   type=int, default=6, 
                    help='Loop')
    parser.add_argument("--Adv",   type=str, default='adv/exp/dog.png', 
                    help='Loop')
    parser.add_argument("--Cut",   action='store_true', 
                    help='Cut the picuters')
    return parser


# basic args and load data
# args = My_args()
parser = config_parser()
args = parser.parse_args()
args.device = device
expname = args.expname+'_'+args.Adv[8:-4]
############################################################################################################
# Load data
K = None
if args.dataset_type == 'llff':
    images, poses, bds, render_poses, i_test = load_llff_data(args.datadir, args.factor,
                                                                recenter=True, bd_factor=.75,
                                                                spherify=args.spherify)
    hwf = poses[0,:3,-1]
    poses = poses[:,:3,:4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]

    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        i_test = np.arange(images.shape[0])[::args.llffhold]

    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                    (i not in i_test and i not in i_val)])

    print('DEFINING BOUNDS')
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
        
    else:
        near = 0.
        far = 1.
    print('NEAR FAR', near, far)

elif args.dataset_type == 'blender':
    images, poses, render_poses, hwf, i_split = load_blender_data(args.datadir, args.half_res, args.testskip)
    print('Loaded blender', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split

    near = 2.
    far = 6.

    if args.white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]

elif args.dataset_type == 'LINEMOD':
    images, poses, render_poses, hwf, K, i_split, near, far = load_LINEMOD_data(args.datadir, args.half_res, args.testskip)
    print(f'Loaded LINEMOD, images shape: {images.shape}, hwf: {hwf}, K: {K}')
    print(f'[CHECK HERE] near: {near}, far: {far}.')
    i_train, i_val, i_test = i_split

    if args.white_bkgd:
        images = images[...,:3]*images[...,-1:] + (1.-images[...,-1:])
    else:
        images = images[...,:3]

elif args.dataset_type == 'deepvoxels':

    images, poses, render_poses, hwf, i_split = load_dv_data(scene=args.shape,
                                                                basedir=args.datadir,
                                                                testskip=args.testskip)

    print('Loaded deepvoxels', images.shape, render_poses.shape, hwf, args.datadir)
    i_train, i_val, i_test = i_split

    hemi_R = np.mean(np.linalg.norm(poses[:,:3,-1], axis=-1))
    near = hemi_R-1.
    far = hemi_R+1.

else:
    print('Unknown dataset type', args.dataset_type, 'exiting')
    # return



# Create log dir and copy the config file
basedir = args.basedir

os.makedirs(os.path.join('experiment', expname), exist_ok=True)
os.makedirs(os.path.join('experiment', expname, 'origin'), exist_ok=True)
# os.makedirs(os.path.join('experiment', expname, 'change'), exist_ok=True)
os.makedirs(os.path.join('experiment', expname, 'result'), exist_ok=True)
############################################################################################################
if args.Cut == True:
    ######## 需要裁剪的图片位置#########
    path_img = args.Adv
    img = Image.open(path_img)
    size_img = img.size
    print('Adv_img size:', size_img)
    x = 0
    y = 0

    ########这里需要均匀裁剪几张，就除以根号下多少，这里我需要裁剪25张-》根号25=5（5*5）####
    x_num = 5
    y_num = 3
    w = int(size_img[0] / x_num)
    h = int(size_img[1] / y_num)
    # # 注意这里是从上到下，再从左到右裁剪的
    os.makedirs(os.path.join('experiment', expname, 'clip_images'), exist_ok=True)
    for k in range(x_num):
        for v in range(y_num):
            region = img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
            #####保存图片的位置以及图片名称###############
            region.save('experiment/'+expname+'/clip_images/' + '%d%d' % (k, v) + '.png')
    print("cut finished")

# else:
#     # adv_img
#     im = asarray(Image.open(args.Adv).resize((W,H)))
#     im32 = im.astype(np.float32) / 255.
#     adv_img = torch.tensor(im32).to(device)
#     save_image(torch.permute(adv_img, (2, 0, 1)), 'experiment/'+expname+'/'+'target.png')



# Cast intrinsics to right types
H, W, focal = hwf
H, W = int(h), int(w)
hwf = [H, W, focal]

if K is None:
    K = np.array([
        [focal, 0, 0.5*W],
        [0, focal, 0.5*H],
        [0, 0, 1]
    ])


# Move testing data to GPU
poses = torch.Tensor(poses).to(device)

# show the target_pose and target_img
N_thresh = args.N_thresh
Loop = args.Loop
target_pose_num = args.target_pose_num
if args.dataset_type == 'llff':
    target_pose_num_list = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
else:
    target_pose_num_list = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80]
net = get_pretrained_net(args)
############################################################################################################
filenames = os.listdir('experiment/'+expname+'/clip_images/')
filenames.sort()
for number, file in enumerate(filenames):

    target_pose, target_img = poses[target_pose_num_list[number]], images[target_pose_num_list[number]]
    c2w=target_pose[:3,:4]
    # target image
    target_img = torch.Tensor(target_img).to(device)
    # get all encoded points
    ndc = False
    if args.dataset_type == 'llff':
        ndc = True
    rays_o_all, rays_d_all, viewdirs_all, z_vals_all, encoded_points_all = get_encoded_points(H, W, K, rays=None, c2w=c2w, near=near, far=far, ndc=ndc, use_viewdirs=args.use_viewdirs, N_samples=args.N_samples, args=args)
    # rays_o_all_copy, rays_d_all_copy, viewdirs_all_copy, z_vals_all_copy, encoded_points_all_copy = get_encoded_points(H, W, K, rays=None, c2w=c2w, near=near, far=far, use_viewdirs=args.use_viewdirs, N_samples=args.N_samples, args=args)
    sh = torch.ones((H,W,3)).shape
    with torch.no_grad():
        my_adv_img, disp, acc, extras = encoded_render_new(viewdirs_all, z_vals_all, rays_o_all, rays_d_all, encoded_points_all, net, sh, args, chunk=1024*64)
    print(torch.permute(my_adv_img, (2, 0, 1)).shape)
    save_image(torch.permute(my_adv_img, (2, 0, 1)), 'experiment/'+expname+'/'+'origin/'+str(target_pose_num_list[number])+'_origin.png')
    coords = torch.stack(torch.meshgrid(torch.linspace(0, H-1, H), torch.linspace(0, W-1, W)), -1)
    coords = torch.reshape(coords, [-1,2])  # (H * W, 2)

    im = asarray(Image.open('experiment/'+expname+'/clip_images/'+file))
    im32 = im.astype(np.float32) / 255.
    adv_img = torch.tensor(im32).to(device)
    # save_image(torch.permute(adv_img, (2, 0, 1)), 'experiment/'+expname+'/'+'target.png')

    # for i in tqdm(range(0, W*H, N_thresh)):
    for num, i in enumerate(tqdm(range(0, W*H, N_thresh))):
    # for num, i in enumerate(range(0, W*H, N_thresh)):
        # loop through all pixels 每个像素是肯定要循环的，不然不更新了。
        rays_o, rays_d = get_rays(H, W, K, c2w)
        if N_thresh+i-1 > W*H-1:
            select_inds = torch.linspace(i, W*H-1, N_thresh).long()  # (N_rand,)
        else:
            select_inds = torch.linspace(i, N_thresh+i-1, N_thresh).long()  # (N_rand,)
        select_coords = coords[select_inds].long()  # (N_rand, 2)
        rays_o = rays_o[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        rays_d = rays_d[select_coords[:, 0], select_coords[:, 1]]  # (N_rand, 3)
        viewdirs = viewdirs_all.reshape((H,W,-1))[select_coords[:, 0], select_coords[:, 1]] # (N_rand, 3)
        z_vals = z_vals_all.reshape((H,W,-1))[select_coords[:, 0], select_coords[:, 1]] # (N_rand, 64)
        
        # target and adv img
        # target_img_s = target_img[select_coords[:, 1], select_coords[:, 0]]  # (N_rand, 3)
        adv_img_s = adv_img[select_coords[:, 0], select_coords[:, 1]] # (N_rand, 3)
        
        # encoded_points
        encoded_points = encoded_points_all.reshape((H,W,64,90))[select_coords[:, 0], select_coords[:, 1]].reshape((-1,64,90))
        encoded_points.requires_grad = True

        # adversarial attacking
        lr = 1e-2
        weight_decay=8e-1
        decay_rate = 0.3
        adv_optimizer = torch.optim.AdamW([encoded_points], lr=lr, weight_decay=weight_decay)
        # adv_optimizer = torch.optim.Adam([encoded_points], lr=lr)
        # adv_optimizer = torch.optim.SGD([encoded_points], lr=8e-1, momentum=0.1)
        adv_points = encoded_points.clone()
        global_step = 1
        loss_list = []
        for j in range(Loop): # 主要在这里加trick，以梯度值作为指标。

            if j < 50:
                lr = 1e-1
            else:
                lr = 1e-2
            adv_optimizer.param_groups[0]['lr'] = lr
            
            sh = rays_d.shape
            rgb, disp, acc, extras = encoded_render_new(viewdirs, z_vals, rays_o, rays_d, adv_points, net, sh, chunk=1024*64, args=args)

            loss = torch.nn.functional.mse_loss(rgb, adv_img_s, reduction='sum')
            loss_list.append(loss.item())
            # loss = torch.nn.functional.smooth_l1_loss(rgb, adv_img_s)
            # print('img_loss', loss)
            loss.backward()
            adv_optimizer.step()
            adv_optimizer.zero_grad()
            adv_points = adv_optimizer.param_groups[0]['params'][0]

            # ###   update learning rate and weight_decay  ###
            new_lrate = lr * (decay_rate ** (global_step / Loop))
            for param_group in adv_optimizer.param_groups:
                param_group['lr'] = new_lrate

            new_weight_decay = weight_decay * (decay_rate ** (global_step / Loop))
            for param_group in adv_optimizer.param_groups:
                param_group['weight_decay'] = new_weight_decay


            global_step = global_step + 1
            ###################################################

        # file = open("experiment/"+expname+"/record/"+str(num+1)+".txt", "w")
        # for cont, item in enumerate(loss_list):
        #     file.write(str(cont+1) + "\t" + str(item) + "\n")
        # file.close()
        print(num+1, 'img_loss', loss)
        encoded_points.requires_grad = False
        encoded_points_all.reshape((H,W,64,90))[select_coords[:, 0], select_coords[:, 1]] = adv_points

        # if (num+1)%N_thresh == 0:
        # sh = torch.ones((H,W,3)).shape
        # with torch.no_grad():
        #     my_adv_img, disp, acc, extras = encoded_render_new(viewdirs_all, z_vals_all, rays_o_all, rays_d_all, encoded_points_all, net, sh, args=args, chunk=1024*64)
        # save_image(torch.permute(my_adv_img, (2, 0, 1)),'experiment/'+expname+'/'+'change/'+str(num+1)+'.png')


    # noise = encoded_points_all - encoded_points_all_copy
    # torch.save(noise, 'experiment/'+expname+'/'+expname+'.pt')

    sh = torch.ones((H,W,3)).shape
    with torch.no_grad():
        my_adv_img, disp, acc, extras = encoded_render_new(viewdirs_all, z_vals_all, rays_o_all, rays_d_all, encoded_points_all, net, sh, args=args, chunk=1024*64)
    save_image(torch.permute(my_adv_img, (2, 0, 1)),'experiment/'+expname+'/'+'result/'+ file)

    sfile = open("experiment/"+expname+'/'+"ssim.txt", "a")
    sfile.write(file + '\t' + str(ssim(my_adv_img.cpu().numpy(), adv_img.cpu().numpy(), multichannel=True)) +'\t'+ str(loss.item()) + '\n')
    sfile.close()

    print("ssim =", ssim(my_adv_img.cpu().numpy(), adv_img.cpu().numpy(), multichannel=True))

############################################################################################################

# 输入图像的路径
path = 'experiment/'+expname+'/'+'result'
filenames = os.listdir(path)
filenames.sort()
list_a = []

for i, filename in enumerate(filenames):
    num_yx = 3
    i += 1
    t = (i-1)//num_yx
    # 获取img
    im = Image.open(os.path.join(path, filename))
    # 转换为numpy数组
    im_array = np.array(im)

    # 如果取的图像输入下一列的第一个，因为每列是3张图像，所以1，4，7等就是每列的第一张
    if (i-1) % num_yx == 0:
        # list_a[t] = im_array
        list_a.append(im_array)

    # 否则不是第一个数，就拼接到图像的下面
    else:
        list_a[t] = np.concatenate((list_a[t], im_array), axis=0)

# 2 合成列以后需要将列都拼接起来
for j in range(len(list_a)-1):
    list_a[0] = np.concatenate((list_a[0], list_a[j+1]),axis=1)

im_save = Image.fromarray(np.uint8(list_a[0]))
im_save.save('experiment/'+expname+'/'+expname+'.png')