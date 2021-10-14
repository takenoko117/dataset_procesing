import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
import cv2
from tqdm import tqdm
import os
import glob

def transform_depth(d_im):
    #From Aly's code
    #print(d_im.shape)
    zy, zx = np.gradient(d_im)
    # Sobel to get a joint Gaussian smoothing and differentation
    # to reduce noise
    normal = np.dstack((-zx, -zy, np.ones_like(d_im)))
    n = np.linalg.norm(normal, axis=2)
    normal[:, :, 0] /= n
    normal[:, :, 1] /= n
    normal[:, :, 2] /= n
    # offset and rescale values to be in 0-255
    normal += 1
    normal /= 2
    normal *= 255
    im_bgr = normal
    im_bgr = cv2.bilateralFilter(im_bgr.astype('uint8'), 9, 75, 75)
    gaussian_3 = cv2.GaussianBlur(im_bgr, (9,9), 10.0)
    unsharp_image = cv2.addWeighted(im_bgr, 1.5, gaussian_3, -0.5, 0, im_bgr)

    return unsharp_image

target_dataset_name = "./handcamv10_row/" #前処理する前の出来立てデータセットのファイルパス
outputdir = "./handcamv10_test_dataset/"
os.makedirs(outputdir, exist_ok=True)
print("outputdir = " + outputdir)

zfill_number = 5 #データセットのファイルの連番の桁数（0001.jpgなら4，00001なら5）
total_index_y = 1
total_index_meta = 1
total_index_rgb = 1
total_index_depth = 1
total_index_hand = 1
total_index_train_split = 1

files = sorted(glob.glob(target_dataset_name + "**"))
for file in tqdm(files):
    #加工後の保存ディレクトリ
    if total_index_train_split % 10 == 1:
        train_split = "test/"
    elif total_index_train_split % 10 == 2:
        train_split = "valid/"
    else:
        train_split = "train/"
    #testのみ作る
    train_split = "test/"
    total_index_train_split += 1
    dest_file = outputdir + train_split + file.split('/')[-1] + '_processed/'
    os.makedirs(dest_file, exist_ok=True)
    
    #y-label data
    y_file = sorted(glob.glob(file+"/y*"))
    y_file = y_file[0]
    y_file = open(y_file, 'r')
    y_dataset = np.array([])
    for line in y_file:
        y_dataset = np.array([])
        y = line.split(":")
        for i in range (1, 5):
            y_dataset = np.append(y_dataset, 0 if int(y[i]) == 0 else 1)
        #Score column 100 means no collision then 0 is saved
        y_dataset = np.append(y_dataset, 0 if int(y[0]) == 100 else 1)
        save_name = dest_file + str(total_index_y).zfill(zfill_number) + "_y.csv"
        np.savetxt(save_name, y_dataset.reshape(-1, 5), delimiter=',', fmt = "%d")
        total_index_y = total_index_y + 1
    y_file.close()
    
    ###---meta-data---###
    meta_file = sorted(glob.glob(file+"/meta*"))
    meta_file = meta_file[0]
    meta_file = open(meta_file, 'r')
    target_sizes_filepath = "./target_sizes.txt"#アイテムの大きさXYZが並んでるものを読み込む
    #This file must be on same directory as collected data from unity
    sizes = np.loadtxt(target_sizes_filepath)
    meta_sizes = sizes[:,1:4]
    scaler = StandardScaler()
    st_meta = scaler.fit_transform(meta_sizes)
    sizes = np.hstack((sizes[:,0:1], st_meta))
    meta = np.array([])
    for line in meta_file:
        m=line.split(':')
        meta = np.array([])
        #Upper --> 1, Lower --> -1
        metadata = 1 if m[0][0] == "U" else -1
        
        #未知アイテム使用時はコメントアウト        #x = y = z = 1
        target = int(m[1].split('_')[0])
        x =sizes[target-1][1]
        y =sizes[target-1][2]
        z =sizes[target-1][3]
        
        meta = np.array([metadata, x, y, z])

        meta = meta.reshape(-1,4)
        save_name = dest_file + str(total_index_meta).zfill(zfill_number) + "_meta.csv"
        np.savetxt(save_name, meta, delimiter=',', fmt = "%d, %f, %f, %f")
        target_rotation = m[7].split(",") #target_rotation (0,90.0,-90.0)のことだけ
        target_rotation_to_class_dict = {" 90.0)\n":0, " 0.0)\n":1, " -90.0)\n":2, " -180.0)\n":3}#x,y,zのだけみてクラス分けする辞書
        target_rotation_class= np.array([])
        target_rotation_class = np.append(target_rotation_class, target_rotation_to_class_dict[target_rotation[2]])#x,y,zのzのとこ([2])を辞書にいれてクラスわけ
        save_name = dest_file + str(total_index_meta).zfill(zfill_number) + "_target_rotation.csv"
        np.savetxt(save_name, target_rotation_class, fmt = "%d")
        #姿勢によってサイズxyzを入れ替える
        if target_rotation_to_class_dict[target_rotation[2]] == 1:
            #xとyを逆にする
            rotation_meta = np.array([metadata, y, x, z])
        else:.
            rotation_meta = np.array([metadata, x, y, z])

        rotation_meta = rotation_meta.reshape(-1,4)
        
        save_name = dest_file + str(total_index_meta).zfill(zfill_number) + "_rotation-meta.csv"
        np.savetxt(save_name, rotation_meta, delimiter=',', fmt = "%d, %f, %f, %f")
        total_index_meta = total_index_meta + 1
    meta_file.close()

        ############## RGB processing ###############    
    rgb_files = sorted(glob.glob(file+"/*_rgb.jpg"))
    for rgb_file in tqdm(rgb_files):
        rgb = cv2.imread(rgb_file)

        cropped = rgb[242:434, 152:344] #クロップ

        save_filepath = dest_file + str(total_index_rgb).zfill(zfill_number) + "_rgb.jpg"
        cv2.imwrite(save_filepath, cropped)
        total_index_rgb += 1

        ############## Handcamera processing ###############
    hand_files = sorted(glob.glob(file+"/*_hand.jpg"))
    for hand_file in tqdm(hand_files):
        target_image = cv2.imread(hand_file)

        target_image = target_image[0:480, 0:480]#正方形にカット，上部分を削った
        target_image  = cv2.resize(target_image, (224, 224)) #224*224になるようにリサイズ

        save_filepath = dest_file + str(total_index_hand).zfill(zfill_number) + "_hand.jpg" #クロップした画像を保存するパス
        cv2.imwrite(save_filepath, target_image)
        total_index_hand += 1

        ############### PROCESS DEPTH ############################
        ############## Generate grid  for Depth images ###############
    depth_files = sorted(glob.glob(file+"/*_depth.csv"))
    for depth_file in tqdm(depth_files):
        depth_file = np.loadtxt(depth_file)

        depth_patch = depth_file[242:434, 152:344]
        depth_image = transform_depth(depth_patch)

        save_filepath = dest_file + str(total_index_depth).zfill(zfill_number) + "_depth.jpg" #クロップした画像を保存するパス
        cv2.imwrite(save_filepath, depth_image)
        total_index_depth += 1

#ブレエエ
#ブレエエブレエエ
#ブレエエブレエエブレエエブレエエブレエエブ
#ブレエエブレエエブレエエブレエエブレエエブブレエエブレエエブレエエブレエエブレエエブ
