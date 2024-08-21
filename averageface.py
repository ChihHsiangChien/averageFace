import numpy as np
import argparse
import cv2
import re
import os
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
from delaunay2D import Delaunay2D
from matplotlib import pyplot as plt
import math


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default='./images', type=str, help="Path to input images")
    ap.add_argument('--width', default=300, type=int, help="Width of the output image")
    ap.add_argument('--height', default=300, type=int, help="Height of the output image")

    return vars(ap.parse_args())

#########
# detect face landmark
#########

def initialize_detectors():
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
    facemark = cv2.face.createFacemarkLBF()
    facemark.loadModel("lbfmodel.yaml")
    return face_cascade, facemark

def process_image(img_path, img_file,  face_cascade, facemark):

    image = cv2.imread(img_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Check if faces is empty or not
    if len(faces) == 0:
        return None

    # 待解決找到兩張臉的問題

    
    _, landmarks = facemark.fit(image, faces)

    if landmarks:
        for i, (x, y, w, h) in enumerate(faces):
            shape = np.array(landmarks[0][0], dtype=int)
            shape2 = np.array2string(shape, precision=0, separator=' ', suppress_small=True)
            shape2 = re.sub('[\[\]]', '', shape2)
            shape2 = re.sub('^\s', '', shape2)

            # 寫入座標
            with open(img_path + '.txt', 'w') as f:
                f.write(shape2 + '\n')

            # 在臉上標記            
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            draw_face(image, shape, img_file)
            '''
            for (x, y) in shape:
                cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
            '''
            '''
    if image is not None:
        # 指定加上標記的臉輸出的資料夾
        output_path = os.path.join(args["path"], "face_Landmarks")
        os.makedirs(output_path, exist_ok=True)
        cv2.imwrite(os.path.join(output_path, img_file), image)
        '''
    #return image

# 在臉上畫標記
def draw_face(image, points, img_file):
    args = parse_arguments()


    # Convert points to a numpy array
    points = np.array(points, np.float32)  # Ensure points are float32

    # Create copies of the image for different outputs
    image_with_points = image.copy()
    image_with_lines = image.copy()

    # Draw landmarks (points) on the image with points only
    for (x, y) in points:
        cv2.circle(image_with_points, (int(x), int(y)), 3, (255, 0, 0), -1)

    # Draw landmarks (points) and lines (Delaunay triangles logic) on the image with lines
    for (x, y) in points:
        cv2.circle(image_with_lines, (int(x), int(y)), 3, (255, 0, 0), -1)

    rect = (0, 0, image.shape[1], image.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(tuple(p))  # Insert as a tuple of floats

    triangle_list = subdiv.getTriangleList()

    # Draw triangles
    for t in triangle_list:
        pt1 = (int(t[0]), int(t[1]))
        pt2 = (int(t[2]), int(t[3]))
        pt3 = (int(t[4]), int(t[5]))
        cv2.line(image_with_lines, pt1, pt2, (0, 0, 255), 1)
        cv2.line(image_with_lines, pt2, pt3, (0, 0, 255), 1)
        cv2.line(image_with_lines, pt3, pt1, (0, 0, 255), 1)
    
    '''
    for (x, y) in shape:
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
    '''    
    # Save the output images
    output_path = os.path.join(args["path"], "face_Landmarks")
    os.makedirs(output_path, exist_ok=True)
    
    # 存外框
    cv2.imwrite(os.path.join(output_path, img_file), image)
    
    # 存特徵點
    cv2.imwrite(os.path.join(output_path, img_file + "_points.jpg" ), image_with_points)
    
    # 存三角框
    cv2.imwrite(os.path.join(output_path, img_file + "_lines.jpg" ), image_with_lines)
    


######
# average
######

# 讀取原圖檔與臉部特徵座標
def read_data(path):
    imgs = [s for s in os.listdir(path) if s.endswith('jpg') or s.endswith('jpeg') or s.endswith('bmp')]
    r_imgs = []
    r_cfgs = []
    for img in imgs:
        r_imgs.append(Image.open(os.path.join(path, img)))
        txt = img + '.txt'
        f = open(os.path.join(path, txt))
        poi = []
        for l in f:
            x, y = l.split()
            poi.append([int(x), int(y)])
        r_cfgs.append(np.array(poi).astype(float))
    return r_imgs, r_cfgs

def warp_points(pois, xform):
    mat, sft = xform
    ret = [mat.dot(p.T).T + sft for p in pois]
    return ret

def get_seg_xform(src, dst):
    def dot(a, b):
        return a[0] * b[0] + a[1] * b[1]
    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]
    def vec_len(vec):
        return np.sqrt(dot(vec, vec))

    v_src = src[1] - src[0]
    v_dst = dst[1] - dst[0]
    l_src = vec_len(v_src)
    l_dst = vec_len(v_dst)
    s = cross(v_dst, v_src) / l_dst / l_src
    c = dot(v_dst, v_src) / l_dst / l_src
    assert(abs(s * s + c * c - 1) < 0.0001)
    rot_mat = np.array([[c, s], [-s, c]])

    r = l_dst / l_src
    scale_mat = np.array([r, 0, 0, r]).reshape(2, 2)

    mat = scale_mat.dot(rot_mat)
    src = warp_points(src, (mat, np.zeros((2, ))))
    sft = dst[0] - src[0]
    return (mat, sft)

def get_AFFINE_data(mat, sft):
    return (mat[0][0], mat[0][1], sft[0], mat[1][0], mat[1][1], sft[1])

def get_tri_xform(src, dst):
    src = src.astype(float)
    dst = dst.astype(float)
    src = np.linalg.inv(np.concatenate((src.T, np.ones((1, 3)))))
    dst = np.concatenate((dst.T, np.ones((1, 3)))).dot(src)
    dst = dst.reshape((-1))
    return tuple(dst[:6])

def draw_points(pois, col):
    x = [p[0] for p in pois]
    y = [p[1] for p in pois]
    plt.scatter(x, y, c=col)

def draw_tri(pois, tri):
    canvas = Image.new('RGB', (600, 600))
    pois = pois.astype(int)
    draw = ImageDraw.Draw(canvas)
    for t in tri:
        draw.polygon([(pois[i][0], pois[i][1]) for i in t])
    canvas.show()

def add_tri(base, img, tri):
    mask = Image.new('RGBA', img.size, (255, 255, 255, 255))
    draw = ImageDraw.Draw(mask)
    tri = tri.astype(int)
    draw.polygon(list(map(tuple, list(tri))), fill=(255, 255, 255, 0), outline=(255, 255, 255, 0))
    return Image.composite(base, img, mask)



def findLandmarks():
    args = parse_arguments()
    face_cascade, facemark = initialize_detectors()

    path = args["path"]

    imgs = [s for s in os.listdir(path) if s.endswith('jpg') or s.endswith('jpeg') or s.endswith('bmp')]

    for img_file in imgs:
        img_path = os.path.join(path, img_file)
        process_image(img_path, img_file, face_cascade, facemark)

def average():
    args = parse_arguments()
    imgs, cfgs = read_data(args["path"])
    w, h = args["width"], args["height"]
    eye_dst = np.array([[int(0.3 * w), int(h / 3)], [int(0.7 * w), int(h / 3)]])

    bound = np.array([(0,0), (w/2,0), (w-1,0), (w-1,h/2), (w-1, h-1), (w/2, h-1), (0, h-1), (0,h/2)]).astype(float)

    tot = len(imgs)
    p_num = len(cfgs[0])
    avg_base = np.array([(0.0, 0.0)] * p_num)
    norm_imgs = []
    for i in range(tot):
        cfg = cfgs[i]
        eye_src = [cfg[36], cfg[45]]
        xform = get_seg_xform(eye_src, eye_dst)
        img = imgs[i].transform(imgs[i].size, Image.AFFINE, get_AFFINE_data(*get_seg_xform(eye_dst, eye_src)), Image.BICUBIC).crop((0, 0, w, h))
        norm_imgs.append(img)
        pois = warp_points(cfg, xform)
        avg_base += pois
        cfgs[i] = np.concatenate((pois, bound))

    avg_base /= tot
    avg_base = np.concatenate((avg_base, bound))

    dt = Delaunay2D()
    for p in avg_base:
        dt.addPoint(p)
    tris = dt.exportTriangles()
    output_path = os.path.join(args["path"], 'aligned_faces')
    os.makedirs(output_path, exist_ok=True)

    bases = []
    for i in tqdm(range(tot)):
        cfg = cfgs[i]
        norm_img = norm_imgs[i]
        base = Image.new('RGBA', norm_img.size)
        for tri in tris:
            src_tri = np.array([cfg[t] for t in tri])
            dst_tri = np.array([avg_base[t] for t in tri])
            xform = get_tri_xform(dst_tri, src_tri)
            img = norm_img.transform(norm_img.size, Image.AFFINE, xform, Image.BICUBIC)
            base = add_tri(base, img, dst_tri)
        # Convert the image to RGB mode before saving
        if base.mode == 'P':
            base = base.convert('RGB')
    
        base.save(os.path.join(output_path, f'{i}.jpg'))
        bases.append(base)


    # 所有圖像轉換成RGB
    bases = [base.convert("RGB") for base in bases]


    for i in range(1, len(bases)):
        if bases[i].mode != bases[i - 1].mode or bases[i].size != bases[i - 1].size:        
            raise ValueError("Images must have the same mode and size for blending")

        bases[i] = Image.blend(bases[i - 1], bases[i], 1.0 / (i + 1))

    final_image = bases[-1]
    final_image.save(os.path.join(output_path, 'average.jpg'))    




def create_gif_and_montage(duration=100, num_rows = 3):

    
    args = parse_arguments()
    input_dir = os.path.join(args["path"], 'aligned_faces')
    output_dir = os.path.join(args["path"], 'montageAndGif')
    os.makedirs(output_dir, exist_ok=True)

    gif_name="output.gif"
    montage_name="montage.jpg"
        
    image_files = sorted([os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith('.jpg')])

    if not image_files:
        raise ValueError("No .jpg images found in the specified folder.")

    images = [Image.open(img).convert("RGB") for img in image_files]

    gif_path = os.path.join(output_dir, gif_name)
    images[0].save(gif_path, save_all=True, append_images=images[1:-1], duration=duration, loop=0)

    # 最後一張圖是average，不用於處理
    images_for_montage = images[:-1]

    width, height = images_for_montage[0].size

    num_images = len(images_for_montage)
    num_cols = math.ceil(num_images / num_rows)

    montage_width = width * num_cols
    montage_height = height * num_rows
    montage_image = Image.new('RGB', (montage_width, montage_height))

    for i, img in enumerate(images_for_montage):
        row = i // num_cols
        col = i % num_cols
        montage_image.paste(img, (col * width, row * height))

    montage_path = os.path.join(output_dir, montage_name)
    montage_image.save(montage_path)

    print(f"GIF saved as {gif_path}")
    print(f"Montage saved as {montage_path}")

if __name__ == "__main__":
    findLandmarks()
    average()
    create_gif_and_montage()