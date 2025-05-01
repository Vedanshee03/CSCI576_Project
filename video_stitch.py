import cv2
import numpy as np
from tqdm import tqdm
import argparse
import multiprocessing as mp 
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time 

class VideoPanoramaStitcher:
    def __init__(self, frame_step=5, downscale=0.5):
        self.frame_step = frame_step
        self.downscale  = downscale

        #ORB for keypoints/descriptors 
        self.detector_orb  = cv2.ORB_create(20000)
        # for keypoints/descriptor
        self.detector_sift  = cv2.SIFT_create(20000)

        self.matcher_sift   = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        self.matcher_orb    = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)


    def sample_frames(self, path):
        cap = cv2.VideoCapture(path)
        #frame_step = 5 #Dynamically adjust based on video length 
        frames, idx = [], 0

        while True:
            ret, f = cap.read()
            if not ret:
                break
            if idx % self.frame_step == 0:
                # downsample 4K→2K
                f = cv2.resize(f, None, fx=self.downscale, fy=self.downscale,
                               interpolation=cv2.INTER_AREA)
                frames.append(f)
            idx += 1
        cap.release()
        return frames

    def detect_and_describe_orb(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector_orb.detectAndCompute(gray, None)

    def detect_and_describe_sift(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return self.detector_sift.detectAndCompute(gray, None)

    def match_keypoints_orb(self, A, B, ratio=0.75):
        raw = self.matcher_orb.knnMatch(A, B, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair 
            if m.distance < ratio * n.distance:
                good.append(m)
        return good 
        #return [m for m,n in raw if m.distance < ratio*n.distance]

    def match_keypoints_sift(self, A, B, ratio=0.75):
        raw = self.matcher_sift.knnMatch(A, B, k=2)
        good = []
        for pair in raw:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio * n.distance:
                good.append(m)
        return good 
        #return [m for m,n in raw if m.distance < ratio*n.distance]

    def estimate_homography(self, kpA, kpB, matches):
        if len(matches) < 4:
            return None
        ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])
        H, _ = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, 4.0)
        return H



    def stitch(self, videos):
        # 1) sample & downscale
        frames = []
        for v in videos:
            frames += self.sample_frames(v)
        if not frames:
            raise RuntimeError("No frames extracted!")

        # 2) init pano
        pano = frames[0].copy()
        kp_p_orb, des_p_orb = self.detect_and_describe_orb(pano)
        kp_p_sift, des_p_sift = self.detect_and_describe_sift(pano)

        # 3) iterate
        for img in tqdm(frames[1:], desc="Stitching"):
            kp_i_orb, des_i_orb = self.detect_and_describe_orb(img)
            kp_i_sift, des_i_sift = self.detect_and_describe_sift(img)
            
            # match ORB
            if des_i_orb is None or des_p_orb is None:
                matches_orb = []
            else:
                matches_orb = self.match_keypoints_orb(des_i_orb, des_p_orb)
            
            # match SIFT
            if des_i_sift is None or des_p_sift is None:
                matches_sift = []
            else:
                matches_sift = self.match_keypoints_sift(des_i_sift, des_p_sift)
            
            # Combine keypoints from ORB and SIFT
            kp_i_combined = kp_i_orb + kp_i_sift
            kp_p_combined = kp_p_orb + kp_p_sift

            # Combine matches from ORB and SIFT
            matches_combined = matches_orb + matches_sift
            if len(matches_combined) < 4:
                print("Not enough matches found.")
                continue

            # Estimate homography using the combined keypoints and matches
            H = self.estimate_homography(kp_i_combined, kp_p_combined, matches_combined)
            if H is None:
                continue

            # compute canvas
            h_p, w_p = pano.shape[:2]
            h_i, w_i = img.shape[:2]
            corners = np.float32([[0,0],[0,h_i],[w_i,h_i],[w_i,0]]).reshape(-1,1,2)
            warped_c = cv2.perspectiveTransform(corners, H)
            all_c = np.vstack((np.float32([[0,0],[0,h_p],[w_p,h_p],[w_p,0]]).reshape(-1,1,2),
                               warped_c))
            x0,y0 = np.int32(all_c.min(axis=0).ravel() - 0.5)
            x1,y1 = np.int32(all_c.max(axis=0).ravel() + 0.5)
            trans = [-x0, -y0]
            Ht = np.array([[1,0,trans[0]],[0,1,trans[1]],[0,0,1]], dtype=np.float64)
            canvas = (x1-x0, y1-y0)

            # warp both into canvas
            pano_w = cv2.warpPerspective(pano, Ht, canvas)
            img_w  = cv2.warpPerspective(img,  Ht.dot(H), canvas)

            # Feather blend 
            mask_p = (pano_w[:,:,0] > 0).astype(np.uint8)
            mask_i = (img_w[:,:,0]  > 0).astype(np.uint8)

            # kernel = np.ones((5,5), np.uint8)
            # mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_CLOSE, kernel)
            # mask_i = cv2.morpologyEx(mask_i, cv2.MORPH_CLOSE, kernel)

            # mask_p = (pano_w.sum(axis=2) > 0).astype(np.uint8)
            # mask_i = (img_w.sum(axis=2) > 0).astype(np.uint8)
            
            # kernel = np.ones((5,5), np.uint8)
            # mask_p = cv2.morphologyEx(mask_p, cv2.MORPH_CLOSE, kernel)
            # mask_i = cv2.morphologyEx(mask_i, cv2.MORPH_CLOSE, kernel)

            # distance transform on each mask
            dp = cv2.distanceTransform(mask_p, cv2.DIST_L2, 5).astype(np.float32)
            di = cv2.distanceTransform(mask_i, cv2.DIST_L2, 5).astype(np.float32)
            wsum = dp + di + 1e-6   # avoid division by zero
            
            # normalized weights
            wp = (dp / wsum)[:, :, None]
            wi = (di / wsum)[:, :, None]

            # blend and convert back to uint8
            pano = (pano_w.astype(np.float32)*wp + img_w.astype(np.float32)*wi).astype(np.uint8)

            # update features
            kp_p_orb, des_p_orb = self.detect_and_describe_orb(pano)
            kp_p_sift, des_p_sift = self.detect_and_describe_sift(pano)

        # 4) final crop
        gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
        _, th = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        ys, xs = np.where(th>0)
        pano = pano[ys.min():ys.max()+1, xs.min():xs.max()+1]

        return pano

def main():
    p = argparse.ArgumentParser()
    p.add_argument("videos", nargs="+")
    p.add_argument("--step",   type=int,   default=5)
    p.add_argument("--down",   type=float, default=0.5,
                   help="downscale factor (0<down<=1)")
    p.add_argument("--output", default="panorama.jpg")
    args = p.parse_args()

    stitcher = VideoPanoramaStitcher(frame_step=args.step,
                                     downscale=args.down)
    pano = stitcher.stitch(args.videos)
    cv2.imwrite(args.output, pano)
    print("Saved:", args.output)

if __name__ == "__main__":
    main()
