import cv2 # for video I/O, feature detection, homography, warping 
import numpy as np 
from tqdm import tqdm # for progress bars 

class VideoPanoramaStitcher:
    def __init__(self, frame_step=30, blend_alpha=0.5):
        """
        frame_step: take every Nth frame from the video
        blend_alpha: weight for blending old panorama vs new frame 
        """
        #ORB detector + descriptor 
        self.detector = cv2.ORB_create(5000)
        #BFMatcher for ORB 
        #Brute force matcher with Hamming distance 
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.frame_step = frame_step
        self.blend_alpha = blend_alpha

    def sample_frames(self, video_path):
        """Read video and return a list of sampled frames"""
        cap = cv2.VideoCapture(video_path) # open video file 
        frames = []
        idx = 0
        while True:
            ret, frame = cap.read() # grab next frame 
            if not ret:
                break #end of video 
            if idx % self.frame_step == 0:
                frames.append(frame) #keep this frame 
            idx += 1
        cap.release() # free video resource 
        return frames 
    
    def detect_and_describe(self, image):
        """Convert to grayscale, find keypoints, and compute descriptors"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kp, des = self.detector.detectAndCompute(gray, None)
        return kp, des
    
    def match_keypoints(self, desA, desB, ratio=0.75):
        """
        Perform KNN matching between descriptor sets A and B, 
        then apply Lowe's ratio test to keep good matches.
        """
        raw_matches = self.matcher.knnMatch(desA, desB, k=2)
        good = []
        for m, n in raw_matches:
            if m.distance < ratio * n.distance:
                good.append(m)
        return good 
    
    def estimate_homography(self, kpA, kpB, matches, reproj_thresh=4.0):
        """
        Given matched keypoints, compute the homography H (3X3 warp matrix)
        mapping points from image A to image B using RANSAC
        """
        if len(matches) < 4:
            return None, None 
        
        #Extract matched point coordinates 
        ptsA = np.float32([kpA[m.queryIdx].pt for m in matches])
        ptsB = np.float32([kpB[m.trainIdx].pt for m in matches])

        #Compute H with RANSAC to reject outliers 
        H, status = cv2.findHomography(ptsA, ptsB, cv2.RANSAC, reproj_thresh)
        return H, status 
    
    def stitch(self, images):
        """
        Take a list of images and stitch them into one panorama:
        -initialize with the first image 
        -for each next image: match, compute H, warp, blend, update panorama
        """

        #Start panorama with the very first frame 
        pano = images[0].copy()
        #Detect features in the current panorama 
        kp_pano, des_pano = self.detect_and_describe(pano)

        #Loop over the remaning frames with a progress bar 
        for img in tqdm(images[1:], desc="Stitching frames"):
            #1) detect features in the new frame 
            kp, des = self.detect_and_describe(img)
            #2) match descriptors (new frame - current panorama)
            matches = self.match_keypoints(des, des_pano)
            #3) estimate warp from new frame to panorama 
            H, status = self.estimate_homography(kp, kp_pano, matches)
            if H is None:
                continue #not good enough good matches-slip this frame 

            #4) determine size of the output canvas 
            # warp corners of both panorama and new frame then take the bounding box 
            h_p, w_p = pano.shape[:2]
            h_i, w_i = img.shape[:2]
            corners_img = np.float32([[0,0], [0,h_i], [w_i, h_i], [w_i, 0]]).reshape(-1, 1, 2)
            warped_corners = cv2.perspectiveTransform(corners_img, H)
            all_corners = np.vstack((
                np.float32([[0,0], [0, h_p], [w_p, h_p], [w_p, 0]]).reshape(-1,1,2),
                warped_corners
            ))

            #Compute the translation needed to keep coordinates non-negative 
            x_min, y_min = np.int32(all_corners.min(axis=0).ravel() - 0.5)
            x_max, y_max = np.int32(all_corners.max(axis=0).ravel() + 0.5)
            translation = [-x_min, -y_min]

            #translation matrix to shift everything into positive quadrant 
            H_trans = np.array([
                [1.0, 0.0, translation[0]],
                [0.0, 1.0, translation[1]],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)

            #final canvas size 
            canvas_size = (x_max - x_min, y_max - y_min)

            #5) Warp both the panorama and the new image 
            pano_warped = cv2.warpPerspective(pano, H_trans, canvas_size)
            img_warped = cv2.warpPerspective(img, H_trans.dot(H), canvas_size)

            #6) create masks where each warped image has valid pixels 
            mask_pano = (pano_warped > 0).astype(np.uint8)
            mask_img = (img_warped > 0).astype(np.uint8)

            #7) blend the overlapping region using the given alpha 
            blended = (
                pano_warped.astype(float) * self.blend_alpha +
                img_warped.astype(float) * (1 - self.blend_alpha)
            ).astype(np.uint8)

            #8) assemble final panorama: where only one image exist take it, otherwise use blend 
            pano = np.where(mask_img==0, pano_warped,
                            np.where(mask_pano==0, img_warped, blended))
            
            #9) update keypoints/descriptors on the new panorama 
            kp_pano, des_pano = self.detect_and_describe(pano)

        return pano 
if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser(
            description="Build an ultra-high resolution mosaic from video frames"
        )

    #CLI arguments for input videos, sampling step, blend weight, and output path 
    parser.add_argument("videos", nargs="+", help="paths to input video files")
    parser.add_argument("--step", type=int, default=30,
                        help="sample every Nth frame from each video")
    parser.add_argument("--alpha", type=float, default=0.5,
                        help="blend weight: higher α favors existing panorama")
    parser.add_argument("--output", default="mosaic.jpg",
                        help="filename for the saved panorama")
    args = parser.parse_args()

     # Collect all video paths:
    #video_paths = list(args.videos)
    #if args.folder:
        # e.g. pick up .mp4 and .avi files in the folder
     #   video_paths += glob.glob(os.path.join(args.folder, "*.mp4"))
      #  video_paths += glob.glob(os.path.join(args.folder, "*.avi"))

    #if not video_paths:
     #   raise ValueError("No videos provided—pass files or use --folder")

    video_paths = args.videos
    #Initialize our stitcher with the user's parameters 
    stitcher = VideoPanoramaStitcher(frame_step=args.step, blend_alpha=args.alpha)

    #1) Sample frames from each provided video
    all_frames = []
    for vid in video_paths:
        frames = stitcher.sample_frames(vid)
        print(f"Sampled {len(frames)} frames from {vid}")
        all_frames.extend(frames)
    #2) Run the stitcher over all collected frames 
    panorama = stitcher.stitch(all_frames)

    #3) Save the final ultra high resolution image 
    cv2.imwrite(args.output, panorama)
    print(f"Saved panorama to {args.output}")

