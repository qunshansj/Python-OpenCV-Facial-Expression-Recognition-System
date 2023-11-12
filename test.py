python


class FaceMask:
    def __init__(self, predictor_path, mask_path):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    def add_alpha_channel(self, img):
        b_channel, g_channel, r_channel = cv2.split(img)
        alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 255
        img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
        return img_new

    def merge_img(self, jpg_img, png_img, y1, y2, x1, x2):
        if jpg_img.shape[2] == 3:
            jpg_img = self.add_alpha_channel(jpg_img)

        yy1 = 0
        yy2 = png_img.shape[0]
        xx1 = 0
        xx2 = png_img.shape[1]

        if x1 < 0:
            xx1 = -x1
            x1 = 0
        if y1 < 0:
            yy1 = -y1
            y1 = 0
        if x2 > jpg_img.shape[1]:
            xx2 = png_img.shape[1] - (x2 - jpg_img.shape[1])
            x2 = jpg_img.shape[1]
        if y2 > jpg_img.shape[0]:
            yy2 = png_img.shape[0] - (y2 - jpg_img.shape[0])
            y2 = jpg_img.shape[0]

        alpha_png = png_img[yy1:yy2, xx1:xx2, 3] / 255.0
        alpha_jpg = 1 - alpha_png

        for c in range(0, 3):
            jpg_img[y1:y2, x1:x2, c] = ((alpha_jpg * jpg_img[y1:y2, x1:x2, c]) + (alpha_png * png_img[yy1:yy2, xx1:xx2, c]))

        return jpg_img

    def apply_mask(self, image_path):
        im_rd = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        img_gray = cv2.cvtColor(im_rd, cv2.COLOR_RGB2GRAY)
        faces = self.detector(img_gray, 0)

        if len(faces) != 0:
            for i in range(len(faces)):
                for k, d in enumerate(faces):
                    cv2.rectangle(im_rd, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0))
                    face = im_rd[d.top():d.bottom(), d.left():d.right()]
                    mask = cv2.resize(self.mask, (face.shape[1], face.shape[0]))
                    face_width = d.right() - d.left()
                    shape = self.predictor(im_rd, d)
                    mouth_width = (shape.part(54).x - shape.part(48).x) / face_width
                    mouth_higth = (shape.part(66).y - shape.part(62).y) / face_width
                    for i in range(68):
                        cv2.circle(im_rd, (shape.part(i).x, shape.part(i).y), 4, (0, 0, 255), -1, 2)
                    x1 = d.left()
                    y1 = d.top()
                    x2 = x1 + face.shape[1]
                    y2 = y1 + face.shape[0]
                    res_img = self.merge_img(im_rd, mask, y1, y1 + mask.shape[0], x1, x1 + mask.shape[1])
        cv2.imshow('out', res_img)
        cv2.waitKey(0)
