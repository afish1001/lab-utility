import cv2
import matlab
import matlab.engine


class Evaluator():
    """
    Evaluation Class
    All the methods are driven from https://github.com/zhengliu6699/imageFusionMetrics
    and
    Liu Z, Blasch E, Xue Z, et al. Objective assessment of multiresolution image fusion algorithms
    for context enhancement in night vision: a comparative study[J]. IEEE transactions on pattern
    analysis and machine intelligence, 2011, 34(1): 94-109.
    There are four classes of evaluation method in the paper, we choose one metric in each class to
    calculate.
    """
    def __init__(self):
        self.engine = matlab.engine.start_matlab()

    @staticmethod
    def np_to_mat(img):
        """
        Transfer numpy to matlab style
        :param img: image, np.array
        :return: matlab style
        """
        img_mat = matlab.double(img.tolist())
        return img_mat

    def get_evaluation(self, img1, img2, fused):
        """
        get evaluation by four metrics: qmi, qg, qy, qcb, the bigger, the better
        :param img1: last image, np.array
        :param img2: next image, np.array
        :param fused: fusion result, np.array
        :return: qmi, qg, qy, qcb, float
        """
        # Transfer BGR to Gray
        # We have tested that qg, qy and qcb only support for gray mode
        if len(img1.shape) == 3:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            fused = cv2.cvtColor(fused, cv2.COLOR_BGR2GRAY)

        # Transfer numpy to mat
        img1_mat = self.np_to_mat(img1)
        img2_mat = self.np_to_mat(img2)
        fused_mat = self.np_to_mat(fused)

        # evaluation
        qmi = self.evaluate_by_qmi(img1_mat, img2_mat, fused_mat)
        qg = self.evaluate_by_qg(img1_mat, img2_mat, fused_mat)
        qy = self.evaluate_by_qy(img1_mat, img2_mat, fused_mat)
        qcb = self.evaluate_by_qcb(img1_mat, img2_mat, fused_mat)
        return qmi, qg, qy, qcb

    def evaluate_by_qmi(self, img1, img2, fused):
        """
        Normalized Mutual Information (QMI)
        As the paper described, the value is not accurate because it tends to become larger when the pixel
        values of the fused image are closer to one of the source images
        Liu Y, Chen X, Peng H, et al. Multi-focus image fusion with a deep convolutional neural network[J].
        Information Fusion, 2017, 36: 191-207.
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qmi
        """
        value = self.engine.metricMI(img1, img2, fused, 1)  # sw = 1 revised MI
        return value

    def evaluate_by_qg(self, img1, img2, fused):
        """
        Gradient-Based Fusion Performance(Qg): evaluate the amount of edge information
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qg
        """
        value = self.engine.metricXydeas(img1, img2, fused)
        return value

    def evaluate_by_qy(self, img1, img2, fused):
        """
        SSIM-Based Fusion Performance(Qy)
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: qy
        """
        value = self.engine.metricYang(img1, img2, fused)
        return value

    def evaluate_by_qcb(self, img1, img2, fused):
        """
        Human Perception inspired fusion metric - chen blum Metric
        Using DoG as contrast preservation filter
        :param img1: last image, matlab mode
        :param img2: next image, matlab mode
        :param fused: fusion result, matlab mode
        :return: Qcb
        """
        value = self.engine.metricChenBlum(img1, img2, fused)
        return value


if __name__=="__main__":
    evaluator = Evaluator()
    img1 = cv2.imread("lytro-20-A.jpg")
    img2 = cv2.imread("lytro-20-B.jpg")
    fused = cv2.imread("lytro-20-C.jpg")    # For testing, C is actually A

    qmi, qg, qy, qcb = evaluator.get_evaluation(img1, img2, fused)
    print("Qmi is {}, Qg is {}, Qy is {}, Qcb is {}".format(qmi, qg, qy, qcb))