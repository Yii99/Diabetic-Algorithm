import tensorflow as tf

class ROC(object):
    def __init__(self):
        # self.pre_axis0_list = []
        # self.pre_axis1_list=[]
        self.pre_list = []
        self.label_list = []

    def update_state(self, labels, label_pred):
        # self.pre_axis0_list.extend(pred_axis0.numpy())
        # self.pre_axis1_list.extend(pred_axis1.numpy())
        self.pre_list.extend(label_pred.numpy())
        self.label_list.extend(labels.numpy())

        # print(self.pre_axis1_list)

    def result(self):
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        FPR_list = []
        TPR_list = []
        pred_axis0, pred_axis1 = tf.unstack(self.pre_list, axis=1)
        for j in range(100):  # 阈值
            for num, label in enumerate(self.label_list):
                if pred_axis1[num] * 100 >= j:
                    if label == 1:
                        TP = TP + 1
                    else:
                        FP = FP + 1
                elif pred_axis0[num] * 100 >= j:
                    if label == 0:
                        TN += 1
                    else:
                        FN += 1
            if FP + TN == 0:
                FPR = 0
            else:
                FPR = FP / (FP + TN)

            if TP + FN == 0:
                TPR = 0
            else:
                TPR = TP / (TP + FN)

            FPR_list.append(FPR)
            TPR_list.append(TPR)
            TP = 0
            TN = 0
            FP = 0
            FN = 0

        # print(FPR_list)

        return FPR_list, TPR_list

    def reset_states(self):
        # self.pre_axis0_list = []
        # self.pre_axis1_list=[]
        self.pre_list = []
        self.label_list = []


class C_M(object):
    def __init__(self, num_class):
        self.conf_matrix = tf.Variable(tf.zeros(shape=[num_class, num_class], dtype=tf.int32), trainable=False)
        self.num_class = num_class

    def update_state(self, labels, label_pred):
        #  print(self.conf_matrix)
        self.conf_matrix.assign_add(tf.math.confusion_matrix(labels, label_pred))

    def result(self):
        return self.conf_matrix

    def reset_states(self):
        self.conf_matrix = tf.Variable(tf.zeros(shape=[self.num_class, self.num_class], dtype=tf.int32),
                                       trainable=False)