# -*- coding: utf-8 -*-
'''
constant definitions
'''
data_root_path = './data/'
#  save path
dir_path = data_root_path + "save"
all_word_cut_name = "all_comment_cut"
chi_feature_name = 'chi_features'
stop_path = data_root_path + 'stop.txt'

# label kind
pos, neg = 0, 1

# field name
label_name = 'label'
comment_name = 'comment'
pred_label_name = 'pred_label'



# file name
no_label_comment_path = data_root_path + 'no_label_comment.txt' # to predict data path
no_label_comment_pred_result_path = data_root_path + 'no_label_comment_pred_result.txt' # result of to predict data path

good_comment_path = data_root_path + 'good-comment.txt'
bad_comment_path = data_root_path + 'bad-comment.txt'

font_path = data_root_path+"msyh.ttf"