# *_*coding:utf-8 *_*
import os

## 所有涉及 transformers 的模型存储路径
AFFECTGPT_ROOT = './'
RESULT_ROOT = os.path.join(AFFECTGPT_ROOT, 'output/results')


#######################
## 所有模型的存储路径
#######################
PATH_TO_LLM = {
    'Qwen25': 'models/Qwen2.5-7B-Instruct',
}

PATH_TO_VISUAL = {
    'CLIP_VIT_LARGE': 'models/clip-vit-large-patch14',

}

PATH_TO_AUDIO = {
    'HUBERT_LARGE':  'models/chinese-hubert-large',
}

#######################
## 所有数据集的存储路径
#######################
DATA_DIR = {
    'BAH':           '/datasets/BAH-process', 
}

PATH_TO_RAW_AUDIO = {
    'BAH': os.path.join(DATA_DIR['BAH'], 'subaudio'),
}
PATH_TO_RAW_VIDEO = {
    'BAH': os.path.join(DATA_DIR['BAH'], 'subvideo'),
}
PATH_TO_RAW_FACE = {
    'BAH': os.path.join(DATA_DIR['BAH'], 'openface_face'),
}
PATH_TO_TRANSCRIPTIONS = {
    'BAH': os.path.join(DATA_DIR['BAH'], 'transcription.csv'),
}
PATH_TO_LABEL = {
    'BAH': os.path.join(DATA_DIR['BAH'], 'label.npz'),
}


#######################
## store global values
#######################
DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
DEFAULT_FRAME_PATCH_TOKEN = '<FrameHere>'
DEFAULT_FACE_PATCH_TOKEN  = '<FaceHere>'
DEFAULT_MULTI_PATCH_TOKEN = '<MultiHere>'
IGNORE_INDEX = -100
