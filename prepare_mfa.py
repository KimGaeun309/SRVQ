import os
import yaml
import argparse

from jamo import h2j
from glob import glob
from tqdm import tqdm

def mfa_align(config):
    # ./raw_data/kss/folder/*.lab
    file_list = sorted(glob(os.path.join(config["path"]["raw_path"], '**/*.lab')))

    phoneme_dict = {}
    for file_name in tqdm(file_list):
        text = open(file_name, 'r', encoding='utf-8').readline().strip('\n')
        phoneme_list = h2j(text).split(' ')
        for i, phoneme in enumerate(phoneme_list):
            if phoneme not in phoneme_dict:
                phoneme_dict[phoneme] = ' '.join(phoneme_list[i])

    os.makedirs(config["path"]["mfa_path"], exist_ok=True)
    dict_txt = os.path.join(config["path"]["mfa_path"], 'kr_dict.txt')
    with open(dict_txt, 'w', encoding='utf-8') as fw:
        for key in phoneme_dict.keys():
            content = '{}\t{}\n'.format(key, phoneme_dict[key])
            fw.write(content)


def mfa_train(config):

    cpu_num = os.cpu_count()
    dataset = config["path"]["raw_path"]

    if not os.path.isdir(dataset):
        print(f'No exist dataset folder: {dataset}')
    else:
        # Path Setting && Make Folder
        os.makedirs(config["path"]["mfa_path"], exist_ok=True)
        dict_txt = os.path.join(config["path"]["mfa_path"], 'kr_dict.txt')
        g2p_model_output = os.path.join(config["path"]["mfa_path"], 'kr_g2p.zip')
        g2p_dict_txt = os.path.join(config["path"]["mfa_path"], 'kr_g2p_dict.txt')
        text_gird_folder = os.path.join(config["path"]["preprocessed_path"], 'TextGrid')
        mfa_model = os.path.join(config["path"]["mfa_path"], 'kr_acoustic.zip')

        # g2p train
        ## require text_full(aggregate), g2p model output path
        print("MFA train_g2p Start")
        os.system(f'mfa train_g2p {dict_txt} {g2p_model_output} -j {cpu_num} --clean')
        print("MFA train_g2p Finish")

        # g2p dict create
        ## Require g2p model, text dataset path, output dict.txt
        print("MFA G2P Start!")
        os.system(f'mfa g2p {dataset} {g2p_model_output} {g2p_dict_txt} -j {cpu_num} --clean')
        print("MFA G2P Finish!")

        # MFA Train
        ## Require Corpus path, g2p dict, Textgrid output path
        ### Model output path -o {path}
        os.makedirs(text_gird_folder, exist_ok=True)
        print('MFA Train Start!')
        # os.system(f'mfa train {dataset} {dict_txt} {mfa_model} --output_directory {text_gird_folder} -j {cpu_num} --clean')
        os.system(f'mfa train {dataset} {dict_txt} {mfa_model} -j {cpu_num} --clean')
        print('MFA Train Finish!')

        print("MFA Align Start!")
        os.system(f'mfa align {dataset} {dict_txt} {mfa_model} {text_gird_folder} -j {cpu_num} --clean')
        print("MFA Align Finish!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="name of dataset",
    )
    args = parser.parse_args()

    config_dir = os.path.join("./config", args.dataset)
    preprocess_config = yaml.load(open(
        os.path.join(config_dir, "preprocess.yaml"), "r"), Loader=yaml.FullLoader)

    # Make dict.txt
    print('MFA Align Start!')
    mfa_align(preprocess_config)
    print('MFA Align Finish!')

    # MFA Train
    mfa_train(preprocess_config)
