# Dataset
> Fill this form to ask for the [dataset](https://forms.gle/L4wMnZsEgJNpnUbf8).

## Dataset Request Form
Please fill out the form below to request access to the dataset used in this project.

## Prepare the dataset

### Option 1: From the original dataset
1. Download the original(slided) dataset.
2. (Optional) Slide the videos into clips with 4 seconds duration.
3. Extract the optical flow frames from the videos using the provided script.
```bash
python sg-mamba/tools/extract_flow_frame.py --video_path /path/to/slided/videos --save_path /path/to/save/flow/frames
```
4. Extract the I3D features from the videos and flow frames using the provided script.
5. Extract the skeleton features from the videos using the provided script.
```bash
python libs/utils/inference_keypoints_npy_api.py # alter the input_root, input_file, output_root inside the script as needed
```


### Option 2: From the preprocessed dataset
1. Download the preprocessed dataset.
2. Download the I3D features and skeleton features.
3. Place the I3D features and skeleton features in the appropriate directories.

Cite:
```bibtex
@article{ruan2023temporal,
  title={Temporal micro-action localization for videofluoroscopic swallowing study},
  author={Ruan, Xianghui and Dai, Meng and Chen, Zhuokun and You, Zeng and Zhang, Yaowen and Li, Yuanqing and Dou, Zulin and Tan, Mingkui},
  journal={IEEE Journal of Biomedical and Health Informatics},
  volume={27},
  number={12},
  pages={5904--5913},
  year={2023},
  publisher={IEEE}
}
```
