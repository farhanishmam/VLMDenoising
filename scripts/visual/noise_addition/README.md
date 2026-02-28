- Download the DARE Dataset's 1_correct subset's validation split images using the <a href="https://huggingface.co/datasets/cambridgeltl/DARE"> link </a> or you can use the <a href= "https://drive.google.com/drive/folders/1n32Cu6d2hEFt-ZSJgorlm7X2t4juQwSe?usp=sharing"> drive link </a>

- Download the VQA-v2 dataset from: https://visualqa.org/

- Organzize the directories accordingly.
- CLEAN_IMAGES_FOLDER will contain clean images (VQav2 subset of 3000 images otr DARE dataset - 657 images)
- NOISY_IMAGES_FOLDER will contain output augmented images (this is the output folder)


Augmented Dataset Generation Procedure:
------------------------------------------
1. For DARE image augmented dataset generation (Inference Dataset), run the noisy_main.py file as it is. Remember to adjust the folders accordingly.

2. For VQAv2 image augmented dataset generation, in the noisy_main.py file, uncomment the lines 51, 52, 53 and comment everything afterwards. Remember to adjust the folders accordingly.

3. Run the command: python noisy_main.py
